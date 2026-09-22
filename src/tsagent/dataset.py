"""Step 1: download, validate and freeze the dataset.

    python -m tsagent.dataset             download from Open-Meteo, validate, write data/
    python -m tsagent.dataset --check     verify data/weather.parquet against data/manifest.json

Source: Open-Meteo Historical Weather API (reanalysis), hourly, stored in UTC.
UTC avoids the missing (March) and duplicated (October) local hours caused by
daylight saving time.

The manifest records exactly which data every later evaluation run used. Its
`content_sha256` hashes the data values, not the parquet bytes, so it does not
change when only the pyarrow version differs.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
)
DEFAULT_LAT, DEFAULT_LON = 48.52, 9.06  # Tübingen
DEFAULT_START, DEFAULT_END = "2020-01-01", "2024-12-31"
DEFAULT_OUT = Path("data")
PARQUET_NAME = "weather.parquet"
MANIFEST_NAME = "manifest.json"
ATTRIBUTION = "Weather data by Open-Meteo.com (https://open-meteo.com/), CC BY 4.0"


# ------------------------------------------------------------------ download
def build_url(lat: float, lon: float, start: str, end: str, variables=VARIABLES) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "GMT",
    }
    return API_URL + "?" + urllib.parse.urlencode(params, safe=",")


def download(url: str, timeout: float = 120.0) -> dict[str, Any]:
    # urlopen also accepts file:// and other schemes; only ever open the Open-Meteo API.
    if not url.startswith(API_URL + "?"):
        raise ValueError(f"refusing to open unexpected URL: {url[:80]}")
    request = urllib.request.Request(url, headers={"User-Agent": "tsagent-dataset/0.1"})  # noqa: S310 (checked above)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (checked above)
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            reason = json.loads(body).get("reason", body)
        except json.JSONDecodeError:
            reason = body
        raise RuntimeError(f"Open-Meteo returned HTTP {e.code}: {reason}") from e


# ------------------------------------------------------------------ parse + validate
def to_dataframe(payload: dict[str, Any], variables=VARIABLES) -> pd.DataFrame:
    tz = payload.get("timezone")
    if tz not in ("GMT", "UTC"):
        raise ValueError(f"expected timestamps in GMT/UTC, API returned timezone={tz!r}")
    hourly = payload.get("hourly") or {}
    missing = [v for v in ("time", *variables) if v not in hourly]
    if missing:
        raise ValueError(f"API response is missing hourly fields: {missing}")
    index = pd.to_datetime(hourly["time"], utc=True)
    df = pd.DataFrame({v: hourly[v] for v in variables}, index=index).astype("float64")
    df.index.name = "time"
    return df


def validate(df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    """Structural checks that must hold exactly. Raises ValueError on failure.
    NaNs are counted and reported, not fatal."""
    expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    index = pd.DatetimeIndex(df.index)
    problems = []
    n_dup = int(index.duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicated timestamps")
    if not index.is_monotonic_increasing:
        problems.append("timestamps are not sorted")
    missing = expected.difference(index)
    if len(missing):
        problems.append(f"{len(missing)} missing hours, first: {missing[0]}")
    extra = index.difference(expected)
    if len(extra):
        problems.append(f"{len(extra)} unexpected timestamps, first: {extra[0]}")
    all_nan = [c for c in df.columns if df[c].isna().all()]
    if all_nan:
        problems.append(f"columns entirely NaN: {all_nan}")
    if problems:
        raise ValueError("dataset failed validation: " + "; ".join(problems))
    return {
        "rows": len(df),
        "expected_rows": len(expected),
        "nan_counts": {c: int(n) for c, n in df.isna().sum().items()},
    }


def content_hash(df: pd.DataFrame) -> str:
    """Hash of the values in a canonical text form, independent of parquet/pyarrow versions."""
    canonical = df.to_csv(float_format="%.6f", date_format="%Y-%m-%dT%H:%M:%SZ", lineterminator="\n")
    return hashlib.sha256(canonical.encode()).hexdigest()


# ------------------------------------------------------------------ save + check
def save(df: pd.DataFrame, payload: dict[str, Any], out_dir: Path, request: dict[str, Any]) -> dict:
    # Validate BEFORE writing anything, so a bad download never replaces good data.
    report = validate(df, request["start_date"], request["end_date"])
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / PARQUET_NAME
    df.to_parquet(parquet)
    # The sandbox container runs as UID 65534, not as you: data must be readable by "others".
    out_dir.chmod(0o755)
    parquet.chmod(0o644)

    units = payload.get("hourly_units", {})
    manifest = {
        "source": "Open-Meteo Historical Weather API (reanalysis)",
        "attribution": ATTRIBUTION,
        "api_url": API_URL,
        "request": request,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "grid_cell": {k: payload.get(k) for k in ("latitude", "longitude", "elevation")},
        "timezone": "UTC",
        "variables": {v: units.get(v) for v in df.columns},
        "time_range": [df.index[0].isoformat(), df.index[-1].isoformat()],
        **report,
        "content_sha256": content_hash(df),
        "file": PARQUET_NAME,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check(out_dir: Path) -> tuple[bool, str]:
    manifest_path, parquet = out_dir / MANIFEST_NAME, out_dir / PARQUET_NAME
    if not manifest_path.is_file():
        return False, f"{manifest_path} not found; run: python -m tsagent.dataset"
    if not parquet.is_file():
        return False, f"{parquet} not found; run: python -m tsagent.dataset"
    manifest = json.loads(manifest_path.read_text())
    actual = content_hash(pd.read_parquet(parquet))
    if actual != manifest["content_sha256"]:
        return False, (
            f"content hash mismatch: manifest {manifest['content_sha256'][:12]}..., "
            f"file {actual[:12]}... The data differs from the version the manifest describes."
        )
    return True, f"ok: {manifest['rows']} rows, content_sha256 {actual[:12]}..."


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tsagent.dataset", description=__doc__.split("\n\n")[0])
    p.add_argument("--lat", type=float, default=DEFAULT_LAT)
    p.add_argument("--lon", type=float, default=DEFAULT_LON)
    p.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD, inclusive")
    p.add_argument("--end", default=DEFAULT_END, help="YYYY-MM-DD, inclusive")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--check", action="store_true", help="verify local data against the manifest, no download")
    args = p.parse_args(argv)

    if args.check:
        ok, message = check(args.out)
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    url = build_url(args.lat, args.lon, args.start, args.end)
    print(f"downloading {url}")
    payload = download(url)
    df = to_dataframe(payload)
    request = {"latitude": args.lat, "longitude": args.lon, "start_date": args.start, "end_date": args.end}
    manifest = save(df, payload, args.out, request)
    print(f"wrote {args.out / PARQUET_NAME}: {manifest['rows']} rows, {manifest['time_range']}")
    print(f"variables: {manifest['variables']}")
    print(f"NaN counts: {manifest['nan_counts']}")
    print(f"content_sha256: {manifest['content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
