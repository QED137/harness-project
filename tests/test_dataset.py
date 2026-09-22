"""Dataset download, validation and freezing, with a fake API response. No network needed."""

import json

import pandas as pd
import pytest

from tsagent import dataset as ds


def fake_payload(start="2020-01-01", days=2, tz="GMT", drop_hour=None, dup_hour=None, none_at=None):
    times = pd.date_range(start, periods=24 * days, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    if drop_hour is not None:
        times.pop(drop_hour)
    if dup_hour is not None:
        times.insert(dup_hour, times[dup_hour])
    n = len(times)
    hourly = {"time": times}
    for i, v in enumerate(ds.VARIABLES):
        values = [float(i * 10 + k % 24) for k in range(n)]
        hourly[v] = values
    if none_at is not None:
        hourly["temperature_2m"][none_at] = None
    return {
        "latitude": 48.5,
        "longitude": 9.0625,
        "elevation": 341.0,
        "timezone": tz,
        "hourly_units": {"time": "iso8601", **{v: "unit" for v in ds.VARIABLES}},
        "hourly": hourly,
    }


REQUEST = {"latitude": 48.52, "longitude": 9.06, "start_date": "2020-01-01", "end_date": "2020-01-02"}


def test_url_contains_request():
    url = ds.build_url(48.52, 9.06, "2020-01-01", "2024-12-31")
    assert url.startswith(ds.API_URL + "?")
    for part in ("latitude=48.52", "start_date=2020-01-01", "end_date=2024-12-31", "timezone=GMT"):
        assert part in url
    assert "hourly=" + ",".join(ds.VARIABLES) in url


def test_parse_gives_utc_float_frame():
    df = ds.to_dataframe(fake_payload())
    assert str(df.index.tz) == "UTC"
    assert list(df.columns) == list(ds.VARIABLES)
    assert all(str(t) == "float64" for t in df.dtypes)


def test_null_values_become_nan_and_are_counted():
    df = ds.to_dataframe(fake_payload(none_at=5))
    report = ds.validate(df, "2020-01-01", "2020-01-02")
    assert report["nan_counts"]["temperature_2m"] == 1


def test_local_timezone_rejected():
    with pytest.raises(ValueError, match="GMT/UTC"):
        ds.to_dataframe(fake_payload(tz="Europe/Berlin"))


def test_missing_variable_rejected():
    p = fake_payload()
    del p["hourly"]["precipitation"]
    with pytest.raises(ValueError, match="precipitation"):
        ds.to_dataframe(p)


def test_complete_data_validates():
    report = ds.validate(ds.to_dataframe(fake_payload()), "2020-01-01", "2020-01-02")
    assert report["rows"] == report["expected_rows"] == 48


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"drop_hour": 10}, "missing hours"),
        ({"dup_hour": 10}, "duplicated"),
        ({"days": 3}, "unexpected timestamps"),
    ],
)
def test_structural_problems_detected(kwargs, message):
    df = ds.to_dataframe(fake_payload(**kwargs))
    with pytest.raises(ValueError, match=message):
        ds.validate(df, "2020-01-01", "2020-01-02")


def test_content_hash_is_stable_and_sensitive():
    df = ds.to_dataframe(fake_payload())
    assert ds.content_hash(df) == ds.content_hash(df.copy())
    changed = df.copy()
    changed.iloc[0, 0] += 0.01
    assert ds.content_hash(changed) != ds.content_hash(df)


def test_save_writes_readable_files_and_manifest(tmp_path):
    out = tmp_path / "data"
    manifest = ds.save(ds.to_dataframe(fake_payload()), fake_payload(), out, REQUEST)
    assert (out / ds.PARQUET_NAME).stat().st_mode & 0o777 == 0o644
    assert out.stat().st_mode & 0o777 == 0o755
    on_disk = json.loads((out / ds.MANIFEST_NAME).read_text())
    assert on_disk == manifest
    assert manifest["rows"] == 48 and manifest["timezone"] == "UTC"
    assert manifest["grid_cell"]["elevation"] == 341.0


def test_invalid_data_is_never_written(tmp_path):
    out = tmp_path / "data"
    with pytest.raises(ValueError):
        ds.save(ds.to_dataframe(fake_payload(drop_hour=3)), fake_payload(), out, REQUEST)
    assert not (out / ds.PARQUET_NAME).exists()


def test_check_passes_then_detects_tampering(tmp_path):
    out = tmp_path / "data"
    ds.save(ds.to_dataframe(fake_payload()), fake_payload(), out, REQUEST)
    assert ds.check(out)[0] is True
    df = pd.read_parquet(out / ds.PARQUET_NAME)
    df.iloc[0, 0] = 999.0
    df.to_parquet(out / ds.PARQUET_NAME)
    ok, message = ds.check(out)
    assert ok is False and "mismatch" in message


def test_check_without_files_explains_what_to_do(tmp_path):
    ok, message = ds.check(tmp_path)
    assert ok is False and "python -m tsagent.dataset" in message


def test_main_end_to_end_with_fake_download(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "download", lambda url: fake_payload())
    out = tmp_path / "data"
    assert ds.main(["--start", "2020-01-01", "--end", "2020-01-02", "--out", str(out)]) == 0
    assert ds.main(["--check", "--out", str(out)]) == 0


def test_download_refuses_other_urls():
    with pytest.raises(ValueError, match="refusing"):
        ds.download("file:///etc/passwd")
