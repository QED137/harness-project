"""Step 5: the verifier. Plain Python, no LLM.

Checks a submitted answer before it is accepted. Each check ends as PASS, FAIL,
WARN or SKIP. Any FAIL rejects the answer and the reasons go back to the model.
WARN is recorded for the evaluation but never blocks, because a verifier that is
too strict rejects correct answers (and the evaluation measures exactly that).

Known limitation: range checks catch gross errors only (e.g. Pa labelled as hPa).
They cannot tell 3.5 km/h from 3.5 m/s, since both are plausible values.
"""

import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .schemas import SubmitAnswerArgs

NOT_ANSWERABLE = "not answerable"

# canonical unit -> (aliases, maximum plausible absolute value)
UNITS: dict[str, tuple[tuple[str, ...], float]] = {
    "°C": (("°c", "c", "degc", "deg c", "celsius", "degrees celsius", "° c"), 60),
    "K": (("k", "kelvin"), 350),
    "°F": (("°f", "f", "degf", "fahrenheit"), 140),
    "%": (("percent", "pct"), 100),
    "mm": (("millimetre", "millimeter", "millimetres", "millimeters"), 20_000),
    "cm": (("centimetre", "centimeter"), 2_000),
    "in": (("inch", "inches"), 800),
    "km/h": (("kmh", "km h-1", "kph", "km/hr"), 300),
    "m/s": (("ms-1", "m s-1", "m/sec", "mps"), 85),
    "kn": (("knot", "knots", "kt"), 165),
    "mph": ((), 190),
    "hPa": (("hpa", "mbar", "mb", "millibar"), 1_100),
    "Pa": (("pa",), 110_000),
    "kPa": (("kpa",), 110),
    "hours": (("h", "hr", "hrs", "hour"), 100_000),
    "days": (("d", "day"), 5_000),
}
TIME_DENOMINATORS = {"h", "hr", "hour", "day", "d", "week", "month", "year", "yr", "a", "decade"}
_ALIASES = {alias: canon for canon, (aliases, _) in UNITS.items() for alias in (canon.lower(), *aliases)}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class CheckStatus(StrEnum):
    PASS = "pass"  # noqa: S105 (a check status, not a password)
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class Check:
    name: str
    status: CheckStatus
    detail: str = ""


@dataclass
class Verdict:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status is not CheckStatus.FAIL for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    def message_for_model(self) -> str:
        lines = ["Error: the answer was not accepted by the verifier:"]
        lines += [f"- {c.name}: {c.detail}" for c in self.failures()]
        lines.append("Fix the problem (compute any conversion with run_python) and call submit_answer again.")
        return "\n".join(lines)


@dataclass
class Evidence:
    """What run_python actually produced during this run."""

    numbers: list[float] = field(default_factory=list)
    text: str = ""  # all results and stdout, lower-cased, for matching text answers
    saw_nan: bool = False
    n_runs: int = 0


# ------------------------------------------------------------------ units
def normalize_unit(unit: str | None) -> tuple[str | None, bool]:
    """Return (canonical unit, is_rate). Unknown units return (None, False)."""
    if unit is None:
        return None, False
    u = " ".join(unit.strip().lower().split())
    if u in _ALIASES:
        return _ALIASES[u], False
    for sep in ("/", " per "):
        if sep in u:
            head, _, tail = u.partition(sep)
            if head.strip() in _ALIASES and tail.strip() in TIME_DENOMINATORS:
                return f"{_ALIASES[head.strip()]}/{tail.strip()}", True
    return None, False


# ------------------------------------------------------------------ evidence
def collect_evidence(sandbox_results: list[Any]) -> Evidence:
    """Gather numbers and text from SandboxResult objects of successful run_python calls."""
    ev = Evidence()
    texts = []
    for r in sandbox_results:
        if r is None or r.status.value != "ok":
            continue
        ev.n_runs += 1
        res = r.result or {}
        if res.get("type") == "number":
            if res.get("is_nan") or res.get("is_inf"):
                ev.saw_nan = True
            elif res.get("value") is not None:
                ev.numbers.append(float(res["value"]))
        if res.get("n_missing"):
            ev.saw_nan = True
        blob = " ".join(str(res.get(k, "")) for k in ("value", "head", "repr")) + " " + (r.stdout or "")
        texts.append(blob)
        ev.numbers += [float(x) for x in _NUMBER.findall(blob)]
    ev.text = " ".join(texts).lower()
    return ev


def _decimals(literal: str) -> int | None:
    """Decimal places of a number as the model wrote it; None for exponent notation."""
    if "e" in literal.lower():
        return None
    return len(literal.split(".")[1]) if "." in literal else 0


def _matches(value: float, literal: str, observed: float) -> bool:
    """True if `value` equals `observed` up to the rounding the model applied."""
    d = _decimals(literal)
    tol = (0.5 * 10.0 ** (-d)) if d is not None else 1e-6 * abs(observed)
    return abs(value - observed) <= tol + 1e-9 * abs(observed)


def submitted_literal(raw_arguments: str, value: float | str) -> str:
    """The value exactly as written in the JSON (so '11' and '11.0' keep their precision)."""
    try:
        raw = json.loads(raw_arguments, parse_float=str, parse_int=str)
        return str(raw.get("value", value))
    except (json.JSONDecodeError, AttributeError):
        return repr(value)


# ------------------------------------------------------------------ verify
def verify(
    answer: SubmitAnswerArgs,
    evidence: Evidence,
    raw_arguments: str = "",
    expected_unit: str | None = None,
) -> Verdict:
    v = Verdict()
    value, unit = answer.value, answer.unit

    if isinstance(value, str) and value.strip().lower() == NOT_ANSWERABLE:
        v.checks.append(Check("not_answerable", CheckStatus.PASS, "agent declined to answer"))
        return v

    is_number = isinstance(value, float | int)

    # 1. type
    if unit is not None and not is_number:
        v.checks.append(Check("type", CheckStatus.FAIL, f"a value with a unit ({unit}) must be a number"))
    else:
        v.checks.append(Check("type", CheckStatus.PASS))

    # 2. unit
    canon, is_rate = normalize_unit(unit)
    if unit is None:
        v.checks.append(Check("unit", CheckStatus.SKIP, "no unit"))
    elif canon is None:
        v.checks.append(Check("unit", CheckStatus.WARN, f"unrecognised unit '{unit}'"))
    else:
        v.checks.append(Check("unit", CheckStatus.PASS, canon))

    # 3. range
    if is_number and canon in UNITS and not is_rate:
        limit = UNITS[canon][1]
        if not math.isfinite(float(value)) or abs(float(value)) > limit:
            v.checks.append(
                Check(
                    "range", CheckStatus.FAIL, f"{value} {canon} is implausible (|value| must be ≤ {limit:g})"
                )
            )
        else:
            v.checks.append(Check("range", CheckStatus.PASS))
    else:
        v.checks.append(Check("range", CheckStatus.SKIP))

    # 4. provenance
    if evidence.n_runs == 0:
        v.checks.append(
            Check("provenance", CheckStatus.FAIL, "no successful run_python call; compute the answer")
        )
    elif is_number:
        literal = submitted_literal(raw_arguments, value)
        if any(_matches(float(value), literal, o) for o in evidence.numbers):
            v.checks.append(Check("provenance", CheckStatus.PASS))
        else:
            v.checks.append(
                Check(
                    "provenance",
                    CheckStatus.FAIL,
                    f"{literal} does not match any number produced by run_python in this run",
                )
            )
    elif str(value).strip().lower() in evidence.text:
        v.checks.append(Check("provenance", CheckStatus.PASS))
    else:
        v.checks.append(
            Check(
                "provenance",
                CheckStatus.FAIL,
                f"'{value}' does not appear in any run_python output of this run",
            )
        )

    # 5. silent NaN (warn only)
    summary = answer.method_summary.lower()
    if evidence.saw_nan and not any(w in summary for w in ("nan", "missing", "null")):
        v.checks.append(
            Check("silent_nan", CheckStatus.WARN, "NaN/missing values appeared but are not mentioned")
        )
    else:
        v.checks.append(Check("silent_nan", CheckStatus.PASS))

    # 6. unit expected by the plan (warn only: the plan can be wrong)
    if expected_unit is not None and unit is not None:
        exp_canon, _ = normalize_unit(expected_unit)
        if exp_canon is not None and canon is not None and exp_canon != canon:
            v.checks.append(
                Check("plan_unit", CheckStatus.WARN, f"planner expected {exp_canon}, got {canon}")
            )
        else:
            v.checks.append(Check("plan_unit", CheckStatus.PASS))
    return v
