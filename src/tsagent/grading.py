"""Compare one agent answer with the ground truth of a question.

Grading is deliberately generous about FORM and strict about SUBSTANCE:
  - an answer given in an equivalent unit is converted first (m/s vs km/h),
  - "July 2023", "2023-07" and "2023-07-15" all identify the month 2023-07,
but the numeric value must be within the question's tolerance, and an answer to a
not-answerable question is only correct if the agent declined.
"""

import math
import re
from dataclasses import dataclass

import pandas as pd

from .questions import NOT_ANSWERABLE, Question
from .schemas import SubmitAnswerArgs
from .verifier import normalize_unit

MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
# value in base unit = (value + offset_before) * factor + offset_after
_TEMPERATURE = {"°C": (0.0, 1.0, 0.0), "K": (-273.15, 1.0, 0.0), "°F": (-32.0, 5 / 9, 0.0)}
_LINEAR = {
    "speed": {"m/s": 1.0, "km/h": 1 / 3.6, "kn": 0.514444, "mph": 0.44704},
    "length": {"mm": 1.0, "cm": 10.0, "in": 25.4},
    "pressure": {"hPa": 1.0, "Pa": 0.01, "kPa": 10.0},
    "percent": {"%": 1.0},
}
# "hours" and "days" are used as labels for COUNTS of records or days, not as durations,
# so they are never rescaled: 90 days of frost is not 2160 hours of frost.
COUNT_UNITS = {"hours", "days"}
# Words a model may use for the same count. "hourly records" == hours for these questions.
COUNT_SYNONYMS = {
    "records",
    "record",
    "hourly records",
    "count",
    "counts",
    "observations",
    "observation",
    "rows",
    "row",
    "entries",
    "entry",
    "values",
    "data points",
    "datapoints",
    "samples",
    "measurements",
    "occurrences",
}


@dataclass
class Grade:
    correct: bool
    reason: str
    expected: float | str
    got: float | str | None = None
    unit_converted: bool = False


def _to_base(value: float, unit: str) -> tuple[float, str] | None:
    if unit in _TEMPERATURE:
        before, factor, after = _TEMPERATURE[unit]
        return (value + before) * factor + after, "temperature"
    for group, units in _LINEAR.items():
        if unit in units:
            return value * units[unit], group
    return None


def convert(value: float, from_unit: str | None, to_unit: str | None) -> float | None:
    """Convert between equivalent units. None if the units are not convertible."""
    a, _ = normalize_unit(from_unit)
    b, _ = normalize_unit(to_unit)
    if a is None or b is None:
        return None
    if a == b:
        return value
    base_a, base_b = _to_base(value, a), _to_base(1.0, b)
    if base_a is None or base_b is None or base_a[1] != base_b[1]:
        return None
    if base_a[1] == "temperature":
        before, factor, after = _TEMPERATURE[b]
        return (base_a[0] - after) / factor - before
    return base_a[0] / base_b[0]


def as_number(value: float | str) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "")
    match = re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return float(text) if match else None


def canonical_text(value: str) -> str:
    """Canonical form of a date-like or numeric text answer."""
    text = " ".join(str(value).strip().lower().split())
    for name, number in MONTHS.items():
        # "july 2023" / "2023 july" -> "2023-07"
        for pattern, order in ((rf"{name}\s+(\d{{4}})", "my"), (rf"(\d{{4}})\s+{name}", "ym")):
            m = re.fullmatch(pattern, text)
            if m:
                return f"{m.group(1)}-{number:02d}" if order in ("my", "ym") else text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}(:\d{2})?", text):
        return pd.Timestamp(text).strftime("%Y-%m-%d %H:%M")
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    if re.fullmatch(r"\d{4}-\d{1,2}", text):
        year, month = text.split("-")
        return f"{year}-{int(month):02d}"
    return text


def text_matches(expected: str, got: str) -> bool:
    e, g = canonical_text(expected), canonical_text(got)
    if e == g:
        return True
    # a day or timestamp also identifies its month, when a month was asked for
    if re.fullmatch(r"\d{4}-\d{2}", e) and g.startswith(e):
        return True
    return bool(re.fullmatch(r"\d{4}", e) and g.startswith(e))


def grade(question: Question, expected: float | str, answer: SubmitAnswerArgs | None) -> Grade:
    if answer is None:
        return Grade(False, "no answer was submitted", expected)

    declined = isinstance(answer.value, str) and NOT_ANSWERABLE in answer.value.strip().lower()

    if isinstance(expected, str) and expected == NOT_ANSWERABLE:
        if declined:
            return Grade(True, "correctly declined", expected, answer.value)
        return Grade(False, "answered a question the dataset cannot answer", expected, answer.value)
    if declined:
        return Grade(False, "declined an answerable question", expected, answer.value)

    if isinstance(expected, str):
        expected_number = as_number(expected)
        got_as_number = as_number(answer.value)
        if expected_number is not None and got_as_number is not None:
            # e.g. "which hour of the day": truth "15", the agent may answer 15 or 15.0
            if abs(got_as_number - expected_number) <= question.tolerance:
                return Grade(True, "matches", expected, got_as_number)
            return Grade(False, f"expected {expected}, got {got_as_number:g}", expected, got_as_number)
        got = str(answer.value)
        if text_matches(expected, got):
            return Grade(True, "matches", expected, got)
        return Grade(False, f"expected {expected}, got {got}", expected, got)

    got_number = as_number(answer.value)
    if got_number is None:
        return Grade(False, f"expected a number, got '{answer.value}'", expected, str(answer.value))

    converted = False
    if question.unit in COUNT_UNITS:
        given = " ".join((answer.unit or "").strip().lower().split())
        canon, _ = normalize_unit(answer.unit)
        acceptable = answer.unit is None or canon == question.unit or given in COUNT_SYNONYMS
        if not acceptable:
            return Grade(False, f"unit '{answer.unit}' does not count {question.unit}", expected, got_number)
    elif question.unit is not None and answer.unit is not None:
        a, _ = normalize_unit(answer.unit)
        b, _ = normalize_unit(question.unit)
        if a != b:
            in_expected_unit = convert(got_number, answer.unit, question.unit)
            if in_expected_unit is None:
                return Grade(
                    False,
                    f"unit '{answer.unit}' is not convertible to '{question.unit}'",
                    expected,
                    got_number,
                )
            got_number, converted = in_expected_unit, True

    if not math.isfinite(got_number):
        return Grade(False, "answer is not a finite number", expected, got_number, converted)
    difference = abs(got_number - expected)
    if difference <= question.tolerance:
        return Grade(True, "within tolerance", expected, got_number, converted)
    return Grade(
        False,
        f"off by {difference:.4g} (tolerance {question.tolerance:g} {question.unit or ''})".strip(),
        expected,
        got_number,
        converted,
    )
