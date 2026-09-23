"""Grading: unit conversion, text answers, tolerances, declining. No LLM, no data."""

import math

import pytest
from pydantic import ValidationError

from tsagent.grading import Grade, as_number, canonical_text, convert, grade
from tsagent.questions import BY_ID, NOT_ANSWERABLE, Question
from tsagent.schemas import SubmitAnswerArgs

NUMERIC = Question("q", "mean temperature in °C?", "°C", lambda df: 0.0, tolerance=0.01)
WIND = Question("w", "mean wind speed in km/h?", "km/h", lambda df: 0.0, tolerance=0.01)
TEXT = BY_ID["warmest_month_2023"]
NONE_Q = BY_ID["snow_depth"]


def answer(value, unit=None, summary="m"):
    return SubmitAnswerArgs(value=value, unit=unit, method_summary=summary)


# ------------------------------------------------------------------ conversion
@pytest.mark.parametrize(
    "value, frm, to, expected",
    [
        (36.0, "km/h", "m/s", 10.0),
        (10.0, "m/s", "km/h", 36.0),
        (0.0, "°C", "K", 273.15),
        (273.15, "K", "°C", 0.0),
        (100.0, "°C", "°F", 212.0),
        (32.0, "°F", "°C", 0.0),
        (1013.25, "hPa", "Pa", 101325.0),
        (2.54, "cm", "mm", 25.4),
        (5.0, "degC", "°C", 5.0),  # alias
    ],
)
def test_convert(value, frm, to, expected):
    assert convert(value, frm, to) == pytest.approx(expected)


@pytest.mark.parametrize(
    "frm, to",
    [
        ("°C", "mm"),
        ("km/h", "hPa"),
        ("furlongs", "mm"),
        ("mm", None),
        ("days", "hours"),  # counts are never rescaled: see COUNT_UNITS
    ],
)
def test_incompatible_units_do_not_convert(frm, to):
    assert convert(1.0, frm, to) is None


# ------------------------------------------------------------------ numeric answers
def test_correct_number():
    g = grade(NUMERIC, 11.3139, answer(11.3139, "°C"))
    assert g.correct and not g.unit_converted


@pytest.mark.parametrize("value, correct", [(11.3139, True), (11.32, True), (11.30, False), (12.0, False)])
def test_tolerance_is_respected(value, correct):
    assert grade(NUMERIC, 11.3139, answer(value, "°C")).correct is correct


def test_equivalent_unit_is_converted_and_accepted():
    g = grade(WIND, 7.2, answer(2.0, "m/s"))  # 2 m/s = 7.2 km/h
    assert g.correct and g.unit_converted and g.got == pytest.approx(7.2)


def test_wrong_value_in_another_unit_is_still_wrong():
    assert not grade(WIND, 7.2, answer(2.5, "m/s")).correct


def test_non_convertible_unit_is_wrong_and_says_so():
    g = grade(WIND, 7.2, answer(7.2, "mm"))
    assert not g.correct and "not convertible" in g.reason


def test_number_sent_as_text_is_accepted():
    assert grade(NUMERIC, 11.3139, answer("11.3139", "°C")).correct
    assert as_number("11.31") == 11.31 and as_number("eleven") is None


def test_text_answer_to_a_numeric_question_is_wrong():
    g = grade(NUMERIC, 11.3, answer("about eleven", "°C"))
    assert not g.correct and "expected a number" in g.reason


def test_missing_unit_on_the_answer_is_taken_at_face_value():
    assert grade(NUMERIC, 11.3139, answer(11.3139, None)).correct


# ------------------------------------------------------------------ text answers
@pytest.mark.parametrize(
    "given, correct",
    [
        ("2023-07", True),
        ("2023-7", True),
        ("July 2023", True),
        ("july 2023", True),
        ("2023 July", True),
        ("2023-07-15", True),  # a day inside the right month identifies it
        ("2023-08", False),
        ("July", False),
    ],
)
def test_month_answers(given, correct):
    assert grade(TEXT, "2023-07", answer(given)).correct is correct


def test_timestamp_answers():
    q = BY_ID["hottest_hour_timestamp_2022"]
    assert grade(q, "2022-06-18 19:00", answer("2022-06-18 19:00")).correct
    assert grade(q, "2022-06-18 19:00", answer("2022-06-18T19:00:00")).correct
    assert not grade(q, "2022-06-18 19:00", answer("2022-06-18 18:00")).correct


def test_year_answers():
    q = BY_ID["warmest_year"]
    assert grade(q, "2022", answer("2022")).correct
    assert grade(q, "2022", answer(2022)).correct  # sent as a number
    assert not grade(q, "2022", answer("2021")).correct


def test_hour_of_day_answer_as_number_or_text():
    q = BY_ID["warmest_hour_of_day_2023"]
    assert grade(q, "15", answer(15)).correct
    assert grade(q, "15", answer("15")).correct
    assert not grade(q, "15", answer(14)).correct


def test_canonical_text():
    assert canonical_text(" July 2023 ") == "2023-07"
    assert canonical_text("2023-7") == "2023-07"
    assert canonical_text("2022-06-18T19:00:00") == "2022-06-18 19:00"


# ------------------------------------------------------------------ declining
def test_declining_an_unanswerable_question_is_correct():
    g = grade(NONE_Q, NOT_ANSWERABLE, answer("not answerable"))
    assert g.correct and g.reason == "correctly declined"


def test_answering_an_unanswerable_question_is_wrong():
    g = grade(NONE_Q, NOT_ANSWERABLE, answer(12.0, "cm"))
    assert not g.correct and "cannot answer" in g.reason


def test_declining_an_answerable_question_is_wrong():
    g = grade(NUMERIC, 11.3, answer("not answerable"))
    assert not g.correct and "declined" in g.reason


# ------------------------------------------------------------------ no answer
def test_missing_answer_is_wrong_not_an_error():
    g = grade(NUMERIC, 11.3, None)
    assert isinstance(g, Grade) and not g.correct and "no answer" in g.reason


def test_nan_and_infinity_never_reach_the_grader():
    """They are rejected one layer earlier, by the tool argument schema."""
    for bad in (math.inf, math.nan):
        with pytest.raises(ValidationError):
            answer(bad, "°C")


# ------------------------------------------------------------------ counts
COUNT_Q = BY_ID["hours_in_2023"]  # "how many hourly records ...", unit "hours"
DAYS_Q = BY_ID["frost_days_2021"]  # unit "days"


@pytest.mark.parametrize(
    "unit", [None, "hours", "records", "hourly records", "count", "observations", "rows"]
)
def test_count_answers_accept_count_words(unit):
    """Regression: the first real evaluation marked correct answers wrong because the
    agent labelled 8760 as 'records' instead of 'hours'."""
    assert grade(COUNT_Q, 8760.0, answer(8760, unit)).correct


@pytest.mark.parametrize("unit", ["days", "°C", "mm"])
def test_count_answers_reject_other_units(unit):
    g = grade(COUNT_Q, 8760.0, answer(8760, unit))
    assert not g.correct and "does not count hours" in g.reason


def test_counts_are_never_rescaled():
    """90 days of frost is not 2160 hours of frost."""
    assert grade(DAYS_Q, 90.0, answer(90, "days")).correct
    assert not grade(DAYS_Q, 90.0, answer(2160, "hours")).correct


def test_wrong_count_is_still_wrong():
    assert not grade(COUNT_Q, 8760.0, answer(8000, "records")).correct
