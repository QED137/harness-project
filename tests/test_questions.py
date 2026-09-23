"""The question set: structure, and every ground truth computed on a synthetic dataset
with the real schema (so these tests need neither the download nor Docker)."""

import numpy as np
import pandas as pd
import pytest

from tsagent import questions as qs
from tsagent.questions import NOT_ANSWERABLE, QUESTIONS, ground_truth
from tsagent.verifier import normalize_unit

CATEGORIES = {
    "aggregation",
    "unit_conversion",
    "time_selection",
    "resampling",
    "counting",
    "multi_step",
    "not_answerable",
}


@pytest.fixture(scope="module")
def df():
    """Five years of hourly data with a seasonal and daily cycle, same columns as the real set."""
    idx = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()
    temp = 10 - 9 * np.cos(2 * np.pi * doy / 365.25) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = temp + rng.normal(0, 2, len(idx))
    frame = pd.DataFrame(
        {
            qs.TEMP: temp.round(2),
            qs.HUM: np.clip(75 - 0.8 * (temp - 10) + rng.normal(0, 8, len(idx)), 5, 100).round(1),
            qs.PRECIP: np.where(rng.random(len(idx)) < 0.12, rng.gamma(1.2, 0.8, len(idx)), 0.0).round(2),
            qs.WIND: np.clip(rng.gamma(2.0, 4.0, len(idx)), 0, None).round(1),
            qs.PRESS: (970 + rng.normal(0, 7, len(idx))).round(1),
        },
        index=idx,
    )
    frame.index.name = "time"
    return frame


# ------------------------------------------------------------------ structure
def test_ids_are_unique_and_set_is_large_enough():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert len(QUESTIONS) >= 50


def test_every_question_is_well_formed():
    for q in QUESTIONS:
        assert q.text.strip().endswith("?") or q.text.strip().endswith(".")
        assert q.tolerance > 0
        assert q.category in CATEGORIES
        assert q.difficulty in {"easy", "medium", "hard"}
        if q.unit is not None:
            assert normalize_unit(q.unit)[0] is not None, f"{q.id}: unknown unit {q.unit}"


def test_questions_with_a_unit_state_it_in_the_text():
    for q in QUESTIONS:
        if q.unit is not None and q.unit not in ("hours", "days"):
            assert q.unit.lower() in q.text.lower() or "kelvin" in q.text.lower(), q.id
        if q.unit in ("hours", "days"):
            assert "how many" in q.text.lower() or "number of" in q.text.lower(), q.id


def test_all_categories_and_difficulties_are_represented():
    assert {q.category for q in QUESTIONS} == CATEGORIES
    assert {q.difficulty for q in QUESTIONS} == {"easy", "medium", "hard"}
    assert sum(q.category == "not_answerable" for q in QUESTIONS) >= 3
    assert sum(q.traps != () for q in QUESTIONS) >= 8


# ------------------------------------------------------------------ ground truth
def test_every_ground_truth_computes_and_has_the_right_type(df):
    for q in QUESTIONS:
        value = ground_truth(q, df)
        if q.category == "not_answerable":
            assert value == NOT_ANSWERABLE, q.id
        elif q.unit is not None:
            assert isinstance(value, float) and np.isfinite(value), q.id  # a unit implies a number
        elif isinstance(value, float):
            assert np.isfinite(value), q.id  # unitless numbers are allowed, e.g. a correlation
        else:
            assert isinstance(value, str) and value.strip(), q.id


def test_ground_truths_are_deterministic(df):
    assert [ground_truth(q, df) for q in QUESTIONS] == [ground_truth(q, df) for q in QUESTIONS]


def test_unit_conversions_match_the_underlying_values(df):
    """The converted questions must equal the plain computation times the right factor."""
    truths = {q.id: ground_truth(q, df) for q in QUESTIONS}
    assert truths["mean_wind_2023_ms"] == pytest.approx(df.loc["2023", qs.WIND].mean() / 3.6)
    assert truths["mean_temp_2023_kelvin"] == pytest.approx(df.loc["2023", qs.TEMP].mean() + 273.15)
    assert truths["mean_pressure_2022_pa"] == pytest.approx(df.loc["2022", qs.PRESS].mean() * 100)


def test_counting_questions_are_whole_numbers_in_range(df):
    for q in QUESTIONS:
        if q.category != "counting" or q.unit == "%":
            continue
        value = ground_truth(q, df)
        assert value == int(value) and value >= 0, q.id
        assert value <= (8784 if q.unit == "hours" else 366), q.id


def test_tricky_questions_really_are_tricky(df):
    """Each trap question must differ from the naive answer it is designed to catch."""
    naive_summer = df.loc["2023-06-01":"2023-08-30", qs.TEMP].mean()  # forgetting the last day
    assert ground_truth(qs.BY_ID["mean_temp_summer_2023"], df) != pytest.approx(naive_summer)

    naive_range = df.loc["2023", qs.TEMP].max() - df.loc["2023", qs.TEMP].min()  # no daily resampling
    assert ground_truth(qs.BY_ID["mean_daily_range_2023"], df) != pytest.approx(naive_range)

    naive_streak = float((df.loc["2023", qs.TEMP] > 25).sum())  # total instead of longest run
    assert ground_truth(qs.BY_ID["longest_hot_streak_2023"], df) < naive_streak


# ------------------------------------------------------------------ CLI
def test_check_runs_on_a_dataset(tmp_path, df, capsys):
    df.to_parquet(tmp_path / "weather.parquet")
    assert qs.main(["--check", "--data-dir", str(tmp_path)]) == 0
    assert f"{len(QUESTIONS)} questions, 0 failed" in capsys.readouterr().out


def test_check_without_dataset_explains_what_to_do(tmp_path, capsys):
    assert qs.main(["--check", "--data-dir", str(tmp_path / "nope")]) == 2
    assert "python -m tsagent.dataset" in capsys.readouterr().err


def test_listing_needs_no_dataset(capsys):
    assert qs.main([]) == 0
    assert f"{len(QUESTIONS)} questions" in capsys.readouterr().out
