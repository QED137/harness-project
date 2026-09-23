"""Step 6: the question set with ground truth.

    python -m tsagent.questions            list the questions
    python -m tsagent.questions --check    compute every ground truth on the real dataset

Each question is precise enough to have exactly one correct answer, and that answer
is computed from the dataset with pandas, never written down by hand.

Rules followed here:
- state the period, the aggregation and the unit, so "correct" is well defined;
- prefer precise-but-tricky over vague: unit conversions, inclusive end dates,
  UTC vs local time, the leap day, resampling before aggregating;
- include questions the dataset cannot answer: the correct behaviour is to decline.

Tolerances are absolute, in the unit of the question. Counts use 0.5 (so a rounded
integer is accepted); temperatures 0.01 °C, i.e. far tighter than any real mistake.
"""

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .dataset import PARQUET_NAME

NOT_ANSWERABLE = "not answerable"
TEMP, HUM, PRECIP, WIND, PRESS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    unit: str | None
    truth: Callable[[pd.DataFrame], float | str]
    tolerance: float = 0.01
    category: str = "aggregation"
    difficulty: str = "easy"
    traps: tuple[str, ...] = field(default=())


# ------------------------------------------------------------------ helpers
def daily(df: pd.DataFrame, column: str, how: str) -> pd.Series:
    return getattr(df[column].resample("D"), how)()


def month_id(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def day_id(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


QUESTIONS: list[Question] = [
    # ---------------------------------------------------------- simple aggregation
    Question(
        "mean_temp_2023",
        "What was the mean 2 m air temperature over the whole year 2023, in °C?",
        "°C",
        lambda df: df.loc["2023", TEMP].mean(),
    ),
    Question(
        "max_temp_all",
        "What is the highest hourly 2 m air temperature in the whole dataset, in °C?",
        "°C",
        lambda df: df[TEMP].max(),
    ),
    Question(
        "min_temp_all",
        "What is the lowest hourly 2 m air temperature in the whole dataset, in °C?",
        "°C",
        lambda df: df[TEMP].min(),
    ),
    Question(
        "median_temp_2023",
        "What was the median hourly 2 m air temperature in 2023, in °C?",
        "°C",
        lambda df: df.loc["2023", TEMP].median(),
    ),
    Question(
        "std_temp_2023",
        "What is the standard deviation of the hourly 2 m air temperature in 2023, in °C? "
        "Use the pandas default (sample standard deviation).",
        "°C",
        lambda df: df.loc["2023", TEMP].std(),
        difficulty="medium",
    ),
    Question(
        "p95_temp_2023",
        "What is the 95th percentile of the hourly 2 m air temperature in 2023, in °C? "
        "Use linear interpolation (the pandas default).",
        "°C",
        lambda df: df.loc["2023", TEMP].quantile(0.95),
        difficulty="medium",
    ),
    Question(
        "p10_temp_2022",
        "What is the 10th percentile of the hourly 2 m air temperature in 2022, in °C? "
        "Use linear interpolation (the pandas default).",
        "°C",
        lambda df: df.loc["2022", TEMP].quantile(0.10),
        difficulty="medium",
    ),
    Question(
        "mean_humidity_2024",
        "What was the mean relative humidity in 2024, in %?",
        "%",
        lambda df: df.loc["2024", HUM].mean(),
    ),
    Question(
        "mean_pressure_2023",
        "What was the mean surface pressure in 2023, in hPa?",
        "hPa",
        lambda df: df.loc["2023", PRESS].mean(),
    ),
    Question(
        "total_precip_2022",
        "What was the total precipitation in 2022, in mm?",
        "mm",
        lambda df: df.loc["2022", PRECIP].sum(),
        tolerance=0.05,
    ),
    Question(
        "total_precip_all",
        "What is the total precipitation over the whole dataset, in mm?",
        "mm",
        lambda df: df[PRECIP].sum(),
        tolerance=0.05,
    ),
    Question(
        "mean_wind_2021",
        "What was the mean 10 m wind speed in 2021, in km/h?",
        "km/h",
        lambda df: df.loc["2021", WIND].mean(),
    ),
    # ---------------------------------------------------------- unit conversion
    Question(
        "mean_wind_2023_ms",
        "What was the mean 10 m wind speed in 2023, in m/s?",
        "m/s",
        lambda df: df.loc["2023", WIND].mean() / 3.6,
        category="unit_conversion",
        difficulty="medium",
        traps=("data is in km/h",),
    ),
    Question(
        "mean_temp_2023_kelvin",
        "What was the mean 2 m air temperature in 2023, in Kelvin?",
        "K",
        lambda df: df.loc["2023", TEMP].mean() + 273.15,
        category="unit_conversion",
        difficulty="medium",
        traps=("data is in °C",),
    ),
    Question(
        "max_temp_2020_fahrenheit",
        "What is the highest hourly 2 m air temperature in 2020, in °F?",
        "°F",
        lambda df: df.loc["2020", TEMP].max() * 9 / 5 + 32,
        tolerance=0.02,
        category="unit_conversion",
        difficulty="medium",
        traps=("data is in °C",),
    ),
    Question(
        "mean_pressure_2022_pa",
        "What was the mean surface pressure in 2022, in Pa?",
        "Pa",
        lambda df: df.loc["2022", PRESS].mean() * 100,
        tolerance=5.0,
        category="unit_conversion",
        difficulty="medium",
        traps=("data is in hPa",),
    ),
    Question(
        "total_precip_2023_cm",
        "What was the total precipitation in 2023, in cm?",
        "cm",
        lambda df: df.loc["2023", PRECIP].sum() / 10,
        tolerance=0.02,
        category="unit_conversion",
        difficulty="medium",
    ),
    # ---------------------------------------------------------- time selection
    Question(
        "mean_temp_summer_2023",
        "What was the mean 2 m air temperature from 1 June 2023 to 31 August 2023 inclusive, in °C? "
        "Include all hours of 31 August.",
        "°C",
        lambda df: df.loc["2023-06-01":"2023-08-31", TEMP].mean(),
        category="time_selection",
        difficulty="medium",
        traps=("inclusive end date", "all 24 hours of the last day"),
    ),
    Question(
        "mean_temp_winter_2022_23",
        "What was the mean 2 m air temperature over the winter months December 2022, January 2023 and "
        "February 2023, in °C?",
        "°C",
        lambda df: df.loc["2022-12-01":"2023-02-28", TEMP].mean(),
        category="time_selection",
        difficulty="medium",
        traps=("winter spans two calendar years",),
    ),
    Question(
        "mean_temp_first_week_2023",
        "What was the mean 2 m air temperature from 1 January 2023 to 7 January 2023 inclusive, in °C?",
        "°C",
        lambda df: df.loc["2023-01-01":"2023-01-07", TEMP].mean(),
        category="time_selection",
    ),
    Question(
        "mean_temp_leap_day_2024",
        "What was the mean 2 m air temperature on 29 February 2024, in °C?",
        "°C",
        lambda df: df.loc["2024-02-29", TEMP].mean(),
        category="time_selection",
        difficulty="medium",
        traps=("leap day",),
    ),
    Question(
        "hours_in_2023",
        "How many hourly records does the dataset contain for the year 2023?",
        "hours",
        lambda df: float(len(df.loc["2023"])),
        tolerance=0.5,
        category="time_selection",
    ),
    Question(
        "mean_temp_july_nights_2023",
        "What was the mean 2 m air temperature in July 2023 for the hours from 22:00 to 04:00 UTC inclusive "
        "(that is, hours 22, 23, 0, 1, 2, 3 and 4), in °C?",
        "°C",
        lambda df: df.loc["2023-07"][df.loc["2023-07"].index.hour.isin([22, 23, 0, 1, 2, 3, 4])][TEMP].mean(),
        category="time_selection",
        difficulty="hard",
        traps=("hour window wraps around midnight", "UTC not local time"),
    ),
    Question(
        "mean_temp_hour3_2023",
        "What was the mean 2 m air temperature in 2023 at 03:00 UTC (that hour only), in °C?",
        "°C",
        lambda df: df.loc["2023"][df.loc["2023"].index.hour == 3][TEMP].mean(),
        category="time_selection",
        difficulty="medium",
    ),
    Question(
        "diurnal_range_2023",
        "In 2023, what is the mean temperature at 15:00 UTC minus the mean temperature at 03:00 UTC, in °C?",
        "°C",
        lambda df: (
            df.loc["2023"][df.loc["2023"].index.hour == 15][TEMP].mean()
            - df.loc["2023"][df.loc["2023"].index.hour == 3][TEMP].mean()
        ),
        category="time_selection",
        difficulty="hard",
    ),
    # ---------------------------------------------------------- resampling
    Question(
        "warmest_month_2023",
        "Which calendar month of 2023 had the highest mean 2 m air temperature? Answer as YYYY-MM.",
        None,
        lambda df: month_id(df.loc["2023", TEMP].resample("MS").mean().idxmax()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "coldest_month_all",
        "Which calendar month in the whole dataset had the lowest mean 2 m air temperature? "
        "Answer as YYYY-MM.",
        None,
        lambda df: month_id(df[TEMP].resample("MS").mean().idxmin()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "driest_month_2023",
        "Which calendar month of 2023 had the lowest total precipitation? Answer as YYYY-MM.",
        None,
        lambda df: month_id(df.loc["2023", PRECIP].resample("MS").sum().idxmin()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "wettest_month_all",
        "Which calendar month in the whole dataset had the highest total precipitation? Answer as YYYY-MM.",
        None,
        lambda df: month_id(df[PRECIP].resample("MS").sum().idxmax()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "most_humid_month_2023",
        "Which calendar month of 2023 had the highest mean relative humidity? Answer as YYYY-MM.",
        None,
        lambda df: month_id(df.loc["2023", HUM].resample("MS").mean().idxmax()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "warmest_year",
        "Which calendar year in the dataset had the highest mean 2 m air temperature? Answer as YYYY.",
        None,
        lambda df: str(df[TEMP].resample("YS").mean().idxmax().year),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "warmest_day_2023",
        "Which calendar day in 2023 had the highest daily mean 2 m air temperature? Answer as YYYY-MM-DD.",
        None,
        lambda df: day_id(daily(df.loc["2023"], TEMP, "mean").idxmax()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "coldest_day_2021",
        "Which calendar day in 2021 had the lowest daily mean 2 m air temperature? Answer as YYYY-MM-DD.",
        None,
        lambda df: day_id(daily(df.loc["2021"], TEMP, "mean").idxmin()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "windiest_day_2023",
        "Which calendar day in 2023 had the highest daily mean 10 m wind speed? Answer as YYYY-MM-DD.",
        None,
        lambda df: day_id(daily(df.loc["2023"], WIND, "mean").idxmax()),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "hottest_hour_timestamp_2022",
        "At which timestamp did the highest hourly 2 m air temperature of 2022 occur? "
        "Answer in UTC as YYYY-MM-DD HH:MM.",
        None,
        lambda df: df.loc["2022", TEMP].idxmax().strftime("%Y-%m-%d %H:%M"),
        category="resampling",
        difficulty="medium",
    ),
    Question(
        "july_2023_precip",
        "What was the total precipitation in July 2023, in mm?",
        "mm",
        lambda df: df.loc["2023-07", PRECIP].sum(),
        tolerance=0.05,
        category="resampling",
    ),
    Question(
        "mean_daily_range_2023",
        "In 2023, what is the mean of the daily temperature range (daily maximum minus daily minimum), "
        "in °C?",
        "°C",
        lambda df: (daily(df.loc["2023"], TEMP, "max") - daily(df.loc["2023"], TEMP, "min")).mean(),
        category="resampling",
        difficulty="hard",
        traps=("resample before subtracting",),
    ),
    Question(
        "max_24h_precip_2023",
        "What is the highest precipitation total over any 24 consecutive hourly records in 2023, in mm? "
        "Use a rolling 24-hour window.",
        "mm",
        lambda df: df.loc["2023", PRECIP].rolling(24).sum().max(),
        tolerance=0.05,
        category="resampling",
        difficulty="hard",
    ),
    Question(
        "warmest_hour_of_day_2023",
        "Across all days of 2023, which hour of the day (0-23, UTC) has the highest mean 2 m air "
        "temperature? "
        "Answer as an integer.",
        None,
        lambda df: str(int(df.loc["2023"].groupby(df.loc["2023"].index.hour)[TEMP].mean().idxmax())),
        category="resampling",
        difficulty="hard",
    ),
    # ---------------------------------------------------------- counting
    Question(
        "frost_hours_2023",
        "How many hourly records in 2023 had a 2 m air temperature below 0 °C?",
        "hours",
        lambda df: float((df.loc["2023", TEMP] < 0).sum()),
        tolerance=0.5,
        category="counting",
    ),
    Question(
        "summer_days_2023",
        "How many calendar days in 2023 had a daily maximum 2 m air temperature of 25 °C or more?",
        "days",
        lambda df: float((daily(df.loc["2023"], TEMP, "max") >= 25).sum()),
        tolerance=0.5,
        category="counting",
        difficulty="medium",
        traps=("daily maximum, not hourly",),
    ),
    Question(
        "frost_days_2021",
        "How many calendar days in 2021 had a daily minimum 2 m air temperature below 0 °C?",
        "days",
        lambda df: float((daily(df.loc["2021"], TEMP, "min") < 0).sum()),
        tolerance=0.5,
        category="counting",
        difficulty="medium",
    ),
    Question(
        "wet_hours_2022",
        "How many hourly records in 2022 had precipitation greater than 0 mm?",
        "hours",
        lambda df: float((df.loc["2022", PRECIP] > 0).sum()),
        tolerance=0.5,
        category="counting",
    ),
    Question(
        "dry_days_2023",
        "How many calendar days in 2023 had a daily precipitation total below 0.1 mm?",
        "days",
        lambda df: float((daily(df.loc["2023"], PRECIP, "sum") < 0.1).sum()),
        tolerance=0.5,
        category="counting",
        difficulty="medium",
    ),
    Question(
        "mild_days_2023",
        "How many calendar days in 2023 had a daily mean 2 m air temperature above 20 °C?",
        "days",
        lambda df: float((daily(df.loc["2023"], TEMP, "mean") > 20).sum()),
        tolerance=0.5,
        category="counting",
        difficulty="medium",
    ),
    Question(
        "humid_hours_share_2023",
        "What percentage of the hourly records in 2023 had a relative humidity above 90%?",
        "%",
        lambda df: float((df.loc["2023", HUM] > 90).mean() * 100),
        tolerance=0.05,
        category="counting",
        difficulty="medium",
    ),
    Question(
        "longest_hot_streak_2023",
        "What is the longest run of consecutive hourly records in 2023 with a 2 m air temperature "
        "above 25 °C? "
        "Answer as a number of hours.",
        "hours",
        lambda df: float(
            (lambda s: s.groupby((~s).cumsum()).sum().max())(df.loc["2023", TEMP] > 25),
        ),
        tolerance=0.5,
        category="counting",
        difficulty="hard",
        traps=("consecutive run, not a total count",),
    ),
    # ---------------------------------------------------------- multi-step
    Question(
        "temp_change_2021_2023",
        "What is the mean 2 m air temperature of 2023 minus the mean of 2021, in °C?",
        "°C",
        lambda df: df.loc["2023", TEMP].mean() - df.loc["2021", TEMP].mean(),
        category="multi_step",
        difficulty="medium",
    ),
    Question(
        "pressure_diff_jan_jul_2023",
        "What is the mean surface pressure of January 2023 minus the mean of July 2023, in hPa?",
        "hPa",
        lambda df: df.loc["2023-01", PRESS].mean() - df.loc["2023-07", PRESS].mean(),
        tolerance=0.02,
        category="multi_step",
        difficulty="medium",
    ),
    Question(
        "temp_humidity_correlation_2023",
        "What is the Pearson correlation coefficient between hourly 2 m air temperature and relative "
        "humidity "
        "in 2023?",
        None,
        lambda df: df.loc["2023", TEMP].corr(df.loc["2023", HUM]),
        tolerance=0.005,
        category="multi_step",
        difficulty="medium",
    ),
    Question(
        "mean_temp_wettest_month_2023",
        "Take the calendar month of 2023 with the highest total precipitation. What was the mean 2 m air "
        "temperature in that month, in °C?",
        "°C",
        lambda df: df.loc[month_id(df.loc["2023", PRECIP].resample("MS").sum().idxmax()), TEMP].mean(),
        category="multi_step",
        difficulty="hard",
        traps=("two steps: find the month, then aggregate",),
    ),
    # ---------------------------------------------------------- not answerable
    Question(
        "snow_depth",
        "What was the mean snow depth in January 2023, in cm?",
        None,
        lambda df: NOT_ANSWERABLE,
        category="not_answerable",
        difficulty="medium",
        traps=("the dataset has no snow variable",),
    ),
    Question(
        "sunshine_hours",
        "How many hours of sunshine were recorded in July 2023?",
        None,
        lambda df: NOT_ANSWERABLE,
        category="not_answerable",
        difficulty="medium",
    ),
    Question(
        "wind_direction",
        "What was the most frequent wind direction in 2023?",
        None,
        lambda df: NOT_ANSWERABLE,
        category="not_answerable",
        difficulty="medium",
    ),
    Question(
        "berlin_temperature",
        "What was the mean 2 m air temperature in Berlin in 2023, in °C?",
        None,
        lambda df: NOT_ANSWERABLE,
        category="not_answerable",
        difficulty="hard",
        traps=("the dataset covers one location only",),
    ),
    Question(
        "temperature_2026",
        "What was the mean 2 m air temperature in 2026, in °C?",
        None,
        lambda df: NOT_ANSWERABLE,
        category="not_answerable",
        difficulty="medium",
        traps=("outside the dataset period",),
    ),
]

BY_ID = {q.id: q for q in QUESTIONS}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / PARQUET_NAME)


def ground_truth(question: Question, df: pd.DataFrame) -> float | str:
    value = question.truth(df)
    return value if isinstance(value, str) else float(value)


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tsagent.questions", description="The evaluation question set."
    )
    p.add_argument("--check", action="store_true", help="compute every ground truth on the dataset")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)

    if not args.check:
        for q in QUESTIONS:
            print(f"{q.id:32} {q.category:15} {q.difficulty:6} {q.unit or '-':6} {q.text[:70]}")
        print(f"\n{len(QUESTIONS)} questions")
        return 0

    try:
        df = load_dataset(args.data_dir)
    except FileNotFoundError:
        print("dataset not found; run: python -m tsagent.dataset", file=sys.stderr)
        return 2

    failures = 0
    for q in QUESTIONS:
        try:
            value = ground_truth(q, df)
        except Exception as e:  # noqa: BLE001  one broken truth must not hide the others
            failures += 1
            print(f"{q.id:32} ERROR {type(e).__name__}: {e}", file=sys.stderr)
            continue
        shown = value if isinstance(value, str) else f"{value:.4f}"
        print(f"{q.id:32} {shown:>22} {q.unit or ''}")
    print(f"\n{len(QUESTIONS)} questions, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
