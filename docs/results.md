# Results

560 runs, 56 questions, model(s): gpt-5-nano.

## Summary per configuration

| config | runs | pass@1 | pass@5 | pass@10 | consistency | median s | tokens in/out |
|---|---|---|---|---|---|---|---|
| loop | 560 | 0.988 | 1.000 | 1.000 | 0.988 | 14.3 | 3549 / 1232 |

## pass@1 by category

| category | loop |
|---|---|
| aggregation | 1.000 |
| counting | 0.988 |
| multi_step | 0.925 |
| not_answerable | 0.980 |
| resampling | 0.993 |
| time_selection | 0.988 |
| unit_conversion | 1.000 |

## pass@1 by difficulty

| difficulty | loop |
|---|---|
| easy | 0.993 |
| hard | 0.950 |
| medium | 0.994 |

## Hardest questions

| question | category | pass@1 | most common failure |
|---|---|---|---|
| mean_temp_wettest_month_2023 | multi_step | 0.80 | off by 0.2025 (tolerance 0.01 °C) |
| berlin_temperature | not_answerable | 0.90 | answered a question the dataset cannot answer |
| dry_days_2023 | counting | 0.90 | off by 1 (tolerance 0.5 days) |
| july_2023_precip | resampling | 0.90 | off by 6.3 (tolerance 0.05 mm) |
| mean_temp_july_nights_2023 | time_selection | 0.90 | off by 0.04572 (tolerance 0.01 °C) |
| pressure_diff_jan_jul_2023 | multi_step | 0.90 | off by 0.4628 (tolerance 0.02 hPa) |
| coldest_day_2021 | resampling | 1.00 | - |
| coldest_month_all | resampling | 1.00 | - |
| diurnal_range_2023 | time_selection | 1.00 | - |
| driest_month_2023 | resampling | 1.00 | - |

## Sandbox events

| event | count |
|---|---|
| runtime_import | 70 |
| static_forbidden_name | 23 |
| static_import | 3 |

## Tool calls

- total tool calls: 1857
- schema-level validation failures: 0.0000 per call
- rule-level validation failures: 0.0000 per call
- invalid JSON: 0.0000 per call
