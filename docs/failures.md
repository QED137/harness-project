# Failure analysis

7 failed runs out of 560 (1.2%).

## Failure categories (suggested from the grading reason)

| category | runs |
|---|---|
| wrong_value | 6 |
| answered_unanswerable | 1 |

## Failed runs by question

### mean_temp_wettest_month_2023 (2 failed)

*Take the calendar month of 2023 with the highest total precipitation. What was the mean 2 m air temperature in that month, in °C?*

Intended trap: two steps: find the month, then aggregate

- run 4 (loop): expected `6.24486111111111`, got `6.447345767575323` °C - off by 0.2025 (tolerance 0.01 °C)

```python
import pandas as pd
monthly_precip = df['precipitation'].resample('MS').sum()
best_month = monthly_precip['2023'].idxmax()
start = best_month
end = best_month + pd.offsets.MonthEnd(0)
mean_temp = df.loc[start:end, 'temperature_2m'].mean()
(best_month, float(mean_temp))
```

```python
import pandas as pd
monthly_precip = df['precipitation'].resample('MS').sum()
best_month = monthly_precip['2023'].idxmax()
start = best_month
end = best_month + pd.offsets.MonthEnd(0)
mean_temp = df.loc[start:end, 'temperature_2m'].mean()
result = (str(best_month.date()), float(mean_temp))
```
- run 5 (loop): expected `6.24486111111111`, got `6.447345767575323` °C - off by 0.2025 (tolerance 0.01 °C)

```python
import pandas as pd
# Use df directly

df_2023 = df[(df.index.year == 2023)]
monthly_precip = df_2023['precipitation'].resample('ME').sum()  # ME for MonthEnd alias
max_month = monthly_precip.idxmax()
max_total = monthly_precip.max()
month_start = max_month - pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(0)  # not reliable; let's simpler: month_start = max_month - pd.offsets.MonthEnd(0) + pd.offsets.MonthBegin(0) ???
# Simpler: compute month start by constructing from year and month
month_start = max_month.replace(day=1)
mask_month = (df_2023.index >= month_start) & (df_2023.index <= max_month)
mean_temp = df_2023.loc[mask_month, 'temperature_2m'].mean()

result = float(mean_temp)
month_year_str = f"{max_month.year}-{max_month.month:02d}"
{ 'result': result, 'month_year': month_year_str, 'max_month_end': str(max_month), 'max_total_mm': float(max_total) }
```

```python
import pandas as pd
# Use df directly

df_2023 = df[(df.index.year == 2023)]
monthly_precip = df_2023['precipitation'].resample('ME').sum()
max_month = monthly_precip.idxmax()
max_total = monthly_precip.max()
month_start = max_month.replace(day=1)
mask_month = (df_2023.index >= month_start) & (df_2023.index <= max_month)
mean_temp = df_2023.loc[mask_month, 'temperature_2m'].mean()
print('max_month=', max_month, 'year=', max_month.year, 'month=', max_month.month, 'max_total_mm=', max_total, 'mean_temp_C=', mean_temp)
result = float(mean_temp)
```

### mean_temp_july_nights_2023 (1 failed)

*What was the mean 2 m air temperature in July 2023 for the hours from 22:00 to 04:00 UTC inclusive (that is, hours 22, 23, 0, 1, 2, 3 and 4), in °C?*

Intended trap: hour window wraps around midnight, UTC not local time

- run 2 (loop): expected `16.098617511520736`, got `16.14433962264151` °C - off by 0.04572 (tolerance 0.01 °C)

```python
import pandas as pd
import numpy as np
# df is preloaded as hourly weather dataset with UTC DatetimeIndex
idx = df.index
# Determine temperature column robustly
if 'temperature_2m' in df.columns:
    col = 'temperature_2m'
else:
    col = None
    for c in df.columns:
        if '2m' in c.lower() and ('temp' in c.lower() or 'temperature' in c.lower()):
            col = c
            break
    if col is None:
        raise ValueError('temperature_2m column not found')

year = idx.year
month = idx.month
hour = idx.hour
is_jul_2023 = (year == 2023) & (month == 7)
mask1 = is_jul_2023 & np.isin(hour, [22, 23])
mask2 = is_jul_2023 & np.isin(hour, [0,1,2,3,4]) & (idx.date != pd.to_datetime('2023-07-01').date())
mask = mask1 | mask2
sel = df.loc[mask, col]
mean_val = sel.mean()
result = mean_val
"}
```

```python
import pandas as pd
import numpy as np
# df is preloaded as hourly weather dataset with UTC DatetimeIndex
idx = df.index
# Determine temperature column robustly
if 'temperature_2m' in df.columns:
    col = 'temperature_2m'
else:
    col = None
    for c in df.columns:
        if '2m' in c.lower() and ('temp' in c.lower() or 'temperature' in c.lower()):
            col = c
            break
    if col is None:
        raise ValueError('temperature_2m column not found')

year = idx.year
month = idx.month
hour = idx.hour
is_jul_2023 = (year == 2023) & (month == 7)
mask1 = is_jul_2023 & np.isin(hour, [22, 23])
mask2 = is_jul_2023 & np.isin(hour, [0,1,2,3,4]) & (idx.date != pd.to_datetime('2023-07-01').date())
mask = mask1 | mask2
sel = df.loc[mask, col]
mean_val = sel.mean()
result = mean_val
```

### july_2023_precip (1 failed)

*What was the total precipitation in July 2023, in mm?*

- run 7 (loop): expected `65.3`, got `71.6` mm - off by 6.3 (tolerance 0.05 mm)

```python
import pandas as pd
import numpy as np
# df is preloaded with hourly data and UTC DatetimeIndex named 'time'
# Ensure index is datetime index
idx = df.index
# We'll filter for July 2023
start = '2023-07-01'
end = '2023-08-01'
subset = df.loc[start:end]
# compute total precipitation in July 2023
total_precip = subset['precipitation'].sum()
result = total_precip
result
```

### dry_days_2023 (1 failed)

*How many calendar days in 2023 had a daily precipitation total below 0.1 mm?*

- run 3 (loop): expected `129.0`, got `130.0` days - off by 1 (tolerance 0.5 days)

```python
import pandas as pd
import numpy as np

# df is preloaded in environment as hourly data with UTC DatetimeIndex
# compute daily precipitation totals for 2023 and count days with total < 0.1 mm

if 'df' not in globals():
    raise SystemExit('df not found')

# Ensure index is datetime and timezone aware; the dataset says UTC index
# Filter to year 2023
series_2023 = df['2023'] if 'df' in globals() else None
if series_2023 is None:
    # fallback: create boolean mask from index
    df_2023 = df[(df.index >= '2023-01-01') & (df.index <= '2023-12-31')]
else:
    df_2023 = series_2023

# Sum precipitation per day
# Some datasets might have precipitation as 0.0; ensure numeric
precip = df_2023['precipitation']
daily = precip.resample('D').sum()
# Count days where sum < 0.1 mm
count_below = (daily < 0.1).sum()

result = int(count_below)
print(result)
```

```python
import pandas as pd

# df is preloaded in the sandbox environment
# Filter 2023 data by date range using the index
mask = (df.index >= '2023-01-01') & (df.index <= '2023-12-31')
df_2023 = df[mask]

# Sum hourly precipitation to daily totals
daily_total = df_2023['precipitation'].resample('D').sum()
count_below_01 = (daily_total < 0.1).sum()
result = int(count_below_01)
print(result)
```

### pressure_diff_jan_jul_2023 (1 failed)

*What is the mean surface pressure of January 2023 minus the mean of July 2023, in hPa?*

- run 9 (loop): expected `1.6784946236558653`, got `2.141276041666856` hPa - off by 0.4628 (tolerance 0.02 hPa)

```python
import pandas as pd
import numpy as np
# df is preloaded with hourly data, index is UTC datetime
# Ensure we are in the correct environment
# Select January 2023 data
jan_start = '2023-01-01'
jan_end = '2023-02-01'
jul_start = '2023-07-01'
jul_end = '2023-08-01'
# Compute means for the specified months
jan_df = df.loc[jan_start:jan_end, 'surface_pressure'] if 'surface_pressure' in df.columns else df.loc[jan_start:jan_end].surface_pressure
jul_df = df.loc[jul_start:jul_end, 'surface_pressure'] if 'surface_pressure' in df.columns else df.loc[jul_start:jul_end].surface_pressure
jan_mean = jan_df.mean()
jul_mean = jul_df.mean()
result = jan_mean - jul_mean
result
```

### berlin_temperature (1 failed)

*What was the mean 2 m air temperature in Berlin in 2023, in °C?*

Intended trap: the dataset covers one location only

- run 5 (loop): expected `not answerable`, got `11.31` °C - answered a question the dataset cannot answer

```python
import pandas as pd

# The dataset df is preloaded in the environment
# Compute mean 2m temperature for 2023
start = '2023-01-01'
end = '2023-12-31 23:00:00'

# Ensure index is datetime
idx = df.index
if not isinstance(idx, pd.DatetimeIndex):
    idx = pd.to_datetime(idx)

mask = (idx >= start) & (idx <= end)
mean_2023 = df.loc[mask, 'temperature_2m'].mean()

result = mean_2023
```

## Notes

Categories above are derived from the grading reason and are only a starting point.
Read the code of each failed run before drawing conclusions.
