# Data provenance

## The six-column selected dataset

The local `selected_data.xlsx` was checked against these six columns of:

- Repository: [Smart-Solar-Energy-Scheduling-with-Machine-Learning](https://github.com/riddhi-vyas/Smart-Solar-Energy-Scheduling-with-Machine-Learning)
- File: `Solar_Weather_FInal_Dataset.xlsx`
- Commit: `b1c62966d61ab326fe71b9327f017f8264cf4ef8`
- [Pinned file](https://github.com/riddhi-vyas/Smart-Solar-Energy-Scheduling-with-Machine-Learning/blob/b1c62966d61ab326fe71b9327f017f8264cf4ef8/Solar_Weather_FInal_Dataset.xlsx)
- Columns: `datetime`, `temp`, `humidity`, `solarradiation`, `Home Usage (kWh)`, `Solar Production (kWh)`
- 5856 rows; local date range 2023-08-01 00:00 to 2024-03-31 23:00.

This establishes the immediate repository source of the selected subset. It does
not independently establish how every value in the upstream workbook was measured
or generated, nor grant a data redistribution license.

The downloader pins that commit and checks the original workbook SHA-256:
`7fa57b98aa22cca6fdec84eb18a32a3d141e41d756c52ad5d84b021dd67ce2bb`.
Preparing a local file records its own SHA and does not automatically claim that
an arbitrary supplied workbook matches the upstream source. The original
selected subset has a different file hash because it contains fewer columns.

There are two expected daylight-saving transitions in the source: repeated
2023-11-05 01:00, and a jump from 2024-03-10 01:00 to 03:00. With California
local-time interpretation these are consecutive, distinct UTC hours. Preparation
preserves all 5856 observations and also saves their local timestamps.

The columns say kWh, so this release explicitly assumes hourly interval energy
and converts to average W. If the upstream unit definition is revised, pass the
correct `--input-unit kwh|kw|w` and document the resulting experiment. The code
does not invent missing weather or target values.

## The daily pivot CSV files are a separate dataset

`daily_pivot_prod_2011-2012.csv` and `daily_pivot_cons_2011-2012.csv` come from
[solarhome-control-bench/data](https://github.com/pierre-haessig/solarhome-control-bench/tree/master/data).
They are a different time period, schema and unit context, without the same
three weather features. They are not the parent tables of `selected_data` and
are not used by this release's forecast pipeline. A manuscript using the selected
workbook should cite its actual source rather than attributing it to these CSVs.

## Distribution and split indices

Raw workbooks and derived target series are not bundled. Use `prepare-data
--download` or provide a legitimately obtained local source file. Check upstream
terms before redistributing any data. The new chronological target-row indices
are in `data/splits/chronological_5856.json`; preprocessing regenerates the same
indices for 5856 rows. This split is deliberately different from the earlier
random test indices and does not recreate the earlier plotted experiment.
