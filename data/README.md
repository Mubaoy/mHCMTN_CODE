# Data directory

No raw/processed measurement data are distributed. Download the pinned workbook
through `python -m mhcmtn prepare-data --download --output results/prepared`.

`splits/chronological_5856.json` contains zero-based **raw target row origins**
for 5856 rows, STL period 24, warmup 48 and lookback 24. The first origin is 72.
Input rows are `[origin-24, origin)`. Validation/test use past observations as
history; training-only scaling and causal decomposition prevent future leakage.

The split has 4612 training, 586 validation and 586 test targets. It is a new
protocol, not the legacy random split used for some old figures. A smaller
`--limit` input produces a different split saved with that run.
