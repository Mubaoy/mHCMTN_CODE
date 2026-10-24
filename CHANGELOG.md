# Changes

## v0.1.1 — 2026-10-24

Fixed the interval-energy to average-power conversion when pandas stores
timestamps in microseconds or milliseconds instead of nanoseconds. Calculate
elapsed hours by dividing timedeltas by an explicit one-hour duration.

The first public CI run with pandas 3.1.0 found that the previous integer-unit
assumption produced a 0.001-hour step for hourly data. The original local
pandas 2.3.3 checks had nanosecond timestamps and did not expose this case.
Added three regression cases verifying that ns/us/ms timestamp storage all
produce a one-hour step and the correct 1000 W per hourly kWh conversion.
The updated local test suite has 24 passing tests.

Reference: pandas documents the change in
[datetime resolution inference](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v3.0.0.html#datetime-timedelta-resolution-inference).

## v0.1.0 — 2026-10-24

Initial public source release, with the previously validated pandas 2.3.3
environment. The v0.1.0 tag is retained unchanged for traceability. Use v0.1.1
or later with newer pandas environments.
