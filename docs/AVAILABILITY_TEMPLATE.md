# Data and code availability

## Data Availability Statement

The dataset used to construct the six-column input is publicly available as
`Solar_Weather_FInal_Dataset.xlsx` in the
[Smart-Solar-Energy-Scheduling-with-Machine-Learning repository](https://github.com/riddhi-vyas/Smart-Solar-Energy-Scheduling-with-Machine-Learning),
at commit `b1c62966d61ab326fe71b9327f017f8264cf4ef8` (accessed on October 24, 2026).
The selected columns and provenance verification are described in
[DATA_PROVENANCE.md](DATA_PROVENANCE.md). Raw workbooks are not redistributed.

## Code Availability

The preprocessing scripts, chronological train/validation/test split indices,
and code for the mHCMTN optimization and forecasting pipeline are available at
https://github.com/Mubaoy/mHCMTN_CODE (version: v0.1.0).

The release documents the newly implemented method and its validation scope.
It does not assert recovery of historical model weights or manuscript results.
See [METHOD.md](METHOD.md) and [VALIDATION.md](VALIDATION.md).

## Environment of the release validation

The implementation was run using Python 3.13.9 and PyTorch 2.9.1 (CPU-only build)
on a computer equipped with an Intel Core i5-12500H processor. GPU acceleration
was not used for the reported local release checks. These environment details
refer to the release validation, rather than all historical experiments.
