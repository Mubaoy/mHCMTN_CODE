# Local release validation

Publication recheck, 2026-10-24: version v0.1.1 has **24 locally passing tests**,
including three ns/us/ms datetime-resolution regressions added after GitHub CI
exposed a pandas 3 interval-unit mismatch. See [CHANGELOG.md](../CHANGELOG.md).
GitHub's Linux CPU jobs also passed on Python 3.11 and 3.13 for the fixed code
commit `f0d7861ed0825e396a0b7fc34ba09f75a22a68ea`:
[successful hosted test run](https://github.com/Mubaoy/mHCMTN_CODE/actions/runs/40074097771).
The source experiment measurements below retain their original 2026-10-22 date
and tested pandas 2.3.3 environment.

Completed on 2026-10-22. These are actual software execution checks, not recovered
historical experiments or evidence that mHCMTN is superior to its baselines.
Machine-readable measured outputs are in `validation_summary.json`.

## Environment and checks

- Windows 11, Python 3.13.9, PyTorch 2.9.1+cpu. CPU execution; CUDA untested.
- `python -m pytest -q`: **21 passed**. In the managed local environment, a
  dedicated writable `--basetemp` was supplied. The tests check randomized
  conservation/nonnegativity including empty groups, meaningful neural gradients,
  all six optimizers, actual-PV isolation, checkpoint equality, causal STL,
  chronological target alignment, DST, all five forecast model reloads and
  independence from changed future observations.
- All six optimizers completed the separate offline synthetic smoke command.
- Verified all 5856 rows / six columns of local selected data against the pinned
  upstream workbook. Verified the full workbook's existing download checksum.
- Prepared all 5856 rows, resolving DST without dropping observations; split
  target counts 4612/586/586. This complete split is supplied with the source.
- Trained all five forecasting methods on real data; neural models used **one
  epoch** for this software check, not the 50-epoch default training protocol.
- Generated and loaded 12-step recursive planning forecasts from the first
  eligible held-out 08:00 local-time origin: 2024-03-08, UTC 16:00. Stored separate
  one-step and recursive metrics against raw observed PV.
- Ran all six optimizers on that STL-LSTM planning case, seed 42, neural steps
  1000 / population generations 1000 / initial population 30. No negative
  scheduled loads; recorded energy-conservation error 0 Wh for each returned
  schedule. Zero-valued flexible loads are allowed by the stated model.
- Recovered the stored best neural decision with `check`; objective and all
  twelve scheduled values exactly equaled the recorded result.
- Executed the combined `pipeline` command independently on a 240-row prefix,
  one forecast epoch, four neural steps and two population generations for all
  six optimizers. This was an additional orchestration smoke test.
- Exported actual-run convergence and schedule PNG/SVG/PDF with Times New Roman,
  18 pt; PNG metadata is approximately 600 dpi. Fonts were visually inspected.

## Measured examples and their limits

The one-epoch STL-LSTM check had rolling one-step test RMSE **842.95 W** on 586
targets. Its separate single-origin 12-step recursive RMSE was **1535.19 W**.
These are different protocols. Both metrics use raw observed PV. They are not
interchangeable, do not establish generalization across planning days, and
should not replace manuscript numbers without new experiments.

The default optimization check returned:

| Method | Normalized matching objective J | Search objective calls |
|---|---:|---:|
| mHCMTN | 0.129009 | 3001 |
| GA | 0.124687 | 30030 |
| PSO | 0.129082 | 30030 |
| MLPO | 0.198022 | 3001 |
| IWO | 0.124687 | 120073 |
| WPA | 0.142931 | 276030 |

This single case does not show mHCMTN outperforming all baselines. Iterations
and seeds were controlled but objective-call budgets differ. Reported timings
in the JSON are execution diagnostics, not controlled speed benchmarks.

## Repeating the checks

After preparing the pinned full dataset, run from the repository root:

```powershell
python -m pytest -q
python -m mhcmtn smoke --output results/smoke --no-plot
python -m mhcmtn forecast --data-dir results/prepared --output results/forecast --forecast-epochs 1 --no-plot
python -m mhcmtn benchmark --case results/forecast/planning_case_stl_lstm.json --output results/default_1000 --config configs/default.json --no-plot
python -m mhcmtn check --checkpoint results/default_1000/mhcmtn/checkpoint.pt
```

Actual-run caches, raw measurements and model weights remain local rather than
being bundled in the upload directory. To claim full experimental results,
train using the declared protocol, evaluate multiple planning origins and seeds,
compare suitable budgets, run the required ablations/statistical analysis, and
align the manuscript equations and scope with the implemented model.
