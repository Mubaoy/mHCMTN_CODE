# Implementation plan and scope

Date: 2026-10-22. Status: new implementation, not recovered historical source.

The requested deliverable is a self-contained directory ready to upload as a
GitHub repository. No remote repository is created by these scripts. Historical
digitized curves, illustrative hyperparameters and manuscript performance
numbers are not experimental evidence for this implementation.

## Implemented method

- Hourly, 12-slot demand-response optimization, with 12 categorical labels and
  three response fractions. A common decoder conserves load energy and prevents
  negative loads for every method.
- mHCMTN: solution/context embedding, 45 latent tokens, sinusoidal positions,
  multihead attention with a learned positive-definite distance metric,
  multiplicative cross-layer hyperconnections, feedforward residual blocks,
  layer fusion, categorical and continuous output heads.
- Hard Gumbel-softmax categorical decisions, straight-through continuous response,
  Adam, gradient clipping, temperature annealing, feasible incumbent tracking.
- GA, PSO, MLPO, IWO, and a documented reconstructed WPA variant, with the same
  objective and physical decoder. Record real candidate evaluation counts.
- Causal rolling-endpoint STL, three stacked LSTM branches, and LSTM, Random
  Forest, XGBoost, SVR forecasting baselines. Train/validation/test targets are
  chronological; transformations use training rows only. Planning uses recursive
  forecasts with persisted weather and a previous-day load forecast.

## Configuration

The default compact mHCMTN is 32/4/2/64 (width/heads/layers/feedforward),
dropout 0.1, manifold coefficient 0.1, hyperconnection coefficient 0.5,
feedforward residual coefficient 0.3. Adam lr 0.001; 1000 steps;
temperature 1.0 -> 0.1. Each solve trains on one instance: effective batch size 1.
An alternative 64/8/2/256 width preset is supplied. These are explicit choices
for the new code; the previous hypothetical parameter table is not a recovered log.

Prediction: 24-step lookback, STL period 24/window 168/seasonal smoother 7;
two 50-unit LSTM layers per component; dropout 0.2; Adam lr 0.001; batch 32;
50-epoch cap, validation early stopping patience 10. Raw LSTM: 64 units,
explicit ReLU candidate and cell-output activation. RF 100 trees; XGBoost
100 trees/lr 0.05/depth 6; RBF SVR C=100/gamma=0.1/epsilon=0.1.

## Verification stages

1. Unit checks for conservation, nonnegativity, gradients, causal decomposition,
   chronological indices, forecasts independent of future observations, actual-PV
   isolation, and checkpoint recovery.
2. Fresh code review under the experiment-bridge workflow, then concrete fixes.
3. Offline synthetic smoke solve with every optimizer, clearly marked synthetic.
4. Real six-column dataset preparation, short runs of all five forecasters,
   a held-out 12-step planning case, and an mHCMTN optimization run.
5. Publish source, configuration, split indices and verification notes. Do not
   redistribute source workbooks or label short checks as paper-level evidence.

Full multiseed forecasting, multiple-day scheduling, convergence comparisons,
ablations, statistical testing, battery/EV constraints, and manuscript revision
remain separate research tasks. They are not claimed as completed experiments.
