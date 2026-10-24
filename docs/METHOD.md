# Method implemented in this release

## Problem and scope

A case has 12 nonnegative PV forecasts `p[t]`, 12 nonnegative baseline load
forecasts `b[t]`, equal slot duration `dt`, and a fixed 12-slot tariff. Power is
in W; energy is W × hours / 1000, in kWh. Each label `c[t]` is valley=1,
flat=2, or peak=3. These labels choose **load-transfer groups**; they do not
change the physical tariff vector. Response fractions `(a_pv, a_pf, a_fv)`
each lie in [0, 0.5]. They represent peak→valley, peak→flat, flat→valley.

From peak slots withdraw `(a_pv + a_pf) * b[t]`; from flat slots withdraw
`a_fv * b[t]`. Sum each withdrawn flow and share it equally over its destination
slots. Disable a flow if its destination group is empty. This gives nonnegative
loads with exactly the original total energy (up to numerical precision).
Unlike an older group-average withdrawal prototype, this decoder cannot make
a small-load source slot negative. No device-specific deadlines, power caps,
battery SOC or ESS decisions are modeled. Conserved flexible-load energy is
the feasibility claim supported by this implementation.

The optimization objective is:

```text
S = max(1 W, max(p), max(b))
J = mean_t ((scheduled_load[t] - p[t]) / S)^2
```

The normalization is fixed for each case, so minimizing J is equivalent to
minimizing power MSE. Predicted-PV MSE is an optimization objective, not a
forecast accuracy metric. Tariffs are context and evaluation inputs; **cost
is not an objective term**. Default tariffs 0.80/0.55/0.30 currency per kWh are
illustrative fixed inputs, not measurements from the upstream dataset.

Report forecast-based metrics and, if supplied, observed-PV metrics separately:
self-consumption = sum(min(load, PV))/sum(PV); self-sufficiency has sum(load)
in its denominator; grid import = max(load-PV,0). Zero denominators produce
JSON null. Observed PV never enters the optimization context or objective.
Actual load after a demand-response intervention is not available: these
metrics evaluate a proposed forecast-load schedule, not a measured intervention.

## mHCMTN

The current 15-vector (12 labels, three fractions) is normalized and projected
into 45 **latent** tokens. Each token receives scalar→embedding projection,
sinusoidal position encoding, and an embedding of 36 context values (PV,
baseline load, fixed tariff). The 45 positions are not 45 observed time steps.

For each attention head, learn a matrix A and define:

```text
G = head_dim * (A A^T + 1e-4 I) / trace(A A^T + 1e-4 I)
D_ij = (q_i-k_j)^T G (q_i-k_j) / head_dim
attention_ij = softmax_j(q_i k_j^T / sqrt(head_dim) - lambda D_ij)
L_geometry = mean_i sum_j attention_ij D_ij
```

Geometry uses learned q/k representations. It is a modeling choice, not proof
that neural outputs satisfy physical constraints. The explicit decoder supplies
the physical guarantees. Each block adds `beta * Linear(h_previous * h_current)`
to its attention residual, then LayerNorm. The feedforward residual is
`LayerNorm(h + dropout(FFN(h)) + gamma*h)`. States from all blocks and the input
are concatenated, projected, and decoded into categorical and response logits.

Categorical sampling uses hard straight-through Gumbel-softmax during training;
the greedy candidate uses argmax without dropout. Temperature follows geometric
annealing from 1 to 0.1. The continuous head adds residual logits to the inverse
reference sigmoid and maps `x=0.2*sigmoid(logit)` through:

```text
a(x) = 0                 for x < .05
       5*(x-.05)         for .05 <= x <= .15
       .5               for x > .15
```

Its custom backward passes the gradient straight through this piecewise map.
This is a surrogate derivative, not the true derivative of the clipped mapping.
The training loss is `J + lambda * L_geometry`. Incumbents are compared by J
only, after decoding hard decisions in float64. A zero-response baseline is
always retained as the initial incumbent. Adam is recreated and trained for
each case; it is **not an amortized model trained across a dataset of schedules**.
Effective optimization batch size is 1. Final weights and the best decision
may differ, so both are saved separately.

## Baselines

All optimizers receive exactly the same forecasts, problem objective and decoder.

- GA: elitism, uniform gene crossover from the better half, per-label mutation,
  Gaussian response mutation (standard deviation .07 in physical fraction units).
- PSO: normalized 15-dimensional positions, inertia .5, cognitive/social 1.6,
  positions clipped to [0,1], nearest-category decoding.
- MLPO: 15→15→15 Tanh network, residual ordinal labels with rounding STE and
  Lagrange one-hot memberships; same continuous response map; Adam lr .01.
- IWO: fitness-dependent seed counts 0–5, polynomially decreasing Gaussian
  dispersion .5→.001, exponent 3, competitive selection capped at 50.
- WPA: reconstructed scout/call/besiege/renew variant. Scouts probe 8 normalized
  random directions for up to 5 walks; call step .1; besiege scale .025 inside
  distance .05; worst 20% renewed while keeping a leader. This is an explicit
  reconstruction, not a recovered historical WPA implementation.

Every history records actual objective calls: neural methods include training
objective, exact training-candidate re-evaluation and greedy-candidate evaluation;
population methods count each fitness candidate. Final reporting-only rechecks
and metrics are excluded. Equal iteration caps do not imply equal compute or
evaluation budgets. Initial/final time covers initialization and solving, not
plotting/output-file writing. Compare multiple seeds, cases and budgets before
making a performance claim.

## Forecasting protocol

Use the six selected source columns. Local times are interpreted in
America/Los_Angeles; repeated autumn hours get their corresponding DST offsets.
Convert to continuous UTC without dropping or fabricating observations. Treat
the two labeled kWh columns as energy per observed hourly interval, converting
to W by `1000/dt`; this unit interpretation remains an explicit source assumption.
Do not substitute the unrelated daily-pivot dataset or silently treat its kW
values as kWh. Weather units are preserved as supplied (temperature is not
relabelled Celsius).

The default split uses zero-based raw target row origins: training 72..4683,
validation 4684..5269, test 5270..5855. Each input contains the 24 preceding
rows. Scaling fits training rows only. Each STL target at t is the endpoint
of a robust STL fit to rows max(0,t-167)..t with period 24 and seasonal smoother
7; the residual is `observed - trend - seasonal`. Initial two-period warmup
rows are excluded from training windows. No complete-series STL fit supplies
features to an earlier prediction.

STL-LSTM uses separate [weather, component] inputs for trend, seasonal and
residual, each two 50-unit PyTorch LSTM layers, dropout 0.2, a linear output.
The component forecasts are inverse-scaled then summed and clipped to >=0.
The raw LSTM baseline has 64 units with explicit Keras-style ReLU candidate and
cell output; recurrent gates use sigmoid. Tree and SVR baselines flatten the
same raw-PV/weather lookback. All methods train on the same target origins and
are scored against **raw measured PV**, not an STL reconstruction or another
model's output. Validation selects the neural checkpoint/early stop; test
targets are not used for fitting.

`test_predictions.csv` reports **rolling one-step-ahead** predictions: earlier
test observations become available as history for later predictions. It is not
a single-origin multi-day open-loop forecast. The planning case instead makes
12 recursive predictions from one held-out origin: no future PV/components or
weather observations are used; weather is held at its last observed value.
The first suitable 08:00 local-time test origin is selected without looking at
scores. Baseline load is the previous day's matching 12 hours, not the future
observed home load. Observed future PV is attached only for post-solve evaluation.

## Departures from historical material

The draft describes both a 15-vector and a 24-vector; this release selects the
existing 12-label + 3-fraction representation and documents it explicitly. Its
metric equations are underspecified; the positive-definite distance above is
this release's resolution. Its 64-wide/8-head description is a supplied optional
preset; the compact default follows the previously discussed implementation
configuration. Original random splits and full-series STL are replaced by the
causal chronological protocol. TensorFlow/Keras prediction weights are not
reused; models are newly trained in PyTorch. No historical curves, table numbers,
ablation scores or speed superiority are established by this code release.
