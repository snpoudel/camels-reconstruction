# How the models work

A walk through the architecture with the real numbers from this repository. Read it once
top to bottom and the three variants the project is built around — `vanilla`, `spatial`,
`donor` — should all be clear, because they differ in exactly one place.

Two further variants were added later and are described in §12: `donor_flow`, which changes
what attention borrows, and `vanilla_lit`, which exists only to match the literature's
training recipe. `src/config.py` lists all five. The project is finished; `EXPERIMENTS.md`
records what each one scored and why it stopped.

---

## 0. The setting

| | |
|---|---|
| basins | **531** CAMELS-US, ordered `[gauged …, ungauged …]` |
| fold 0 split | **455 gauged** (model may learn from their flow) · **76 ungauged** (held out entirely) |
| days | 12,784 (1980-01-01 → 2014-12-31), no gaps |
| training period | water years 1990–2004 = **5,479 days**, 100.000% flow-complete for every basin |
| test period | water years 2005–2014 = **3,652 days** |
| features | **36** per basin per day: 6 daily forcings + 30 static catchment attributes |
| input window | **270 days** ending on the target day |

"Ungauged" means the model never sees those 76 basins' discharge — not as an input, not in
the loss, not as a donor. It does see their *weather and catchment attributes*, which in
reality you always have (gridded meteorology, soil and geology maps).

---

## 1. The one unusual thing: a batch is a *day*, not a sample

In the standard multi-basin LSTM recipe, training samples are (basin, day) pairs shuffled
together into minibatches of a few hundred. **Here, one step is one day, and every basin is
processed at once.** The batch dimension *is* the basin dimension.

```python
for t in shuffled(train_days):          # 5,479 steps per epoch
    x = X[:n_enc, t-269 : t+1]          # every basin's window ending at t
    pred = model(x, ...)                # one flow value per basin, for day t
    loss = nse_loss(pred, Q[:n_query, t], sigma)
```

It has to be this way for `spatial` and `donor`: attention compares a basin against the
others **on the same day**, so they all have to be in the forward pass together.

Two consequences worth knowing:

- An epoch is 5,479 steps, not `5,479 × 455 / batch_size`. Each step is heavy (531 sequences
  of 270 days) but there are relatively few of them.
- **`vanilla` is trained the same way**, even though it doesn't need to be. That is
  deliberate: it keeps the optimiser, the number of gradient steps and the gradient noise
  identical across variants, so none of those can explain a difference between them. It does
  mean `vanilla` is not literally the standard recipe — it is the standard *architecture*
  trained under this repository's batching.

---

## 2. Inputs to a single step

For target day `t`:

| tensor | shape | what it is |
|---|---|---|
| `x` | `[N, 270, 36]` | each basin's own forcings + attributes, days `t-269 … t` |
| `Z_d` | `[455, 14]` | **`donor` only** — each gauged basin's recent flow, see §5 |

`x` **never contains discharge**, for any variant, for any basin. That is the leak the whole
design exists to avoid.

---

## 3. The shared encoder (all three variants)

One LSTM — 256 hidden units, 1 layer, forget-gate bias 3 — is applied to every basin's
sequence, and only the **last** hidden state is kept:

```
x  [N, 270, 36]  →  LSTM  →  h  [N, 256]
```

`h[i]` is a 256-number summary of "what basin *i* has been doing over the last 270 days,
given its weather and its physical attributes". The same weights are used for every basin;
basins differ only through their inputs.

**`h` is then sliced two ways, and this is the crux of the whole architecture:**

```
h  [531 × 256]
├── rows   1 … 455   gauged basins   →  the DONOR POOL  (keys and values)
└── rows 456 … 531   ungauged basins
    ┌───────────────────────────────────────────────────────────┐
    │ QUERIES = the rows you want a prediction for               │
    │   training  → rows 1…455  (only gauged basins have a loss) │
    │   inference → whichever rows you are predicting            │
    └───────────────────────────────────────────────────────────┘
```

The donor pool is **fixed by the fold** — always those same 455 gauged basins. It is *not*
"whatever happens to be in the batch". Queries and donors are two overlapping slices of one
matrix.

---

## 4. `vanilla` — no attention

```
ctx = h_q                                   # the query's own summary, nothing else
Q̂  = Linear(256→256) · ReLU · Dropout(0.4) · Linear(256→1)   → mm/day
```

Every basin is independent. Nothing about basin *j* can reach basin *i*. This is the
state-of-the-art baseline for prediction in ungauged basins.

---

## 5. `spatial` — borrow a *state* from similar basins

Queries and donors are compared in a learned 64-dimensional subspace:

```
Q = W_q · h_q        [n_query × 64]
K = W_k · h_d        [455 × 64]
A = softmax( Q Kᵀ / (√64 · τ) )      [n_query × 455]     τ = learned temperature
V = h_d                              [455 × 256]
c = A · V                            [n_query × 256]
ctx = h_q + LayerNorm(c)             # residual: the query's own state is always kept
```

Two rows of `A` are forced to zero before the softmax:

- **the self-diagonal** — a basin never attends to itself;
- **donors with no observation** in the 7-day window.

The rank-64 subspace matters: it gives attention a *similarity* space separate from the
space used to predict flow. Without it, "which basins resemble me" and "how much water comes
out of me" are forced to share one representation, which is what collapsed in the earlier
version of this work.

Note what is being borrowed: `V = h_d` is built from the **donors' weather and attributes**.
No discharge is involved anywhere in `spatial`.

---

## 6. `donor` — borrow a state *and* a recent flow signal

Identical to `spatial` except for one line:

```
V = h_d + MLP(Z_d)       MLP: Linear(14→32) · ReLU · Linear(32→256)
```

`figures/architecture.png` draws the same forward pass; there the projected term
`MLP(Z_d)` is written `q_d`.

### What `Z_d` is

For each gauged basin *d* and each of the 7 days `t-6 … t`:

```
logq  = log( Q_d + 0.01 · p_mean_d )         # basin-relative floor, safe for zero flow
Z_d   = ( logq − mean_d ) / std_d            # mean_d, std_d from the TRAINING period only
```

then `Z_d` is concatenated with a 0/1 observed-flag, giving `[455, 14]`.

**Why an anomaly and not raw discharge.** `Z_d` is dimensionless — it says *"I am 1.8
standard deviations above my own normal today"*. Raw m³/s would not transfer: a large wet
basin and a small dry one are not comparable. "Unusually wet **for me**" is. The mean and
standard deviation belong to the donor, which is gauged, so using them is legitimate.

### The division of labour

This is the part worth internalising:

> The **donors** tell the model *what kind of day this is* — dimensionless, transferable.
> The **query's own encoder** supplies *how much water that means* — its area, its rainfall,
> its soils.

That is why the model can predict an ungauged basin's absolute flow in mm/day without ever
knowing anything about that basin's hydrograph.

---

## 7. Worked example — one training step

455 gauged basins, target day *t*:

| step | shape | |
|---|---|---|
| `x` | `[455, 270, 36]` | ungauged basins are **not even encoded** during training |
| `h` | `[455, 256]` | |
| queries | rows 1…455 | every gauged basin is a query — they are the ones with a loss |
| donors | rows 1…455 | the same 455 |
| `A` | `[455, 455]` | diagonal masked → each basin draws on the other **454** |
| `c`, `ctx` | `[455, 256]` | |
| `Q̂` | `[455]` | flow on day *t* for all 455 |
| loss | scalar | basin-normalised squared error vs `Q[:455, t]` |

**Training is the ungauged problem, rehearsed 455 times per step.** Because the diagonal is
masked, every gauged basin must reconstruct its own discharge from *other* basins' discharge
plus everyone's weather — exactly the situation an ungauged basin is in. Nothing has to be
changed at inference for it to work; the model was never allowed to solve the easy version.

---

## 8. Worked example — inference

The donor pool is always the same 455 gauged basins. Only the *query* set changes.

### Case A — predicting a basin that **was** in training (gauged)

Its row index is < 455, so it **is** in the donor pool and the diagonal mask removes it from
its own attention row. It gets 454 donors.

> Without that mask it would retrieve its own `Z_d` and invert it straight back to its own
> discharge at NSE 1.000000 — a pure read-off. This is why the mask runs at **evaluation**
> too, not only in training.

### Case B — predicting **one brand-new ungauged** basin

Give it an index ≥ 455. The mask compares query index against donor index, and 455 never
equals anything below 455, so nothing is masked: **it sees all 455 donors.**

You must run **456 sequences** through the LSTM — the 1 query plus all 455 donors — because
attention needs every donor's `h_d` and every donor's `Z_d`. **You cannot predict a single
ungauged basin in isolation with `spatial` or `donor`.** Only `vanilla` can do that.

### Case C — predicting **20 new ungauged** basins

Encode 455 + 20 = **475** sequences per day; one forward pass returns all 20 predictions.

**The 20 do not influence each other.** Each attends only to rows 1…455, and none of the 20
is in that range. So predicting all 20 together is numerically identical to predicting each
one separately against the same donor pool — batching them is purely a speed choice.

### Case D — what `src/evaluate.py` actually does

Queries = all 531 rows, donors = the 455 gauged, for each of the 3,652 test days. One pass
per day. The 2×2 reporting cells are slices of that single output, never separate runs:

| cell | rows | meaning |
|---|---|---|
| `gauged/test` | 1…455 | out of sample in **time** — trained basins, unseen years |
| `ungauged/test` | 456…531 | out of sample in **space** — the real target |

---

## 9. What you need available at prediction time

| | `vanilla` | `spatial` | `donor` |
|---|---|---|---|
| query basin's 270-day forcings + attributes | ✔ | ✔ | ✔ |
| 455 donors' 270-day forcings + attributes | — | ✔ | ✔ |
| 455 donors' **observed discharge**, days `t-6 … t` | — | — | ✔ |
| stored per-donor log-flow mean and σ; feature scaler | — | ✔ | ✔ |

The last row of `donor` is its real operational cost: it needs a **live gauge network
reporting up to and including the prediction day**. That is realistic — you would have
today's readings from nearby gauges — but it means `donor` vs `vanilla` is an
**information** comparison, not an architecture one. `donor` sees 455 concurrent discharge
measurements the others never get.

---

## 10. Invariants, and what breaks if you drop them

| invariant | what goes wrong without it |
|---|---|
| self-diagonal masked in training **and** evaluation | a basin inverts its own `Z_d` back to its own flow — NSE 1.000000 |
| output is mm/day, never rescaled by the query's own flow statistics | injects the single most informative unknown in ungauged prediction |
| scaler, loss σ and `Z_d` statistics fit on **gauged basins, training period only** | test-period information leaks into training |
| feature columns asserted by name, never by count | if `qobs` survives the drop, NSE ≈ 0.99 and it looks like a triumph |
| LayerNorm only, never BatchNorm | BatchNorm couples samples in a batch — and here a batch is *all the basins* |

The query basin's own flow appears in exactly one place: `sigma` in the loss, which weights
its error term. It shapes the gradient and never enters the forward pass, and no loss is
computed at evaluation at all.

---

## 11. The three variants, side by side

| | `vanilla` | `spatial` | `donor` |
|---|---|---|---|
| attention | none | over 455 gauged | over 455 gauged |
| attended value | — | `h_d` | `h_d + MLP(Z_d)` |
| discharge used as input | none | none | donors' only |
| can predict one basin alone | **yes** | no | no |
| isolates | baseline | does borrowing a *state* help? | does borrowing a *flow signal* help? |

Everything else — 256 hidden, 270-day window, dropout 0.4, 30 epochs, the learning-rate
schedule, gradient clipping, the loss — is identical across all three, which is what stops
those choices from confounding the comparison.

---

## 12. The two later variants

### `donor_flow` — borrow the flow signal *only*

`donor` sums two things into one attended value, `V = h_d + MLP(Z_d)`, and then adds the
result back through a residual. That makes the two contributions impossible to attribute
separately. `donor_flow` takes them apart: the attended value becomes the donors' flow
anomaly itself, and it reaches the head by concatenation rather than a sum.

```
A   = softmax( Q Kᵀ / (√64 · τ) )        # unchanged
c   = A · Z[:455, t-6 : t+1]             # [n_query, 7]   the anomaly itself, not a projection
Q̂  = head( cat[ Dropout(h_q), c ] )      # 263 → 256 → ReLU → Dropout → 1
```

No `donor_mlp`, no `LayerNorm`, no residual add — so `ctx` in the sense used in §5 does not
exist here. Four details are deliberate:

- **Dropout is applied to `h_q` only, never to `c`.** At p = 0.4 it would zero about 3 of the
  7 donor-days, and 0 is a *meaningful* value in anomaly space — it means "exactly normal".
  Zeroing would not blur the signal, it would feed a plausible falsehood.
- **No LayerNorm on `c`.** Normalising a 7-vector subtracts the mean across the seven days,
  and that mean — "the region is running high right now" — is precisely the signal.
- **A donor is dropped unless it reported on all seven days** (`.all(1)`, where `donor` uses
  `.any(1)` plus flag bits). Without flag bits a gap would read as "exactly normal". This
  costs 0 donors on an average test day and 3 of 455 on the worst.
- **The head stays two layers.** Recovering flow from a log anomaly is roughly
  `exp(mu + sigma·z)`, a product a single linear layer cannot express.

The known risk, which is why the result was read carefully: attention now receives gradient
through a 7-dimensional value instead of a 256-dimensional one, and retrieval was already
the weak part of this model. It scored 0.747 against `donor`'s 0.743 on held-out basins —
a difference smaller than the spread between random starts. See `EXPERIMENTS.md`.

### `vanilla_lit` — the literature's recipe, not this project's

Every variant above is trained the same way so that none of those choices can explain a
difference between them (§1). That makes `vanilla` comparable to `spatial` and `donor`, but
*not* literally the standard recipe. `vanilla_lit` closes that gap and exists for one
purpose: to check this repository's plain model against published numbers on equal terms.

| | all other variants | `vanilla_lit` |
|---|---|---|
| a training step is | one date, all 455 gauged basins at once | 256 independently sampled (basin, window) pairs |
| forget-gate bias | 3 | 5 |

It is therefore the **deliberate exception** to the frozen-settings rule. It must not be
used as the baseline in an architecture comparison — `vanilla` is that baseline — and
`donor_flow` must not be compared against it.
