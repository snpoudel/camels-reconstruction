# Benchmarks — external reference points

Published results our numbers are read against. One section per study, same short template —
details stay in the linked paper and code, so do not expand these entries.

## Kratzert et al. (2019, WRR) — the standard PUB reference

> Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
> Kratzert, Klotz, Herrnegger, Sampson, Hochreiter & Nearing. *WRR* 55(12), 11344–11354.
> [doi](https://doi.org/10.1029/2019WR026065) · [kratzert/lstm_for_pub](https://github.com/kratzert/lstm_for_pub)

**Basins and inputs.** 531 CAMELS-US basins — the *same 531 gauge IDs* as our
`data/camels531.csv`, verified. 5 daily extended-NLDAS forcings (precip, shortwave radiation,
Tmax, Tmin, vapour pressure) + 27 static CAMELS attributes concatenated at every time step.
Hydrological signatures excluded — they are derived from observed discharge.

**Periods.** Train WY1981–1995, test WY1996–2010 — 15 water years each.

**Folds and seeds.** k-fold over basins, **k = 12** (~44 held out per fold), so every basin is
ungauged exactly once. **10 restarts** (seeds 300–309) with the partition regenerated per seed
— 120 models. Headline numbers are the **10-member ensemble mean hydrograph** per basin, not
averaged metrics; single members score lower.

**Architecture.** Plain LSTM with statics concatenated (not EA-LSTM): 1 layer, hidden 256,
dropout 0.4, seq_len 270, forget-gate bias 5, batch 2000, 30 epochs, Adam 1e-3 → 5e-4 @11 →
1e-4 @21, clip 1, basin-normalised NSE loss. Nearly identical to `src/config.py`.

| 531 basins, WY1996–2010 | out-of-sample in | median NSE |
|---|---|---|
| LSTM, k-fold PUB | **space + time** | **0.693** |
| LSTM, all basins gauged | time only | **0.744** |

**Being ungauged costs only ~0.05 NSE** (0.744 → 0.693) — the headroom an architecture change
competes for, and why the go/no-go bar here is +0.03.

**Not comparable to ours:** Daymet vs NLDAS forcings, different decade, 10- vs 15-year test
window, one fold of 76 basins vs 12-fold coverage of all 531, single runs vs a 10-member
ensemble — each shifts median NSE by about the effect size we are chasing. A target and a
sanity check on `vanilla`, not a score we are measured against.

**Data, if we reproduce it exactly later.** Inputs are all public and splits are
seed-deterministic, so the published seeds give the exact partitions: CAMELS v1.2 + attributes
v2.0 ([NCAR](https://ral.ucar.edu/solutions/products/camels)) and the extended NLDAS forcings
([HydroShare](https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/)).
**Their trained models and simulated hydrographs were never released** — only per-basin
ensemble metrics (`analysis/stats/pub_lstm.csv`, 531 rows), enough for a paired basin-by-basin
comparison but not a hydrograph one; the medians above were recomputed from those files and
reproduce the paper's abstract.
