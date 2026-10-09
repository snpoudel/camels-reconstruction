# HESS technical note — outline

Working outline. Citations are checked in `lit_review.md`, and [LR x.y] points to the section
there. Numbers come from the metrics files in `figures/`, at two decimals.

# Technical note: Improved streamflow reconstruction of deep learning model with spatial residual correction

**Format:** Introduction / Data and methods / Results / Discussion / Conclusions [LR 8].

**Target size:** about 3,750 words of main text (Intro 850, Methods 1,150, Results 1,000,
Discussion 550, Conclusions 200); 4 figures, 1 table; **23 references** (listed at the
end); a short Supplement.

## Storyline (the note in one paragraph)

Regional deep-learning models are now the best tool for reconstructing daily streamflow where
there is no gauge, and they are increasingly available as nationally to globally trained products. At an ungauged site
the model's error is unavailable. On the same day, though, it is *measured* at the nearby gauges,
and because model errors are spatially organised, those errors say something about the target.
Process-model studies have used this to correct simulations at ungauged rivers. Whether it still
works for a regional LSTM, which already learns from every basin at once, is not obvious, and to
our knowledge it has not been tested. We show it does:
- a three-parameter simple IDW method, training-free post-processing correction improves reconstruction at about 80% of 531
  basins, in both the ungauged and the discontinued-gauge setting;
- the gain is largest where a gauge is close;
- a more flexible learned correction cannot beat simple IDW method.

---

## Short summary (non-technical, ≤500 characters; required by HESS)

> Many rivers have no streamflow record, or a record that stopped. Machine-learning models provides
> state of the art estimate of the missing flows. These models performace has a regional pattern and therefore a shared error structure spatially. We correct a model's
> estimate at a river without a gauge by borrowing the errors it made the same day at nearby
> gauged rivers. This simple step needs no retraining, runs in seconds on any pretrained models output, and
> improved estimates at about 80% of 531 US rivers.

## Abstract (draft, 209 words, no citations)

> Regional deep-learning models are the state of the art to streamflow prediction.
> Their errors, however, are not independent from basin to basin: on a given day, nearby basins
> tend to be wrong in the same direction. We test whether this shared error can be removed after
> the fact, from the errors the model makes on the same day at nearby gauges. The correction
> weights the log errors of the K nearest gauges by inverse distance and removes a fraction of
> their weighted mean from the target's prediction. It trains nothing, needs no forcing data, and
> its three parameters are chosen by cross-validation that never reads the basins being
> corrected. For a regional LSTM over 531 CAMELS-US basins, median Nash–Sutcliffe efficiency
> rises from 0.69 to 0.75 at basins never seen in training and from 0.76 to 0.82 at gauges
> discontinued after training. It improves at about 80 % of basins, and low-flow bias roughly
> halves. Kling–Gupta efficiency improves mainly at discontinued gauges (0.79 to 0.83). Gains are
> largest where a gauge lies within about 20 km and fade with distance. A neural network that
> learns a day-varying correction from the same inputs does no better. Spatial residual
> correction is a cheap addition to deep-learning streamflow reconstruction that needs no
> retraining.

---

## 1. Introduction (~850 words)

Each paragraph opens with its topic sentence (in italics); the bullets are what it covers.

**P1. Need.** *Daily streamflow records are needed at many sites that have none.*
- **Uses:** water management, drought and flood assessment, and detecting change all need
  long, continuous daily records (Hannah et al., 2011).
- **Ungauged basins:** most river reaches have never been gauged. Predicting flow there is the
  long-standing PUB problem (Sivapalan et al., 2003; Hrachowitz et al., 2013).
- **Discontinued gauges:** many records are cut short. In the 1990s more than 100 long-record US
  gauges a year were being discontinued (Lanfear and Hirsch, 1999).
- **Framing:** both cases are *reconstruction*: estimating a historical record at a site without
  its own observations while nearby gauges did observe. Ungauged basins are the main case;
  discontinued gauges are the second.

**P2. Deep learning is the state of the art.** *Regional deep-learning models are now the best
tool for these sites, and are increasingly delivered as trained products.*
- **Performance:** regional LSTMs trained on hundreds of basins beat calibrated conceptual models,
  even at basins held out of training (Kratzert et al., 2019a, b).
- **Products:** they run globally in ungauged basins (Nearing et al., 2024) and underpin
  historical reconstruction products (Klotz et al., 2026; Gupta et al., 2026).
- **Observed streamflow is the strongest extra input** (Feng et al., 2020; Nearing et al., 2022).
  But those methods use the target's own recent flows, which reconstruction sites lack.
- **Point to land:** at these sites, the only observed streamflow is at the neighbours.

**P3. Neighbours.** *Neighbouring gauges have always been the main source of information at
unmonitored sites, but using them with deep-learning models has so far meant retraining.*
- **Classical routes:**
  - transferring their flows (Archfield and Vogel, 2010; Farmer, 2016);
  - transferring model parameters (Oudin et al., 2008).
- **Deep-learning route:** recent work feeds donor gauges' flows into a retrained LSTM, which
  improves reconstruction at ungauged and partially gauged sites (Wi et al., 2025).
- **The practical barrier:** that means rebuilding and retraining the model. Many users work with
  a model that is already trained, and lack the data, compute or access to retrain it.
  Retraining deep-learning models at scale is also costly. No citation is needed: this is a
  logical point.

**P4. A lighter route.** *Leave the model as it is, and correct its output with the errors it
makes at nearby gauges.*
- **Why this can work:** model skill and errors are spatially organised. Across CONUS, both
  conceptual models and LSTMs do worst in the Great Plains and the southwest (Newman et al.,
  2015; Feng et al., 2020).
- **It has worked for process-based models:**
  - Lachance-Cloutier et al. (2017) combined a regionally calibrated process model with the
    errors at nearby gauges to reconstruct daily flow at ungauged rivers.
  - Matthews et al. (2025) spread gauge errors to ungauged reaches when post-processing
    forecasts.
- **Bridge to P5:** such corrections need the model's errors to be spatially correlated
  (Lachance-Cloutier et al., 2017).

**P5. Gap and this note.** *Whether this works for a regional deep-learning model is not
obvious.*
- **The tension:**
  - a regional LSTM is trained on all basins at once, so it could already absorb the regional
    part of its error;
  - but it never sees the neighbours' observations on the day it predicts.
- **Gap:** to our knowledge, no study has corrected a deep-learning model's output this way, at
  large sample, or separated the ungauged and discontinued-gauge settings.
- **This note:**
  - a simple spatial residual correction: the K nearest gauges' same-day log errors,
    inverse-distance weighted, with one shrinkage factor, nothing trained;
  - tested on a regional LSTM over 531 CAMELS-US basins.
- **Questions, each answered in one Results section:**
  1. Are the LSTM's daily errors spatially structured? (3.1)
  2. Does the correction improve reconstruction in both settings, and does a more flexible,
     learned correction do better? (3.2)
  3. Where does it help most? (3.3)
- **Positioning (decided):** build on the process-model work. Claim novelty only as "to our
  knowledge, the first such test on deep-learning output at large sample".

---

## 2. Data and methods (~1,150 words)

### 2.1 Data and model
- **CAMELS-US:** 531 basins (Newman et al., 2015; Addor et al., 2017), Daymet forcing, daily USGS
  discharge.
- **Periods:** train on water years 1981–1995; test on 1996–2010 (Kratzert et al., 2019b).
- **LSTM:** set up as in Kratzert et al. (2019a, b), untuned on purpose.
  - Architecture: one layer of 256 units, dropout 0.4, a 365-day input sequence.
  - Training: 30 epochs; learning rate 1e-3 → 5e-4 → 1e-4; NSE* loss.
  - Inputs: six Daymet forcings and 30 static attributes (Table S1).
  - Output: the mean of 5 seeds.

### 2.2 Spatial residual correction
- **Lead-in:** the correction is applied to the finished LSTM output in two reconstruction
  settings, which decide which gauges can serve as donors.
- **Ungauged:**
  - 7 random folds of about 76 basins; for each fold, an LSTM is trained on the other 455.
  - Out of sample in space and time.
  - Donors: the fold's 455 training basins.
- **Discontinued gauge:**
  - One LSTM trained on all 531 basins.
  - Out of sample in time.
  - Donors: the other 530 basins, with the target's own record withheld (one gauge goes dark at a
    time).
- **Scope:** the correction needs neighbours reporting on the same days. It is reconstruction
  within a gauged region, so it is judged against the same model uncorrected, not against
  published PUB scores.

#### 2.2.1 Inverse-distance-weighted correction (Fig. 2a)
- **Equation:** for target i on day t, over the K nearest donors j that reported that day:
  - r_j(t) = ln(q̂_j + ε) − ln(q_j + ε);
  - w_ij ∝ d_ij^−p, normalised to sum to 1;
  - q̃_i(t) = (q̂_i + ε)·exp(−λ Σ_j w_ij r_j(t)) − ε.
- **The parameters:**
  - K: how many donors.
  - λ: how much of their error to believe.
  - p: how fast weight falls with distance.
- **Properties:** no forcing data, no training, milliseconds per basin; it reads only finished
  hydrographs and gauge coordinates, and never the target's own discharge.
- **Context, one sentence:** the correction follows the process-model approach of
  Lachance-Cloutier et al. (2017), here with inverse-distance weights and one shrinkage, so
  there is no error model to fit.
- **Choosing K, λ and p:**
  - Grid search maximising the median cross-validated KGE: K 1–39 (odd), λ 0–1 in steps of 0.05,
    p 0.5–3 in steps of 0.5.
  - Every basin in the search is corrected from all the others but itself.
  - Ungauged: within each fold's 455 training basins.
  - Discontinued: on the other six groups when group g is corrected.
  - The basins being corrected are never read.

#### 2.2.2 Learned correction with an MLP (Fig. 2b)
- **Purpose:** a more flexible benchmark. It tests whether a correction that can vary from day
  to day, and weight donors freely, does better than fixed distance weights.
- **Inputs:** the K donors' log residuals, missing-donor flags, log distances, day of year (sine
  and cosine), and the target's log simulated flow.
- **Output:** a daily log correction δ, with q̃ = (q̂ + ε)·exp(δ) − ε. IDW is the special case
  δ = −λ Σ w r.
- **Network and training:**
  - 2 × 64 SiLU units, dropout 0.2.
  - Loss 1 − KGE; Adam at 1e-3, cosine-decayed over 40 epochs.
  - Mean of 5 seeds.
- **K** is chosen from {1, 3, 5, 10, 15, 20, 30} on a held-back validation group.
- **Training data:** the same basins as the IDW search; the basins being corrected are never
  read.

### 2.3 Evaluation
- **Efficiency scores:** NSE (Nash and Sutcliffe, 1970) and KGE (Gupta et al., 2009, the
  original form).
- **Bias:**
  - Overall percent bias.
  - Percent bias on observed-flow bands: the top 10%, 30–90% and bottom 30% of days, defined
    explicitly (no citation needed).
- **Reporting:**
  - Medians, CDFs, and the share of basins improved.
  - Differences below about 0.01 count as ties, given sampling uncertainty (Clark et al., 2021)
    and seed spread.

---

## 3. Results (~1,000 words)

Facts only; interpretation goes to Sect. 4.

### 3.1 The LSTM's daily errors are spatially structured (Fig. 1) — Q1
- **Correlation with distance:** in the discontinued setting, daily residual correlation between
  basin pairs falls with distance:
  - r ≈ 0.54 at 20 km;
  - 0.43 at 50 km;
  - 0.29 at 100 km;
  - 0.19 at 200 km;
  - about 0.10 at 500 km.
- **Not just seasonality:** removing the seasonal cycle barely changes this (0.52, 0.41, 0.26,
  0.15, 0.07).
- **Closing line:** on a given day, nearby gauges observe a substantial part of the model's error
  at the target.

### 3.2 The correction improves reconstruction in both settings; a learned benchmark does no better (Table 1, Fig. 3) — Q2

| | Ungauged | Discontinued |
|---|---|---|
| Median NSE | 0.69 → 0.75 | 0.76 → 0.82 |
| Basins with NSE improved | 80% | 79% |
| Median KGE | 0.70 → 0.71 | 0.79 → 0.83 |
| Basins with KGE improved | 63% | 68% |
| Low-flow bias | +76% → +38% | +55% → +28% |
| Mid-flow bias | +21% → +9% | +17% → +7% |
| High-flow bias | −10% → −13% | −7% → −9% |

- **The whole distribution moves** (CDFs), not just the median.
- **Bias:**
  - Low- and mid-flow biases roughly halve.
  - Overall bias goes to about 0.
  - High-flow bias becomes slightly more negative.
- **KGE at ungauged basins** moves much less than NSE. Report which KGE component (r, α or β)
  holds it back.
- **Selected settings:** one sentence; per-fold values in Table S2.
- **The learned (MLP) benchmark:**
  - The medians tie at two decimals: NSE 0.74 vs 0.75 (ungauged) and 0.82 vs 0.82
    (discontinued); KGE 0.72 vs 0.71 and 0.83 vs 0.83.
  - Basin by basin it is mixed: the MLP is higher on KGE at 57–63% of basins and on NSE at 45%
    (ungauged) and 51% (discontinued).
  - Target attributes added to the MLP didn't help either (one fold, one sentence).

### 3.3 Gains are largest where a gauge is close (Fig. 4) — Q3
- **Median NSE gain by quartile of distance to the nearest donor:**

  | Nearest donor | Ungauged | Discontinued |
  |---|---|---|
  | Under ~20 km | +0.085 | +0.097 |
  | Second quarter | +0.053 | +0.064 |
  | Third quarter | +0.038 | +0.036 |
  | Over ~55–60 km | +0.015 | +0.007 |

  - Rank correlation between gain and distance: −0.34 (ungauged) and −0.48 (discontinued).
    Scatter in Fig. S2.
- **Geography:**
  - Baseline skill is lowest in the Great Plains and the southwest.
  - Describe where the gains fall. *To write from Fig. 4.*
- **Where it hurts (about 20% of basins):** report how these basins sit with respect to donor
  distance. *To check.*

---

## 4. Discussion (~550 words, an opening plus three paragraphs)

**Opening (2–3 sentences, no new citations).** *Reconstructing streamflow at sites without a
record matters, and regional deep-learning models are now the best tool for it. Yet they do not
use what nearby gauges observed on the same day. This note shows that adding that information
after the fact, with no retraining, improves their reconstructions.*
- Restate in one line what was found: shared errors (Sect. 3.1), consistent gains in both
  settings (Sect. 3.2), and the largest gains near gauges (Sect. 3.3).
- Then hand over to D1–D3: why the shared error exists, how this fits the literature and why
  a simple correction suffices, and practical use and limitations.

**D1. Why a jointly trained LSTM still leaves shared error.** *The model cannot see what nearby
gauges recorded on the day it predicts, so errors that nearby basins share stay in its output.*
- **Shared forcing errors:**
  - A storm that the gridded precipitation misses or misplaces affects every basin it covers on
    that day.
  - Daymet, the LSTM's forcing, has regional precipitation biases (Newman et al., 2015), and
    gridded products miss the same storms across a region (Lundquist et al., 2015).
- **Regional processes the static attributes describe poorly** (e.g. groundwater) leave
  regionally clustered biases (Feng et al., 2020).
- **Link to the results:** this is what Fig. 1 measures, and why the gain falls with donor
  distance (Sect. 3.3).

**D2. Neighbours' errors improve a deep-learning model, and a simple correction is enough.**
*Even a state-of-the-art deep-learning model can be improved by using the error pattern at donor
gauges. This is in line with what neighbour errors have done for process models, and with the
value of donor information for deep-learning reconstruction.*
- **In line with the literature:**
  - Correcting process models with neighbouring gauges' errors improved ungauged
    reconstruction, with the largest gains near gauges, as here (Lachance-Cloutier et al.,
    2017). It also improved ungauged forecast post-processing (Matthews et al., 2025).
  - Donor gauges' flows improve deep-learning reconstruction when built into the model (Wi et
    al., 2025). Here the same information helps without retraining.
- **The MLP had more to work with but did no better:** it could weight donors freely and vary its
  correction with season and flow (Sect. 3.2). Little signal is left beyond the
  distance-weighted average.
- **Several donors and partial trust:** each donor's error also contains its own local noise,
  such as observation error. Averaging several donors (K > 1) and believing only part of their
  error (λ < 1) damps it.

**D3. Practical use and limitations.**
- **Use:**
  - The correction runs on finished output in seconds, with no retraining and no access to the
    training pipeline. That suits pretrained deep-learning products used for reconstruction
    (Nearing et al., 2024; Klotz et al., 2026).
  - It is generic. As deep-learning models and training methods for streamflow prediction
    improve, this lightweight post-processing can simply be applied on top of whatever model
    comes next.
- **Limitations (only these three):**
  1. **Same-day neighbours are needed.** It applies to reconstruction, not to forecasting or
     long-term projections.
  2. **Tested on CAMELS-US only.** The gain depends on how close the donor gauges are (Sect. 3.3),
     so it may differ in regions or continents with a different gauge density.
  3. **It can hurt at some sites** (about 20% of basins lose NSE). This is likely because a full
     correction is applied even when the donors are far away or poorly related to the target,
     so it should be applied with care where gauges are sparse.

---

## 5. Conclusions (~150–200 words, one short paragraph)

A brief summary that answers the three questions, then gives a one-sentence outlook.
1. A regional LSTM's daily errors remain spatially correlated (r ≈ 0.3 at 100 km), despite joint
   training.
2. A three-parameter, training-free correction raises median NSE by about 0.06 in both settings
   and improves about 80% of basins. A more flexible, learned correction gives no further gain.
3. Gains are largest where a gauge is close.

**Outlook:** a cheap complement to deep-learning reconstruction products and donor-informed
models.

---

## End matter (required by HESS)
1. **Code and data availability:**
   - a Zenodo DOI for this repository;
   - CAMELS (doi:10.5065/D6MW2F4D, doi:10.5065/D6G73C3Q).
2. **Supplement:** link.
3. **Author contributions:** CRediT.
4. **Competing interests.**
5. **Acknowledgements:** including how AI tools were used.
6. **Financial support.**

---

## Figures and table

All are generated and placed in `doc/manuscript.md`. Figure numbers follow first citation:
the method diagram is cited in Methods, so it is Fig. 1.

| | Content | File (`figures/lstm/`) |
|---|---|---|
| **Fig. 1** | Method pipeline: shared steps 1, 2, 4; fork at 3a (IDW) vs 3b (MLP) | `fig01_method` (`make_lstm_method.py`) |
| **Fig. 2** | LSTM residual correlation vs distance, with and without seasonality | `fig02_error_structure` |
| **Fig. 3** | NSE and KGE CDFs, both settings | `fig03_cdf` |
| **Fig. 4** | LSTM NSE and ΔNSE after IDW and MLP, both settings | `fig04_nse_maps` |
| **Table 1** | Medians by setting: LSTM, +IDW, +MLP | in the manuscript |
| **Fig. S1–S2** | Boxplots of all metrics, ungauged / discontinued | `figS1_ungauged_boxplots`, `figS2_discontinued_boxplots` |
| **Fig. S3** | NSE gain vs nearest-donor distance | `figS3_gain_distance` |

## Supplement
- **Table S1:** LSTM inputs (forcings and static attributes).
- **Table S2:** Selected K, λ and p per fold, both settings.
- **Fig. S1:** Boxplots of all metrics, both settings (`figures/lstm/01_*`, `02_*`).
- **Fig. S2:** NSE gain vs distance to the nearest donor.

---

## References (core set, 23)

Every citation in this outline, and nothing else. Details are in `lit_review.md`; check each
against the final PDF.

1. Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. (2017). *HESS* 21, 5293–5313.
   doi:10.5194/hess-21-5293-2017
2. Archfield, S. A. and Vogel, R. M. (2010). *WRR* 46, W10513. doi:10.1029/2009WR008481
3. Clark, M. P., et al. (2021). *WRR* 57, e2020WR029001. doi:10.1029/2020WR029001
4. Farmer, W. H. (2016). *HESS* 20, 2721–2735. doi:10.5194/hess-20-2721-2016
5. Feng, D., Fang, K., and Shen, C. (2020). *WRR* 56, e2019WR026793.
   doi:10.1029/2019WR026793
6. Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F. (2009). *J. Hydrol.* 377, 80–91.
   doi:10.1016/j.jhydrol.2009.08.003
7. Gupta, R. S., Wi, S., and Steinschneider, S. (2026). *Sci. Data* 13, 624.
   doi:10.1038/s41597-026-07000-0
8. Hannah, D. M., et al. (2011). *Hydrol. Process.* 25, 1191–1200. doi:10.1002/hyp.7794
9. Hrachowitz, M., et al. (2013). *Hydrol. Sci. J.* 58, 1198–1255.
   doi:10.1080/02626667.2013.803183
10. Klotz, D., et al. (2026). *ESSD* 18, 5485–5504. doi:10.5194/essd-18-5485-2026
11. Kratzert, F., et al. (2019a). *HESS* 23, 5089–5110. doi:10.5194/hess-23-5089-2019
12. Kratzert, F., et al. (2019b). *WRR* 55, 11344–11354. doi:10.1029/2019WR026065
13. Lachance-Cloutier, S., Turcotte, R., and Cyr, J.-F. (2017). *J. Hydrol.* 550, 294–306.
    doi:10.1016/j.jhydrol.2017.05.011
14. Lanfear, K. J. and Hirsch, R. M. (1999). *Eos* 80, 605–607. doi:10.1029/99EO00406
15. Lundquist, J. D., et al. (2015). *J. Hydrometeorol.* 16, 1773–1792.
    doi:10.1175/JHM-D-15-0019.1 (confirm the author list)
16. Matthews, G., Cloke, H. L., Dance, S. L., and Prudhomme, C. (2025). *HESS* 29, 6157–6179.
    doi:10.5194/hess-29-6157-2025
17. Nash, J. E. and Sutcliffe, J. V. (1970). *J. Hydrol.* 10, 282–290.
    doi:10.1016/0022-1694(70)90255-6
18. Nearing, G. S., et al. (2022). *HESS* 26, 5493–5513. doi:10.5194/hess-26-5493-2022
19. Nearing, G., et al. (2024). *Nature* 627, 559–563. doi:10.1038/s41586-024-07145-1
20. Newman, A. J., et al. (2015). *HESS* 19, 209–223. doi:10.5194/hess-19-209-2015
21. Oudin, L., et al. (2008). *WRR* 44, W03413. doi:10.1029/2007WR006240
22. Sivapalan, M., et al. (2003). *Hydrol. Sci. J.* 48, 857–880.
    doi:10.1623/hysj.48.6.857.51421
23. Wi, S., Gupta, R. S., and Steinschneider, S. (2025). *J. Hydrol.* 661, 133764.
    doi:10.1016/j.jhydrol.2025.133764

---

## Open items before drafting
1. **Make Fig. 1 (LSTM-only) and Fig. S2.**
2. ~~Check the basins where the correction hurts, and the limiting KGE component.~~ Done and
   written into `manuscript.md`:
   - hurt basins sit farther from donors (median 57 vs 32 km);
   - the ungauged KGE gain is limited by α (0.92 → 0.90).
3. **Confirm bibliographic details:** the Lundquist et al. (2015) author list.
4. **Time-split check, deferred:** tune on 1996–2002, score on 2003–2010.
