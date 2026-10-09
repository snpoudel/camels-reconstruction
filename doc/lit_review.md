# Literature review — HESS technical note

Supporting literature for `tech_note_outline.md`. Every paper listed was read at least to its
abstract from a primary source (publisher or DOI page, HESS/ESSD, arXiv, PMC, NSF-PAR, USGS
pubs). Papers that couldn't be verified are in §9.

**Legend:**
- ★ **core:** in the note's reference list (23 references; see the outline's References).
- ○ **background:** not cited in the note; use only if a reviewer asks or a new sentence needs it.
- **[idx]:** a paywalled paper whose publisher abstract was read through OpenAlex or Semantic
  Scholar.

Volume, page and article numbers come from Crossref. Check each against the final PDF before
submission. "Use for" means the specific claim a paper supports, and no more.

---

## The story the literature supports

1. **Records are missing where they're needed.**
   - Daily records underpin water management, environmental flows and change detection.
   - Most reaches are ungauged.
   - Long-record gauges keep being discontinued (§1).
2. **Regional deep-learning models are the state of the art** for ungauged prediction.
   - They are now distributed as reconstruction and forecasting products (§2.1–2.2).
   - Since then, gains have come mainly from more data and from streamflow observations. At
     ungauged or discontinued sites, those observations can only come from neighbours (§2.3).
3. **Neighbouring gauges have always been the key resource:**
   - transferring their flows (§3.1);
   - transferring parameters (§3.2);
   - recently, feeding their flows into a **retrained** LSTM (§3.3).
4. **Model errors are spatially organised, whatever the model** (§4). That is what makes
   neighbours' errors informative.
5. **For process-based models, correcting a simulation with neighbouring gauges' same-day errors
   works.** It is operational in Quebec (§3.4). The authors note that it needs spatially
   correlated errors and that "any model can be used".
6. **The gap.** To our knowledge, no one has applied such a correction to a regional
   deep-learning model at large sample, or in both the ungauged and discontinued-gauge settings
   (§3.4). Nor has anyone asked whether a learned correction beats a simple one (§5).

## Citation map (outline section → references)

The note cites only these 23 references.

| Outline | References |
|---|---|
| Intro P1: need | Hannah et al. 2011; Sivapalan et al. 2003; Hrachowitz et al. 2013; Lanfear & Hirsch 1999 |
| Intro P2: deep-learning state of the art | Kratzert et al. 2019a, b; Nearing et al. 2024; Klotz et al. 2026; Gupta et al. 2026; Feng et al. 2020; Nearing et al. 2022 |
| Intro P3: neighbours | Archfield & Vogel 2010; Farmer 2016; Oudin et al. 2008; Wi et al. 2025 |
| Intro P4: errors are spatial; process-model precedent | Newman et al. 2015; Feng et al. 2020; Lachance-Cloutier et al. 2017; Matthews et al. 2025 |
| Methods | Newman et al. 2015; Addor et al. 2017; Kratzert et al. 2019a, b; Lachance-Cloutier et al. 2017; Nash & Sutcliffe 1970; Gupta et al. 2009; Clark et al. 2021 |
| Discussion | Newman et al. 2015; Lundquist et al. 2015; Feng et al. 2020; Lachance-Cloutier et al. 2017; Matthews et al. 2025; Wi et al. 2025; Nearing et al. 2024; Klotz et al. 2026 |

Everything else in this file is background. Cite it only if a reviewer asks or a new sentence
needs it.

---|---|
| Intro P1: need | Hirsch 1982; Poff et al. 2010; Hannah et al. 2011; Sivapalan et al. 2003; Hrachowitz et al. 2013; Blöschl et al. 2013; Krabbenhoft et al. 2022; Lanfear & Hirsch 1999; Ruhi et al. 2018 |
| Intro P2: deep-learning state of the art | Kratzert et al. 2019a, b; Arsenault et al. 2023; Nearing et al. 2024; Klotz et al. 2026; Gupta et al. 2026; Feng et al. 2020; Nearing et al. 2022; Heudorfer et al. 2025 |
| Intro P3: neighbours | Hughes & Smakhtin 1996; Archfield & Vogel 2010; Farmer 2016; Oudin et al. 2008; Parajka et al. 2013; Wi et al. 2025 |
| Intro P4: errors are spatial; process-model precedent | Newman et al. 2015; Knoben et al. 2020; Gründemann et al. 2026; Betterle et al. 2017a; Lundquist et al. 2015; Lachance-Cloutier et al. 2017; Martel et al. 2023; Matthews et al. 2025; Loonat et al. 2020 |
| Intro P5: gap | Poudel & Steinschneider 2026 |
| Methods | Newman et al. 2015; Addor et al. 2017; Kratzert et al. 2019a, b; Nash & Sutcliffe 1970; Gupta et al. 2009; Yilmaz et al. 2008; Clark et al. 2021 |
| Results and discussion | Feng et al. 2020; Knoben et al. 2019; Parajka et al. 2015; Patil & Stieglitz 2012; Kirschstein & Sun 2024; Vlah et al. 2024; Frame et al. 2021; Newman et al. 2015; Lundquist et al. 2015; Beck et al. 2020; Fan 2019; Coxon et al. 2015; Archfield & Vogel 2010 |

---

## 1. Why reconstruction is needed

### 1.1 Gauge networks are sparse, and records get cut short

| | Reference | What it actually says | Use for |
|---|---|---|---|
| ○ | Ruhi, Messager & Olden (2018), *Nature Sustainability* 1, 198–203. doi:10.1038/s41893-018-0047-7 | Stations reporting to global open datasets declined over four decades. In the US the decline reversed around 2000, but monitoring is volatile in more than a quarter of CONUS basins. | Global decline. Do **not** say the US network keeps shrinking. |
| ★ | Lanfear & Hirsch (1999), *Eos* 80(50), 605–607. doi:10.1029/99EO00406 [idx] | More than 100 USGS gauges with 30+ years of record were discontinued each year for lack of funding. | Discontinued long-record gauges. |
| ○ | Krabbenhoft et al. (2022), *Nature Sustainability* 5, 586–592. doi:10.1038/s41893-022-00873-0 | **Placement bias, not decline:** gauges sit on large perennial rivers in populated basins. | "Gauges are sparse and unrepresentative" only. |
| ○ | Norris (2009, rev. 2011), USGS Fact Sheet 2009–3020. doi:10.3133/fs20093020 | US gauges fell from about 8,300 (1970) to about 6,800 (1998); long-record gauges are discontinued by funding instability. | US discontinuations. |
| ○ | Kiang et al. (2013), USGS SIR 2013–5013. doi:10.3133/sir20135013 | US coverage is uneven: sparse with short records in the arid Southwest. | US coverage gaps. |
| ○ | Crochemore et al. (2020), *Hydrol. Sci. J.* 65(5), 699–711. doi:10.1080/02626667.2019.1659509 [idx] | Open data availability has fallen since the 1980s (21,586 series, 13 providers). | Decline in open data. |
| ○ | Elmi et al. (2024), *Scientific Data* 11, 240. doi:10.1038/s41597-024-03078-6 | Only 14% of GRDC gauges were active in 2020, partly because of reporting delay. | A global figure, with the caveat. |

### 1.2 Why continuous daily records matter

| | Reference | What it actually says | Use for |
|---|---|---|---|
| ○ | Hirsch (1982), *WRR* 18(4), 1081–1088. doi:10.1029/WR018i004p01081 [idx] | Record extension builds series for water-resources simulation and optimisation. Introduces MOVE. | Discontinued-gauge reconstruction is record extension. |
| ★ | Hannah et al. (2011), *Hydrol. Process.* 25(7), 1191–1200. doi:10.1002/hyp.7794 [idx] | A commentary: archives are threatened by declining coverage and truncated series. Long records are needed for drought and flood assessment, PUB and change detection. | General motivation; no numbers. |
| ○ | Poff et al. (2010), *Freshwater Biology* 55(1), 147–170. doi:10.1111/j.1365-2427.2009.02204.x [idx] | ELOHA: regional environmental-flow standards rest on modelled daily hydrographs. | Environmental flows. |

### 1.3 Prediction in ungauged basins (PUB)

| | Reference | Use for |
|---|---|---|
| ★ | Sivapalan et al. (2003), *Hydrol. Sci. J.* 48(6), 857–880. doi:10.1623/hysj.48.6.857.51421 [idx] | The PUB decade; also notes "existing measurement networks are declining". |
| ★ | Hrachowitz et al. (2013), *Hydrol. Sci. J.* 58(6), 1198–1255. doi:10.1080/02626667.2013.803183 | PUB review. |
| ○ | Blöschl et al. (eds.) (2013), *Runoff Prediction in Ungauged Basins*, Cambridge Univ. Press. doi:10.1017/CBO9781139235761 | PUB synthesis. Confirm the subtitle. |

---

## 2. Deep learning for ungauged prediction and reconstruction

### 2.1 State of the art

| | Reference | What it actually shows |
|---|---|---|
| ★ | Kratzert et al. (2019a), *HESS* 23, 5089–5110. doi:10.5194/hess-23-5089-2019 | One LSTM over 531 basins with static attributes beats regionally and locally calibrated models. The benchmark outputs (incl. HBV) were released with it. |
| ★ | Kratzert et al. (2019b), *WRR* 55(12), 11344–11354. doi:10.1029/2019WR026065 [idx] | k-fold PUB on 531 CAMELS basins: out-of-sample LSTM median NSE 0.69, vs 0.64 for basin-calibrated SAC-SMA. **Our ungauged baseline (0.69) reproduces this.** |
| ○ | Arsenault et al. (2023), *HESS* 27, 139–157. doi:10.5194/hess-27-139-2023 | The LSTM beats regionalised hydrological models at 93–97% of 148 ungauged catchments. |
| ★ | Nearing et al. (2024), *Nature* 627, 559–563. doi:10.1038/s41586-024-07145-1 | Global AI flood forecasting in ungauged basins, as reliable as GloFAS nowcasts; operational in 80+ countries. |
| ○ | Kratzert et al. (2018), *HESS* 22, 6005–6022. doi:10.5194/hess-22-6005-2018 | LSTMs beat SAC-SMA+Snow-17 on 241 CAMELS basins. |
| ○ | Feng et al. (2022), *WRR* 58(10), e2022WR032404. doi:10.1029/2022WR032404 [idx] | A differentiable HBV approaches the LSTM (NSE 0.73 vs 0.75, 671 basins). |
| ○ | Feng et al. (2023), *HESS* 27, 2357–2373. doi:10.5194/hess-27-2357-2023 | In PUB the LSTM and the differentiable model tie (0.65 vs 0.64); the differentiable model is better in ungauged *regions*. |
| ○ | Poudel & Steinschneider (2026), *HESS* 30, 5173–5193. doi:10.5194/hess-30-5173-2026 | Our own hybrid-model paper (its outputs are no longer used in the note). 16 parallel HBVs parameterised by a neural network, 531 CAMELS-US basins. Simpler designs do as well; **latitude and longitude alone generalise spatially about as well as full attribute sets.** |

### 2.2 Deep-learning reconstruction products

| | Reference | What it actually does |
|---|---|---|
| ★ | Klotz et al. (2026), *ESSD* 18(7), 5485–5504. doi:10.5194/essd-18-5485-2026 | EARLS: LSTM daily reconstructions with uncertainty for more than 10,000 European basins, 1953–2023. Uses **no** neighbour observations. |
| ★ | Gupta, Wi & Steinschneider (2026), *Scientific Data* 13, 624. doi:10.1038/s41597-026-07000-0 | 1951–2013 daily reconstruction at 656 Great Lakes gauges from a regional LSTM with climate, attributes and nearby gauges' runoff, because gauges "go in and out of service". |

### 2.3 What has improved LSTMs since

**Mostly matched rather than beat the LSTM:**
- ○ Liu, Bian, Lawson & Shen (2024), *J. Hydrol.* 637, 131389. doi:10.1016/j.jhydrol.2024.131389. Transformers at best match the LSTM on CAMELS.
- ○ Acuña Espinoza et al. (2024), *HESS* 28, 2705–2719. doi:10.5194/hess-28-2705-2024. Hybrids don't surpass the LSTM (CAMELS-GB).
- ○ Heudorfer, Gupta & Loritz (2025), *GRL* 52(6), e2024GL113036. doi:10.1029/2024GL113036 [idx]. Static attributes contribute little, especially out of sample.
- ○ Frame et al. (2021), *JAWRA* 57(6), 885–905. doi:10.1111/1752-1688.12964 [idx]. An LSTM post-processor of the target's own NWM output adds little and is worse in ungauged basins.

**Brought real gains (more data or observations):**
- ○ Kratzert et al. (2021), *HESS* 25, 2685–2703. doi:10.5194/hess-25-2685-2021. Three forcing products give +0.05 median NSE (gauged test only). The LSTM combines them differently by location.
- ○ Kratzert et al. (2024), *HESS* 28, 4187–4201. doi:10.5194/hess-28-4187-2024. More training basins help (no ungauged test).
- ○ Gauch, Mai & Lin (2021), *Environ. Model. Softw.* 135, 104926. doi:10.1016/j.envsoft.2020.104926. More basins and years help.
- ○ Feng, Lawson & Shen (2021), *GRL* 48(14), e2021GL092999. doi:10.1029/2021GL092999 [idx]. Ensembles and soft data reduce failures in ungauged regions.
- ★ Feng, Fang & Shen (2020), *WRR* 56(9), e2019WR026793. doi:10.1029/2019WR026793 [idx]. Lagged discharge as input gives median NSE 0.86, but needs **the target's own** recent flows.
- ★ Nearing et al. (2022), *HESS* 26, 5493–5513. doi:10.5194/hess-26-5493-2022. Autoregression and data assimilation (NSE 0.80 → 0.88) also need the target's own flows.

**What this supports in the note:**
1. Architecture changes have mostly matched the LSTM.
2. Attributes add little out of sample.
3. The largest gains come from streamflow observations. At ungauged or discontinued sites these
   can only come from neighbours.

Don't write that retraining "brings little improvement": it doesn't hold.

---

## 3. Using neighbouring gauges

### 3.1 Transferring observed flows (classical; no model)

| | Reference | What it actually does |
|---|---|---|
| ○ | Hughes & Smakhtin (1996), *Hydrol. Sci. J.* 41(6), 851–871. doi:10.1080/02626669609491555 [idx] | Patches and extends daily flow from several source sites through flow-duration-curve percentiles. |
| ★ | Archfield & Vogel (2010), *WRR* 46(10). doi:10.1029/2009WR008481 [idx] | Daily series are critical. The nearest gauge isn't always the best donor; choosing by map correlation does better. **Also the "distance-only" limitation.** |
| ★ | Farmer (2016), *HESS* 20, 2721–2735. doi:10.5194/hess-20-2721-2016 | Ordinary kriging of daily flows for historical records beats single-donor transfer. |
| ○ | Hirsch (1979), *WRR* 15(6), 1781–1790. doi:10.1029/WR015i006p01781 [idx] | Record reconstruction from nearby gauges (drainage-area ratio, regression). |
| ○ | Shu & Ouarda (2012), *WRR* 48(2). doi:10.1029/2011WR011501 [idx] | Distance-weighted multi-donor area-ratio method (109 Quebec stations): several distance-weighted donors beat one. IDW of **flows**. |
| ○ | Skøien & Blöschl (2007), *WRR* 43, W09419. doi:10.1029/2006WR005760 [idx] | Top-kriging of runoff series in Austria: daily NSE 0.87 vs 0.67 for a regionalised model. |
| ○ | Patil & Stieglitz (2012), *HESS* 16, 551–562. doi:10.5194/hess-16-551-2012 | Distance-based transfer of daily flows across 756 US basins: good in humid mountains (NSE > 0.7), poor in the dry West (< 0.3). |
| ○ | Parajka, Merz, Skøien & Viglione (2015), *J. Hydrol. Hydromech.* 63(3), 228–234. doi:10.1515/johh-2015-0024 | Top-kriging beats model regionalisation above about 2 stations per 1,000 km²: **gauge density decides**. |
| ○ | Farmer et al. (2015), USGS SIR 2014–5231. doi:10.3133/sir20145231 | 19 methods for historical daily flow (southeastern US); nearest-neighbour QPPQ did best. |
| ○ | Skøien, Merz & Blöschl (2006), *HESS* 10, 277–287. doi:10.5194/hess-10-277-2006 | Top-kriging method paper. |
| ○ | Harvey, Dixon & Hannaford (2012), *Hydrol. Res.* 43(5), 618–636. doi:10.2166/nh.2012.110 | 15 infilling methods for UK daily flows; donor choice matters. |
| ○ | Arriagada, Karelovic & Link (2021), *J. Hydrol.* 598, 126454. doi:10.1016/j.jhydrol.2021.126454 [idx] | Random-forest gap filling from other gauges (Chile). Needs the target's own overlapping record. |

### 3.2 Transferring model parameters (regionalisation)

| | Reference | What it actually does |
|---|---|---|
| ★ | Oudin et al. (2008), *WRR* 44, W03413. doi:10.1029/2007WR006240 [idx] | Spatial proximity is the best regionalisation in a dense network (913 French catchments). |
| ○ | Parajka et al. (2013), *HESS* 17, 1783–1795. doi:10.5194/hess-17-1783-2013 | Meta-analysis of 34 PUB studies: in dense networks, proximity and geostatistics win. |
| ○ | Parajka, Merz & Blöschl (2005), *HESS* 9, 157–171. doi:10.5194/hess-9-157-2005 | Kriging and similarity-based parameter transfer did best (320 catchments). |
| ○ | Pool, Vis & Seibert (2021), *WRR* 57, e2021WR030437. doi:10.1029/2021WR030437 [idx] | 19 regionalisation approaches over 671 US catchments. |
| ○ | Merz & Blöschl (2004), *J. Hydrol.* 287, 95–123. doi:10.1016/j.jhydrol.2003.09.028 | Proximity beats attribute-based transfer (308 Austrian catchments). **Abstract not read;** from secondary sources. |

### 3.3 Feeding neighbours' flows into a deep-learning model (retraining)

| | Reference | What it actually does | Relation to this note |
|---|---|---|---|
| ★ | **Wi, Gupta & Steinschneider (2025)**, *J. Hydrol.* 661, 133764. doi:10.1016/j.jhydrol.2025.133764 | About 200 Great Lakes gauges. An MLP picks 5 donors by predicted rank correlation, and an LSTM takes their flows, the correlations and the target's weather. It beats climate-only and donor-only LSTMs, especially at partially gauged and natural ungauged sites. It also beats a drainage-area-ratio baseline. | The same two settings, but donor information is built into a **retrained** model. This note adds a training-free step to **any existing** model's output; the two are complementary. |
| ★ | Gupta et al. (2026) (§2.2) | The Wi et al. model applied as a reconstruction product. | Same route. |
| ○ | Vlah et al. (2024), *HESS* 28, 545–573. doi:10.5194/hess-28-545-2024 | NEON reconstruction: regression on nearby donor gauges beat CAMELS-trained LSTMs (median KGE 0.81 vs 0.71). | Neighbours' observations carry information the LSTM lacks. The targets had field measurements. |
| ○ | Shokri, Bennett & Robertson (2026), egusphere-2026-2950 (**preprint**) | Fine-tunes a global LSTM's weights for each target on IDW-weighted donor basins (CAMELS-AUS). | Another retraining route; shares only IDW weighting, of training losses. Optional. |

### 3.4 Correcting a model with its errors at nearby gauges (closest prior art)

**Summary:**
- **Closest precedent:** Lachance-Cloutier et al. (2017), in a **process model**. It corrects a
  regionally calibrated HYDROTEL simulation at ungauged rivers by adding a weighted sum of
  same-day log errors at nearby gauges, with weights from statistical (optimal) interpolation.
  It is used operationally for historical reconstruction in Quebec.
- **Author's decision:** cite it as the process-model work the note builds on. The note is not
  framed as replicating or benchmarking against it.
- **Deep-learning case:** the only precedent found propagates a GNN's same-day residuals within
  one 5-gauge basin (Sun et al., 2022).
- **Not found:** an application to a regional deep-learning model at large sample, or one that
  separates the ungauged and discontinued-gauge settings. Write "to our knowledge".

**Side-by-side comparison**

Key: = same · ≈ partly · ≠ different.

| | This note | Lachance-Cloutier 2017 | Sun 2022 | Matthews 2025 | Kim 2026 | Loonat 2020 |
|---|---|---|---|---|---|---|
| What is transferred | log residual | = same-day log error | = residual | ≈ additive error vector | ≈ residual | ≈ long-term mean anomaly |
| Same-day | yes | = yes | = yes | ≈ hindcast steps, with persistence | ≠ lagged 1 day | ≠ long-term mean |
| Weights | IDW, sum to 1, × global λ | ≠ optimal interpolation, fitted correlation (L = 33.8 km) | ≠ graph conditioning | ≠ ensemble covariance | ≠ nested area ratio | ≠ kriging |
| Fades far from gauges | no | ≠ yes (→ model beyond ~200 km) | ≈ yes | ≈ yes | – | ≈ yes |
| Base model | LSTM | ≠ HYDROTEL | = deep learning (GNN) | ≠ LISFLOOD | ≠ HYMOD | ≠ AWRA-L |
| Setting | historical reconstruction | = historical reconstruction | ≈ simulation | ≠ forecast hindcasts | ≈ simulation | ≈ long-term bias |
| Scale | 531 basins, CONUS | 75 stations, Quebec | 1 basin, 5 gauges | 89 stations | 1 basin | 780 catchments |
| Verdict | – | **Closest precedent (process model)** | Same idea, small | Related | Related | Related |

**★ Lachance-Cloutier, Turcotte & Cyr (2017)**, *J. Hydrol.* 550, 294–306.
doi:10.1016/j.jhydrol.2017.05.011. **Full text read** (PDF in `doc/`; don't commit it).
- **Method:**
  - Daily ẑ_e = m_e + Σ w_i (z_i − m_i) on ln(specific discharge) (Eq. 15). m is the HYDROTEL
    simulation, and z_i − m_i is the same-day log error at gauge i.
  - The weights come from optimal interpolation with a fitted error correlation
    R(h) = 0.55 (1 + h/L) e^(−h/L), L = 33.8 km (Eqs. 16–17), and an observation-to-model
    error-variance ratio of 0.25.
  - The weights do not sum to 1; the correction converges to the model beyond ~200 km.
- **Model:** HYDROTEL regionally calibrated on 42 of 75 stations (1990–1999), run 1980–2009.
- **Evaluation:** leave-one-out over 75 stations, plus "near/far/smaller/larger" subsamples,
  scored with KGE′.
- **Results:**
  - Median KGE′: SI 0.86, ordinary and top-kriging 0.84; the raw model is worst.
  - SI improves the raw model at almost every station (one deterioration, −0.05).
  - The gain over the raw model is largest at "near" sites.
- **Quotable:**
  - "A condition for SI to produce good results is that the error of the background field is
    spatially correlated" (Sect. 4).
  - "In principle, any model can be used" as the first guess (Sect. 4).
  - The modelling effort behind a large-scale model "is likely to be a barrier to a more
    widespread use of data assimilation techniques" (Sect. 4).

**○ Martel et al. (2023)**, *J. Hydrol.* 623, 129777. doi:10.1016/j.jhydrol.2023.129777. Full
text read.
- Uses the optimal-interpolation reconstruction as the operational baseline in gauged and
  ungauged basins (95 basins, leave-one-out); it beat the raw model at 70–75% of basins.
- Notes it "does not provide correction for sites where the closest observation is too far".
- The same group's LSTM paper (Martel et al., 2025, *HESS* 29, 4951) applies no such correction
  to LSTM output.

**○ Sun, Jiang, Yang, Xie & Chen (2022)**, *HESS* 26, 5163–5184.
doi:10.5194/hess-26-5163-2022. Full text read.
- A GNN surrogate of the NWM, fine-tuned at 5 gauges. Same-day residuals are propagated to
  unobserved nodes by graph Gaussian conditioning (one hyperparameter, by leave-one-out).
- One HUC-8 basin. On an extra gauge, KGE rose from 0.06 to 0.89.

**★ Matthews, Cloke, Dance & Prudhomme (2025)**, *HESS* 29, 6157–6179.
doi:10.5194/hess-29-6157-2025. Full text read.
- LETKF-style ensemble post-processing of EFAS/LISFLOOD **forecast hindcasts**: an additive
  error vector spread from gauges to ungauged reaches through localised ensemble covariances.
- IDW is used only for the initial guess. 89 Rhine–Meuse stations, proxy-ungauged leave-one-out.
- Notes post-processing "is usually only applied at the locations of river gauges".

**○ Kim, Han & Kwon (2026)**, *Sci. Rep.* 16, 25245. doi:10.1038/s41598-026-49028-7. Full text
read.
- An upstream gauge's AR(1) residual at day t, area-scaled, corrects a downstream site at day
  t+1. HYMOD, one nested UK basin (NSE 0.74 → 0.80); not transferable to isolated basins.

**○ Loonat, van Dijk, Hutchinson & Weerts (2020)**, *WRR* 56(7), e2019WR026240.
doi:10.1029/2019WR026240. Full text read.
- Kriges the **long-term mean** model–observation anomaly (AWRA-L, 780 catchments) and removes
  41% of systematic bias.
- "Spatial model runoff estimates typically exhibit spatially correlated biases."
- Daily correction is left as future work.

**○ Bourgin et al. (2015)**, *HESS* 19, 2535–2546. doi:10.5194/hess-19-2535-2015. Full text read.
- Transfers neighbours' error **distributions** to set uncertainty bounds at ungauged
  catchments (907 French catchments). It does not correct the simulation.

**Not verified beyond the abstract:**
- Georgievová et al. (2023), *J. Hydrol.* 129373: a Kalman filter merges neighbour observations
  with a conceptual model (165 Czech catchments). Read it before citing.

### 3.5 Related corrections that are not day-by-day (optional)

| | Reference | What it actually does |
|---|---|---|
| ○ | Pugliese et al. (2018), *HESS* 22, 4633–4648. doi:10.5194/hess-22-4633-2018 | Top-kriged flow-duration curves used to quantile-map E-HYPE output at ungauged nodes. |
| ○ | Farmer, Over & Kiang (2018), *HESS* 22, 5741–5758. doi:10.5194/hess-22-5741-2018 | Bias-corrects simulated daily flows at ungauged sites with regionally estimated flow-duration curves (1,168 US gauges). |
| ○ | Farmer & Vogel (2016), *WRR* 52, 5619–5633. doi:10.1002/2016WR019129 [idx] | Model residuals carry information; ignoring them distorts flow distributions. |
| ○ | Skøien et al. (2016), *Proc. IAHS* 373, 109–114. doi:10.5194/piahs-373-109-2016; Skøien et al. (2021), *J. Hydrometeorol.* 22, 2731–2749. doi:10.1175/JHM-D-21-0008.1 | Top-kriging of forecast post-processing *parameters* to ungauged sites. |
| ○ | Hales et al. (2023), *J. Hydrol.* 626, 130279 [idx]; Bennett et al. (2022), *J. Hydrometeorol.* 23, 769–783 [idx] | Flow-duration-curve and quantile-mapping bias corrections transferred across reaches. |

### 3.6 Contrast: corrections that use the target's own observations or output

- ○ Refsgaard (1997), *Nordic Hydrol.* 28(2), 65–84. doi:10.2166/nh.1997.0005 [idx]. Forecast-updating procedures.
- ○ Xiong & O'Connor (2002), *Hydrol. Sci. J.* 47(4), 621–639. doi:10.1080/02626660209492964 [idx]. Simple AR error updating is as good as an ANN.
- ○ Liu et al. (2012), *HESS* 16, 3863–3887. doi:10.5194/hess-16-3863-2012. Review of hydrologic data assimilation.
- ○ Konapala et al. (2020), *ERL* 15(10), 104022. doi:10.1088/1748-9326/aba927. An LSTM learns SAC-SMA residuals at the same site.
- ○ Liu, Koch, Stisen, Troldborg & Schneider (2024), *HESS* 28, 2871–2893. doi:10.5194/hess-28-2871-2024. An LSTM corrects a national model at gauged and ungauged sites, learned from the target's own simulation.
- ○ El Ouahabi et al. (2026), *HESS* 30, 3549–3574. doi:10.5194/hess-30-3549-2026. A regional quantile random forest post-processor; transfers learned error relationships, not concurrent errors.
- Feng et al. (2020) and Nearing et al. (2022) (§2.3) need the target's own flows; Frame et al. (2021) post-processes the target's own NWM output.

---

## 4. Model errors are spatially organised

### 4.1 Performance has the same geography, whatever the model

| | Reference | What it actually says |
|---|---|---|
| ★ | Newman et al. (2015), *HESS* 19, 209–223. doi:10.5194/hess-19-209-2015 | CAMELS; SAC-SMA on 671 basins. "Poorly performing basins are most common along the high plains and desert southwest." |
| ○ | Knoben et al. (2020), *WRR* 56, e2019WR025975. doi:10.1029/2019WR025975 | Best-of-36-model KGE is "subject to strong spatial organization", lowest in the central plains and parts of the southwest. |
| ★ | **Feng, Fang & Shen (2020)** (§2.3) | **LSTM** on CAMELS: poor basins "concentrated along the Great Plains", the southwestern border, south Texas karst; good in the Northwest, Rockies, Appalachians and Atlantic states. Baseflow biases are suspected to come from "inadequate geologic descriptions". |
| ○ | Gründemann et al. (2026), *HESS* 30, 3439–3453. doi:10.5194/hess-30-3439-2026 | NWM v3.0, ~4,900 US basins: lowest in the drier centre, higher on the west coast, the mountains and east of the 100th meridian, "comparable with" earlier studies. |
| ○ | Lane et al. (2019), *HESS* 23, 4011–4032. doi:10.5194/hess-23-4011-2019 | Over 1,000 GB catchments: "all four models showed a similar spatial pattern of performance". |
| ○ | Mai et al. (2022), *HESS* 26, 3537–3572. doi:10.5194/hess-26-3537-2022 | 13 models, Great Lakes: poor in regulated, urban and agricultural areas. |

Don't cite Kratzert et al. (2019a) for performance maps: it has none.

### 4.2 Daily flows and model errors are correlated between nearby basins

| | Reference | What it actually says |
|---|---|---|
| ○ | Betterle, Schirmer & Botter (2017a), *WRR* 53(2), 1646–1663. doi:10.1002/2016WR019195 | Spatial correlation of daily flows is driven mainly by synchronous effective rainfall. |
| ○ | Betterle, Radny, Schirmer & Botter (2017b), *WRR* 53(12), 10354–10373. doi:10.1002/2017WR021144 [idx] | Rainfall frequency and intensity outweigh drainage differences (78 eastern-US pairs). |
| ○ | Loonat et al. (2020) (§3.4) | Model runoff biases are spatially correlated. |
| ★ | Lachance-Cloutier et al. (2017) (§3.4) | HYDROTEL's daily log errors correlate with distance (R ≈ 0.48 at 20 km, ≈ 0.11 at 100 km). Their measure differs from our Fig. 1, so don't compare the numbers. |

### 4.3 Why errors are shared (Discussion)

**Shared forcing errors:**
- ★ Newman et al. (2015), Sect. 4.5: Daymet has too little precipitation in some regions and a Pacific Northwest cold bias. Our LSTM uses Daymet.
- ★ Lundquist et al. (2015), *J. Hydrometeorol.* 16(4), 1773–1792. doi:10.1175/JHM-D-15-0019.1 [idx]: the same storms are missed by several gridded products at once. **The best evidence for same-day shared forcing errors.** Confirm the author list.
- ○ Beck et al. (2020), *J. Climate* 33(4), 1299–1315. doi:10.1175/JCLI-D-19-0332.1 [idx]: systematic underestimation over mountains and the Pacific coast.
- ○ Behnke et al. (2016), *Ecol. Appl.* 26(5), 1338–1351. doi:10.1002/15-1061 [idx]: gridded precipitation accuracy varies by ecoregion.

**Regional processes that attributes miss:**
- ○ Fan (2019), *WIREs Water* 6(6), e1386. doi:10.1002/wat2.1386 [idx]: leaky catchments.
- ○ Bouaziz et al. (2018), *HESS* 22, 6415–6434. doi:10.5194/hess-22-6415-2018: intercatchment groundwater flow clustered by geology.
- ○ Wenger et al. (2010), *WRR* 46, W09513 [idx]: larger errors at groundwater-influenced sites.
- Also Feng et al. (2020); Knoben et al. (2020); Lane et al. (2019) (§4.1).

**Observation errors are mostly local, so they add independent noise:**
- ○ Coxon et al. (2015), *WRR* 51(7), 5531–5546. doi:10.1002/2014WR016532 [idx]: "local conditions dominated" discharge uncertainty at 500 UK stations. A reason for K > 1 and λ < 1.
- ○ McMillan, Krueger & Freer (2012), *Hydrol. Process.* 26(26), 4078–4111 [idx]; McMillan, Westerberg & Krueger (2018), *WIREs Water* 5(6), e1319 [idx]: typical 10–40% uncertainties.

---

## 5. Does a flexible learner beat a simple one?

For the MLP-ties-IDW result. Each reference comes with a caveat.

- ○ Kirschstein & Sun (2024), *Proc. ICML*, PMLR 235, 24713–24725 (arXiv:2405.19836). GNNs don't benefit from river topology, and none beats an MLP baseline. *Caveat:* forecasting, where the target's own lagged flow is available.
- ○ Vlah et al. (2024) (§3.3). Donor regression beat LSTMs in 21 of 23 comparisons. *Caveat:* the targets had field measurements.
- ○ Nearing et al. (2022): simple autoregression beat variational data assimilation.
- ○ Frame et al. (2021): a learned post-processor was no better than a plain LSTM.
- ○ Huang et al. (2021), ICLR (arXiv:2010.13993): label propagation plus simple models beats GNNs. Computer science; at most one citation.
- **Counterpoint, report it:** Wi et al. (2025) found their learned donor model beat drainage-area-ratio transfer.

---

## 6. Metrics

**Check against the code:**
- `src/shared.py` computes the **original Gupta et al. (2009) KGE** (α = σs/σo). Don't cite Kling et al. (2012).
- `figures/evaluate.py` computes **day-matched percent bias on observed-flow percentile bands**: top 10%, 30–90%, bottom 30%, with no log transform. That is **not** Yilmaz et al.'s FHV/FMS/FLV. Define the bands and cite Yilmaz as inspiration only.

| | Reference | Use for |
|---|---|---|
| ★ | Nash & Sutcliffe (1970), *J. Hydrol.* 10(3), 282–290. doi:10.1016/0022-1694(70)90255-6 | NSE. |
| ★ | Gupta, Kling, Yilmaz & Martinez (2009), *J. Hydrol.* 377, 80–91. doi:10.1016/j.jhydrol.2009.08.003 | KGE as implemented. |
| ○ | Knoben, Freer & Woods (2019), *HESS* 23, 4323–4331. doi:10.5194/hess-23-4323-2019 | NSE and KGE aren't directly comparable: why ungauged KGE barely moves while NSE rises. |
| ★ | Clark et al. (2021), *WRR* 57(9), e2020WR029001. doi:10.1029/2020WR029001 [idx] | Sampling uncertainty of NSE and KGE: why differences under ~0.01 are ties. |
| ○ | Yilmaz, Gupta & Wagener (2008), *WRR* 44. doi:10.1029/2007WR006716 [idx] | Inspiration for the flow bands. Confirm article number W09417. |
| ○ | Ruzzante et al. (2026), *HESS* 30, 2337–2355. doi:10.5194/hess-30-2337-2026 | High NSE can hide poor interannual variance. |

---

## 7. Data and model sources

| | Reference | Notes |
|---|---|---|
| ★ | Newman et al. (2015); Addor et al. (2017), *HESS* 21, 5293–5313. doi:10.5194/hess-21-5293-2017 | CAMELS. Data DOIs 10.5065/D6MW2F4D and 10.5065/D6G73C3Q. |
| ★ | Kratzert et al. (2019a, b) | LSTM setup and ungauged k-fold design. |
| ○ | Poudel & Steinschneider (2026). Code: github.com/snpoudel/diff-hydro | **Not used in the note** (hybrid and HBV removed, 2026-10-08; keep in case they're added back). Hybrid outputs: Daymet; 106 basins held out (ungauged), 425 trained (discontinued). The author confirmed that the outputs and the 1980–1995 window used here are correct. |
| ○ | Seibert, Vis, Lewis & van Meerveld (2018), *Hydrol. Process.* 32(8), 1120–1125. doi:10.1002/hyp.11476; Kratzert (2019), CAMELS benchmark models, HydroShare. doi:10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1 | **Not used in the note.** HBV upper benchmark (ensemble of 100 basin-calibrated HBVs). **Maurer forcing; calibrated WY2000–2008, evaluated WY1990–1999**, so its "discontinued" case runs backwards in time. State this. |
| ○ | Shen et al. (2023), *Nat. Rev. Earth Environ.* 4(8), 552–567. doi:10.1038/s43017-023-00450-9 | Differentiable-modelling background. |

---

## 8. HESS technical-note conventions

**Recent notes** (word counts approximate, including captions):

| Note | Headings | Main text | Figures / tables | Abstract |
|---|---|---|---|---|
| Knoben et al. (2019), *HESS* 23, 4323 | Introduction / … / Conclusions | ~3.8k | 1 / 0 | ~240 |
| Nearing et al. (2022), *HESS* 26, 5493 | Introduction / Methods / Results / Conclusions and discussion | ~5.1k | 5 / 2 | ~95 |
| Acuña Espinoza et al. (2025), *HESS* 29, 1749 | Introduction / Data and methods / Results and discussion / Conclusions | ~4.1k | 2 / 0 | ~190 |
| Ruzzante et al. (2026), *HESS* 30, 2337 | Introduction / Methods / Data / Results and Discussion / Conclusions | ~6.9k | 4 / 2 | ~200 |
| Gründemann et al. (2026), *HESS* 30, 3439 | Introduction / Data and Methods / Results / Discussion / Conclusions | ~8.1k | 4 / 1 | ~320 |

**Patterns:**
- Merged sections are common.
- 2–5 figures and 0–2 tables.
- 4–8k words of main text.

**Required by HESS:**
- The "Technical note:" prefix.
- A non-technical short summary of ≤500 characters.
- An abstract without citations.
- Code and data availability, with DOIs.
- Author contributions.
- A competing-interests statement.
- Acknowledgements, describing any **AI use**.
- Financial support.

---

## 9. Checked and rejected, or not verified

**Looks relevant but isn't:**
- **Shokri et al. (2026)** as a residual correction: it fine-tunes LSTM weights (listed in §3.3 only as another retraining route).
- **Ghaneei et al. (2024)**, *WRR* 60(11): cluster fine-tuning; training-based.
- **Ma et al. (2021)**, *WRR* 57(5): transfer learning; training-based.
- **Krabbenhoft et al. (2022)** for "networks are declining": it's about placement bias.
- **Hannah et al. (2011)** for "2,500 US and 460 Canadian gauges discontinued": the figure wasn't found in the paper.
- **Ren et al. (2022)**, *HESS* 26: groundwater wells, not streamflow.
- **Janbain et al. (2023)**: a single station's water level.
- **Tursun et al. (2024)**: GloFAS inputs, not donor gauges.
- **Tang et al. (2023)**, *WRR*: an LSTM post-processor of global models with no neighbour residuals.
- **Kratzert et al. (2021, 2024)** as *ungauged* evidence: no PUB experiments.
- **Nearing et al. (2022), Feng et al. (2020)** as ungauged correction: they need the target's own flows.
- **Frame et al. (2021)** as a neighbour method: it uses the target's own NWM output.
- **El Gharamti et al. (2024)**, *HESS* 28: evaluated only at assimilated gauges.
- **Rakovec et al. (2012); Clark et al. (2008)**: state updating for forecasting.
- **Hunt et al. (2022); Tran et al. (2025); Choi & Kim (2025); Mbuvha et al. (2022)**: own-station or forecast post-processing.
- **Ramírez Molina et al. (2025); Moshe et al. (2020) HydroNets; Kratzert et al. (2021) EGU21 GNN abstract; Sun et al. (2021) GNN**: change the model or its inputs. Sun (2021)'s use of neighbour observations is unconfirmed.
- **Grey et al. (2025)**, *J. Hydrol.* 662: chooses between an LSTM and geostatistics per site; method details unconfirmed.
- **Fang & Shen (2020)**: soil moisture.
- **Farmer et al. (2018), Pugliese et al. (2018), Hales et al. (2023)** as concurrent-residual methods: they are distributional corrections (§3.5).
- **Le et al. (2022)**, hess-2022-320: rejected at HESS.
- **Kling et al. (2012)**: the code uses the 2009 KGE.
- **"Betterle et al. 2017, Spatial and temporal patterns…"**: no such title.
- **Newman et al. (2017); Addor et al. (2018)**: only weakly about spatial patterns.
- **Mishra & Coulibaly (2009); Fekete et al. (2012); Vogel & Stedinger (1985); Koya & Roy (2024); Willard et al. (2025); Jia & Benson (2020)**: verified but not needed.

**Not verified (don't cite without reading):**
- Fekete et al. (2015), *Science* 349.
- Farmer & Vogel (2013), *J. Hydrol.* 477.
- Fennessey (1994) thesis.
- Dembélé et al. (2019).
- Madsen & Skotner (2005).
- Booker & Woods (2013).
- Steinschneider, Yang & Brown (2015), *Hydrol. Sci. J.* 60(6). Its residuals may come from parameter regression.
- Henn et al. (2018).
- Georgievová et al. (2023) (abstract only).
