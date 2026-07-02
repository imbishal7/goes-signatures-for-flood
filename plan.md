# Plan — richer GOES spectral feature set (brightness temperatures + BTDs)

## DECISION (2026-07-02) — being implemented now

**Goal: pure GOES/GLM signatures — climatology is REMOVED entirely** (no `ClimHead`, no
climatology feature). The whole point is to measure what the satellite alone can extract with
a richer, physically-motivated feature set. Committed set below.

**Bands read from NetCDF:** B8, B10, B11, B14, B15 (the ABI+GLM precip IR set; no B9/B13).
**NO full-resolution image is stored.** Everything is on the **50 km grid** (59×95):
- **Deep models** (3D-CNN / ConvLSTM / CNN+attn) convolve over the `(8, 19, 59, 95)` `seq`
  grid (+ `sum`, + lead) — a CNN over the feature grid gives spatial/neighbourhood context
  that XGBoost (per-cell independent) cannot. This is the clean "spatial vs per-cell" test.
- **Tabular** (XGB / LR) use the same features flattened per cell.

Cache is tiny (~3.6 MB/sample, ~9 GB total) — no storage problem, no /mnt/disk1, no /2 image.
Rationale: the full-res image branch added almost nothing over pooled features last run, so we
drop it and put the deep models on the feature grid instead.

**Per-cell `seq` (8 per frame, 8 frames → 8×8×59×95):** b14_min, frac_b14<220,
frac_b14<235, b8_min, btd_b10−8, btd_b14−15, glm_count, glm_energy.

**Per-cell `sum` (18 summaries → 18×59×95):** max_cool_b14, dt_b14_last3h, b14_trend_24h,
b14_trend_12h, n_cold_235, n_cold_220, max_consec_cold, cold_growth_6h, last6h_min_b14,
glm_daily, glm_max_3h, glm_hours_lit, glm_trend, nb3_min_b14, nb3_max_frac220, nb3_max_glm,
nb5_min_b14, nb5_max_glm.  (Neighborhood = cheap 3×3/5×5 grid filters — proxy for
"storm nearby/upstream"; full optical-flow motion deferred.)

Cell-branch input = 8×8 (seq) + 18 (sum) + 8 (lead) = **90 channels**, no climatology.
Cache arrays: `_img` (unchanged), `_seq`, `_sum`, `_t`, `_y`. **Full rebuild required.**
Dropped from the proposal (redundant/expensive): b14_p05/mean, frac<210, B13, optical-flow
motion/upstream, several BTDs.

---

## Original notes (motivation + full menu, for reference)

Motivation: our current run showed the raw imagery adds only a small increment over the
pre-pooled per-cell features + climatology. A more physically-motivated spectral set
(water-vapor bands + brightness-temperature differences) may carry more
precipitation/atmospheric-river signal than the current 7-channel stack.

## 1. Raw brightness temperatures (core ABI channels)

Use these emissive IR channels (all readable **day and night** — important, our samples
include night frames):

| band | meaning |
|------|---------|
| B8  | upper-level water vapor |
| B9  | mid-level water vapor |
| B10 | low-level water vapor |
| B11 | cloud-top phase |
| B13 | clean IR window |
| B14 | traditional IR window |
| B15 | dirty IR / split-window |

Rationale: the precipitation-retrieval paper selected **B8, B10, B11, B14, B15** (IR bands
sensitive to water vapor and precipitation); the atmospheric-river paper showed **B8–B10**
water-vapor imagery captures filament-like AR structures.

## 2. Brightness-temperature differences (BTDs)

Derive:

| BTD | meaning |
|-----|---------|
| B10 − B8  | vertical water-vapor structure |
| B9  − B10 | mid/low-level moisture gradient |
| B14 − B10 | cloud depth / window–vapor contrast |
| B11 − B14 | cloud phase, cirrus, ice/liquid distinction |
| B14 − B15 | split-window cloud/moisture signal |
| B13 − B15 | clean/dirty IR contrast |

Rationale: the ABI/GLM precipitation paper notes **BTD10−8 ≈ water-vapor
concentration/distribution** and **BTD11−14** helps distinguish thick/thin cirrus and
ice/liquid cloud properties.

Total proposed spectral stack: **7 raw + 6 BTD = 13 channels per frame.**

## 3. How to use them — two representations (both derivable from the same 13 fields)

Each band/BTD is a full CONUS field per frame, so they can feed the pipeline **two ways**:

- **(A) Image channels** for the CNN encoders (the deep models' image branch) — preserves
  spatial structure (AR filaments, cloud morphology). Would replace the current 7-channel
  `img` stack.
- **(B) Per-50 km-cell features** — pool each field into per-cell statistics (mean, min,
  cold-cloud fraction, gradient, temporal change) on the 59×95 grid, for the deep cell-branch
  **and** the tabular models (which have no image branch).

Current pipeline already does both (7 image channels + 4 pooled GOES cell features); this
plan enriches the spectral basis behind both.

## 4. Constraints / open questions (decide before building)

- **Cache rebuild required.** New raw bands to read from NetCDF: add B9, B13, B15 (currently
  read 8/10/11/14). Then recompute image stack + per-cell features + rebuild the whole cache.
- **Storage.** Image cache scales with channel count. Current 7-channel `img` ≈ 274 GB on the
  root NVMe. **13 channels ≈ ~510 GB → will NOT fit the NVMe.** Options: (a) store fewer
  image channels but derive all 13 as per-cell features; (b) coarser image downsample (/3
  or /4); (c) overflow to /mnt/disk4; (d) compute BTDs on-the-fly in the model from stored
  raw bands (store 7 raw, form the 6 BTDs at load time — no extra disk).
- **Which representation to prioritize?** Given imagery currently adds little over per-cell
  features, the cheaper first test may be to add the new bands/BTDs as **per-cell features**
  (feeds XGBoost + cell-branch) and measure lift before committing to a 13-channel image
  rebuild.
- **Redundancy check.** Some of these are near-duplicates (B13≈B14, B15 close to B14). Run the
  same univariate-signal + correlation screen we used before to trim before caching.
