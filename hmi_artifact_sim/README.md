# HMI `OBS_VR` artifact simulation

Synthetic forward model of the SDO/HMI Doppler pipeline (MDI-like algorithm,
Couvidat et al. 2011) built to test one claim:

> a ~350 m/s oscillatory residual seen over sunspot umbrae, with a ~24 h period,
> is an **instrumental artifact** produced by inverting a Zeeman-split,
> Wilson-deformed line profile with a look-up table calibrated on the **quiet
> Sun**, modulated by the spacecraft radial velocity `OBS_VR(t)`.

Nothing is downloaded from JSOC; everything is computed from the equations.

## Running

```bash
python -m hmi_artifact_sim.tests        # 18 validation checks
python -m hmi_artifact_sim.experiments  # experiments A, B, C + sensitivity, writes figures/
```
Dependencies: `numpy`, `scipy`, `matplotlib`.

## Layout

| file | contents |
|---|---|
| `config.py` | every constant and every phenomenological parameter (`K_C`, `K_D`, `K_S`, `B_GAUSS`, `A_ORB`, …). Nothing is hardcoded elsewhere. |
| `profiles.py` | Module 1 — `gaussian_profile`, `wilson_parameters`, `zeeman_shift`, `umbral_profile` |
| `pipeline.py` | Modules 2–3 — sampling, Fourier coefficients, `velocity_from_profile`, `cog_velocity`, `LookupTable` |
| `artifact.py` | Module 4 — `measure_artifact` |
| `tests.py` | validation suite |
| `experiments.py` | the three experiments + sensitivity, writes `figures/` |

## Verdict

**The mechanism is real and reproduces the right order of magnitude, but it does
not need the profile-deformation calibration to be the headline — the headline
is that the deformation carries exactly zero real velocity and the pipeline
converts it into hundreds of m/s.**

| quantity | model | observed |
|---|---|---|
| real (centre-of-gravity) velocity bias of the deformed profile | `1.7e-8` m/s | — |
| 24 h artifact amplitude, spec look-up table (±7000 m/s) | **147 m/s** (294 m/s peak-to-peak) | ~350 m/s |
| 24 h artifact amplitude, extended table (±9500 m/s, no extrapolation) | **223 m/s** (445 m/s peak-to-peak) | ~350 m/s |
| period of the velocity artifact | **24.00 h exactly** | ~24 h |
| `B_LOS` contamination | 58 G peak-to-peak on a 2500 G field | — |

Against the spec's nominal ~1 m/s "naive geometric" benchmark this is an
amplification of **~150–220×, i.e. 2.2–2.3 orders of magnitude**, inside the
"2–3 orders of magnitude" band the spec set as the validation criterion.
Measured against the *actual* real velocity content of the deformed profile
(`1.7e-8` m/s, i.e. zero) the artifact is 100 % instrumental.

## Results in detail

### Experiment A — validation (`figures/expA_validation.png`, `figures/lookup_table.png`)

With `z_W = 0, B = 0` the artifact is `8.3e-8` m/s over
`v_true ∈ [-3, 3] km/s × OBS_VR ∈ [-3, 3] km/s` — acceptance criterion 8.1
passes with seven orders of margin. `B_LOS` is identically 0.

The uncorrected 6-point estimator is *not* the identity; its residual reaches
179 m/s over ±7 km/s and splits into two distinct instrumental effects:

* **6-point aliasing** — the 5th and 7th harmonics of the profile fold onto the
  1st. Peak 83 m/s, and it is what dominates at ordinary velocities
  (−12.2 m/s at `v = 2 km/s`, versus −0.09 m/s of leakage). Periodic in
  velocity with period `T·dv/dλ / 6 ≈ 3341` m/s.
* **finite-window leakage** — the 6 filters span exactly one "period"
  `T = 0.4128 Å`; past ~±4 km/s the Doppler-shifted line starts to fall out of
  that window. Peak 262 m/s, confined to the extremes.

The quiet look-up table inverts this to `9e-8` m/s. Its local gain
`dv_true/dv_apparent` ranges over **0.961 – 1.214**, so the inversion locally
stretches errors by up to 21 %.

### Experiment B — artifact vs Wilson depression (`figures/expB_wilson_sweep.png`, `figures/expB_map.png`)

**Structural result, and the most important one in this report: at
`OBS_VR = 0` the artifact is identically zero** (`1.8e-12` m/s) for *every*
`z_W`, not approximately zero. The Wilson + Zeeman deformation is symmetric in
wavelength and the 6 filters are symmetric about `λ0`, so the LCP and RCP
responses are exact mirror images and cancel in `V = (v_L + v_R)/2`. The quiet
look-up table is likewise an odd function, so it preserves that cancellation.

The literal Experiment B of the spec (`v_true = 0, OBS_VR = 0`) therefore returns
zero by construction. **The artifact cannot exist without a non-zero total
shift — `OBS_VR` is not merely the modulator of the artifact, it is its
enabling condition.** That is a strong point in favour of the hypothesis: the
mechanism predicts the 24 h signature rather than merely accommodating it.

Evaluated at the orbital maximum `OBS_VR = 3000 m/s`, `B = 2500 G`:

| `z_W` [km] | 0 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|
| artifact [m/s] | −37.2 | +16.9 | +74.4 | **+135.5** | +200.4 |

Monotone increasing, mean slope **0.297 (m/s) per km** of Wilson depression.
Contribution breakdown at `z_W = 600 km`:

| configuration | artifact |
|---|---|
| Zeeman only (`z_W = 0`, `B = 2500 G`) | −37.2 m/s |
| Wilson only (`B = 0`, `z_W = 600 km`) | −19.5 m/s |
| both | **+135.5 m/s** |
| real velocity bias of the same profile | `1.7e-8` m/s |

The two effects are strongly **non-additive** (−37 − 20 ≠ +136): the artifact is
a genuinely non-linear coupling of the Zeeman splitting to the profile
narrowing, not a sum of two biases.

### Experiment C — the 24 h signature (`figures/expC_24h_signature.png`)

`z_W = 600 km`, `B = 2500 G`, `v_true = 0`,
`OBS_VR(t) = 3000·sin(2πt/24h)` over 48 h.

* Velocity artifact: **fundamental period exactly 24.00 h**, amplitude
  **147 m/s** (spec table) or **223 m/s** (extended table). The waveform is
  distinctly non-sinusoidal: harmonics of 29 m/s at 4.8 h and 20 m/s at 8 h
  (only *odd* harmonics of the 24 h fundamental are present).
* `B_LOS`: 2496 → 2554 G, **58 G peak-to-peak — but at a 12 h period**, with a
  6 h harmonic.

**The velocity artifact is odd in `OBS_VR` and the magnetogram artifact is even
in `OBS_VR`.** So the model predicts a 24 h oscillation in umbral Dopplergrams
accompanied by a 12 h oscillation in the same pixels' magnetograms, in phase
quadrature-of-frequency with it. That is a falsifiable, cheap-to-test
signature that no real solar velocity field would produce, and it is the
sharpest discriminator this simulation offers.

### Extrapolation — the largest single uncertainty

With the spec's ±7000 m/s table, **33 % of the look-up calls in Experiment C are
extrapolations**: the table's apparent-velocity range only reaches ±6821 m/s,
while an LCP profile at `B = 2500 G` and `OBS_VR = 3000` presents an apparent
velocity near 8.4 km/s. Rebuilding the table over ±9500 m/s (still inside the
±10023 m/s unambiguous range) removes every extrapolation and **raises the
predicted amplitude from 147 to 223 m/s**.

This is precisely Couvidat's caveat that in active regions the correction
"becomes a basic extrapolation". The real pipeline uses
`hmi.lookup_corrected_expanded` — an *expanded* table — so the ±9500 m/s number
is the more relevant of the two, and the 147/223 m/s spread is an honest
systematic uncertainty of this model, not numerical noise (the LUT grid step is
17 m/s).

### Sensitivity (`figures/sensitivity.png`)

* **vs `B`**: not monotone. Below the phase-wrap threshold the amplitude is
  *structured*, with local maxima near 750, 1500 and 2450 G, because the Zeeman
  splitting slides the two polarisations along a non-linearity that is periodic
  in velocity with a 3341 m/s period. Above **B ≈ 3250 G** the split polarisation
  plus the orbital shift exceeds the ±10023 m/s unambiguous range, the
  first-harmonic phase wraps, and the algorithm saturates rather than merely
  biasing — the steep rise past that point should not be read as a prediction.
* **vs line narrowing**: the default 10 % narrowing at `z_W = 600 km` gives
  147 m/s; reaching the observed 350 m/s requires **≈ 20 % narrowing**, which is
  not an unreasonable umbral value but is not independently constrained here.

## What is robust and what is calibration

**Robust — pure instrument geometry, no free parameters:**

1. The 6-point Fourier estimator is non-linear, with a residual periodic in
   velocity (period 3341 m/s) and a finite-window breakdown past ~±4 km/s.
2. The quiet look-up table corrects that non-linearity *only for a quiet-width
   profile*; its local gain reaches 1.21.
3. A wavelength-symmetric umbral deformation produces exactly zero artifact at
   zero total shift, so the artifact is switched on by `OBS_VR` and inherits its
   period exactly.
4. `V` is odd and `B_LOS` is even in `OBS_VR` → 24 h in velocity, 12 h in the
   magnetogram.
5. Only the line **width** matters. `Ic` and `Id` cancel identically in the
   first Fourier harmonic (a constant continuum contributes nothing, and a
   global depth factor does not move the phase) — verified to `2.7e-12` m/s in
   `tests.py`. So `K_C` and `K_D` have **no effect whatsoever** on the velocity
   artifact in this model, and the entire Wilson calibration reduces to `K_S`.

**Calibration-dependent — refine before quoting a number:**

* `K_S`, the line narrowing per km of Wilson depression (10 % at 600 km here).
  The predicted amplitude grows a little faster than linearly in it
  (≈ power 1.25: 5 % → 63 m/s, 10 % → 147 m/s, 20 % → 354 m/s), so 20 %
  narrowing reproduces the observed 350 m/s exactly.
* The Gaussian profile shape itself. A real umbral Fe I 6173 in LCP is not a
  shifted Gaussian: it retains a π component and unsplit stray light, giving a
  double-humped profile with far more power at the 5th harmonic — exactly the
  harmonic that aliases. This model is therefore likely to **under**-predict.
* `A_orb = 3000 m/s` and the assumed sinusoidal `OBS_VR`.

## Deviations from the specification, and why

1. **`atan2(b1, a1)` → `atan2(-b1, -a1)`.** For an absorption line the first
   harmonic carries an intrinsic π phase, so the literal `atan2(b1, a1)` returns
   π for an unshifted line and Experiment A fails by a constant 10023 m/s.
   Couvidat's Eq. 6 uses `arctan(b1/a1)`, whose principal branch removes that π
   silently. We keep atan2's quadrant safety and remove the π explicitly. Found
   by running Experiment A first, exactly as §6 prescribes.
2. **`artifact = V_doppler − v_true` → `V_doppler − (v_true + OBS_VR)`.** With
   the literal definition, Experiment A returns `OBS_VR` instead of 0,
   contradicting its own acceptance criterion. The real pipeline removes the
   known spacecraft radial velocity from the Doppler map, so subtracting the
   full applied shift is the reading that makes Experiment A an identity test.
3. **Zeeman magnitude.** §2.2b predicts ~0.05–0.07 Å at 2500 G. The formula the
   spec itself gives, with `g = 2.5`, yields **0.1112 Å = 1.85 σ = 5402 m/s
   equivalent**. The code is correct and the spec's expected range is too low
   (0.05–0.07 Å corresponds to `g ≈ 1.2`, or to `B ≈ 1300 G`). 0.11 Å at 2500 G
   is the standard value and is why HMI saturates on strong umbral fields.
4. **Added a physical-truth estimator** (`cog_velocity`, centre of gravity over a
   ±1.2 Å window) and a **dense-window estimator** (`dense_window_velocity`).
   The spec's amplification comparison needed a reference for "the velocity an
   ideal instrument would report"; the centre of gravity supplies it exactly,
   and the dense-window variant separates aliasing from leakage.
5. **Added an extended look-up table** (±9500 m/s) because the spec's ±7000 m/s
   grid extrapolates on 33 % of calls in Experiment C (§9 asked for this to be
   documented; it turned out to matter at the 50 % level).

## Honest summary

The hypothesis survives. A profile deformation carrying **zero** real velocity
signal is turned by the HMI 6-point inversion into a **24 h oscillation of
150–220 m/s** with realistic umbral parameters, together with a 12 h, ~60 G
oscillation in the magnetogram. That is the right period, the right sign of the
argument, and within a factor of ~1.6–2.4 of the observed 350 m/s — reachable
inside the plausible range of the one calibration parameter that matters
(`K_S`), and likely under-predicted by the Gaussian-profile assumption.

What this model does **not** establish: that 350 m/s specifically is the
predicted number. The amplitude depends on `K_S`, on the true umbral profile
shape, and on the real table's velocity coverage. The next steps that would
settle it are a Milne-Eddington synthesis of the umbral Stokes profiles in place
of the shifted Gaussian, and validation against the actual
`hmi.lookup_corrected_expanded` table.

---

# Part II — the calculation with the **real** HMI profiles

Everything above uses a synthetic instrument: a Gaussian Fe I line sampled as
six delta functions, inverted with a look-up table we build ourselves. Part II
replaces each of those links with the real one from the JSOC pipeline and
re-runs the experiment, with the focus on what happens to the **Zeeman
correction**.

Notebook: **`notebooks/06_perfiles_reales.ipynb`** (in Spanish, step by step,
with pseudocode for each imported function).
Figures: `figures/R1…R10_*.png`.
Tests: `python -m hmi_artifact_sim.tests_real`.

## What changed

| | Part I (synthetic) | Part II (real) |
|---|---|---|
| sampling of the 6 filters | delta, `I_j = L(λ_j)` | integral `I_j = Σ T_j(λ)·L(λ)` with the six transmission profiles rebuilt per pixel |
| quiet-Sun Fe I line | Gaussian, `Id = 0.55`, FWHM ≈ 100 mÅ | the pipeline's own pseudo-Voigt + 2 wing Gaussians, with centre-to-limb scaling |
| phase → velocity scale | `T = 6·68.8 mÅ` ⇒ ±10 023 m/s | `pv1 = dvtune·(N−1)` ⇒ ±8 202 m/s |
| inversion table | built from the Gaussian | the **real** `hmi.lookup_corrected_expanded`, downloaded from JSOC, plus our reconstruction as a control |

Nothing in Part I was modified: `python -m hmi_artifact_sim.tests` still passes
18/18.

## New modules

| file | what it does |
|---|---|
| `jsoc_data.py` | downloads and caches every real input, and re-checks ten constants against the official `HMI_lookup.c` source |
| `real_filters.py` | rebuilds the 6 transmission profiles `T_j(λ)` per pixel (`HMI_lookup.c:1088-1094`), plus `mu_of_pixel` |
| `real_line.py` | the pipeline's quiet Fe I line (pseudo-Voigt + 2 Gaussians, calibrations 11/12/13) and the umbral deformation applied to it |
| `real_pipeline.py` | integration through the filters, the Fourier estimator at the real velocity scale, and both look-up tables |
| `zeeman.py` | the full Zeeman triplet (σ±, π) with field inclination γ |
| `real_artifact.py` | `RealPipeline`, the per-pixel/per-epoch model; `observables(...)` returns V, B_LOS, artifact, saturation count |
| `tests_real.py` | 35 checks, including the reproduction of the real LUT |

## Where the data comes from

Everything is public and needs no JSOC account. The cache lands in
`hmi_artifact_sim/data/` (~325 MB, git-ignored).

- **`github.com/JSOC-SDP/proj`**, subtree `lev1.5_hmi/`: `libs/lev15/HMIparam.h`
  (the 7 FSRs, the measured blocker S/N 11 and front window S/N 3, `ntest`,
  `dvtest`, `dlamdv`), `apps/*_June09_cal_128*.bin` (phases and contrasts of
  E2–E5 and contrasts of NB/WB/E1), and `apps/HMI_lookup.c` itself.
- **`hmi.phasemaps_extended[FSN]`** for the day's NB/WB/E1 phase maps, queried
  through the `jsoc_info` CGI and fetched straight from `/SUM*/`.
  Note: `hmi.phasemaps_corrected`, the series usually cited, is **not** reachable
  from outside Stanford; `phasemaps_extended` carries the same phases.
- **`hmi.lookup_corrected_expanded[FSN]`** for the real table itself.

Three epochs are compared, all with `HCAMID = 2` (side camera): FSN 23845068
(2011-06-22, cal 11), 65714753 (2013-12-18, cal 13) and 104314890 (2016-04-05,
cal 13).

## Three things only visible by reading the official source

1. **`pv1 = dvtune·(N−1)`, not `dvtune·N`** (`HMI_lookup.c:541`), with the
   literal comment `WARNING THIS IS WRONG, SHOULD BE dvtune*N`. That is a 5/6
   factor in the phase→velocity conversion, and it goes straight into the slope
   of the inversion.
2. **The phase sign is `atan2(−f1s, −f1c)`** (`HMI_lookup.c:1137`) — exactly
   what `pipeline.velocity_from_coefficients` already did. The docstring note
   there is now confirmed against the official code.
3. **`ReferenceFeLine.bin` is never read.** The repository ships R. Ulrich's
   line, but its read is commented out (`HMI_lookup.c:236-240`); the pipeline
   builds the line analytically instead. Using the `.bin` by mistake introduces
   a ~2.4 km/s bias, because its minimum sits at −62 mÅ rather than at line
   centre. This cost a debugging session and is worth flagging.

One more trap: the tuning positions that reproduce the table are the
`HWL4POS/HWL2POS/HWL1POS` of the **lookup** record, not the `HCMNB/HCMWB/HCME1`
of the phase map. They differ by one HCM step (6°) in several epochs.

## The validation that makes the rest trustworthy

The reconstructed table follows the real `lookup.fits` to

| epoch | cal | rms at disk centre | rms at μ ≈ 0.6 |
|---|---|---|---|
| 2011-06-22 | 11 | **3.3 m/s** | 28.4 m/s |
| 2013-12-18 | 13 | 12.7 m/s | 40.7 m/s |
| 2016-04-05 | 13 | 11.4 m/s | 16.1 m/s |

over the full ±9.8 km/s range. The residual even *identifies the calibration*:
for FSN 23845068 it is 3.3 m/s with cal 11 (what the record's `COMMENT`
declares), 80.5 m/s with cal 12 and 205.7 m/s with cal 13. From here on, the
sampling geometry is measured, not assumed.

## Results

| | synthetic | **real profiles** |
|---|---|---|
| Exp. A, max \|artifact\| | < 1 m/s | 7.4 m/s (grid step is 24 m/s) |
| Exp. B, slope at OBS_VR = 3000 | — | 0.187 (m/s) per km of z_W |
| Zeeman only, line at rest (z_W = 0, B = 2500 G) | −0.0 m/s | +0.1 m/s |
| Zeeman only, with OBS_VR = 3000 | — | +0.6 m/s |
| Zeeman + Wilson (600 km, 2500 G, OBS_VR = 3000) | — | **+113 m/s** |
| Exp. C, 24 h semi-amplitude, rigid Zeeman | 223 m/s | 161 m/s |
| Exp. C, full triplet, γ = 45° | — | **406 m/s** |
| Exp. C, full triplet, γ = 60° | — | **479 m/s** |
| Exp. C, `raw` estimator (no table) | — | 413 m/s |
| Exp. C, `delta` estimator | — | 236 m/s |
| observed residual | 350 m/s | 350 m/s |

### What this establishes

1. **The pipeline's Zeeman correction works when it should.** With the profile
   symmetric and at rest, the LCP/RCP mean cancels the splitting to better than
   2 m/s up to 3000 G, even with the real filters and their sidelobes, and
   `B_LOS` recovers the input field.

2. **The artifact is neither Zeeman nor Wilson alone: it is their product.**
   At `z_W = 0` the residual stays within a few m/s for any `B` and any
   `OBS_VR`. At `z_W = 600 km` and `B = 2500 G` it exceeds 100 m/s as soon as
   `OBS_VR` shifts the line. Neither ingredient on its own gets there.

3. **The 24 h signature appears at the right scale.** With the rigid shift the
   semi-amplitude is ~160 m/s, between a third and a half of the observed
   350 m/s. With the **full Zeeman triplet** and a field inclined 40–60° — the
   normal case for an umbra seen away from disk centre — it reaches 380–480 m/s
   and brackets the observed value. The π component is responsible: it stays at
   line centre and enters both polarisations with the same weight, so the mean
   cannot remove it.

4. **The magnetogram is contaminated by the same mechanism.** `B_LOS` oscillates
   on the *even* harmonics of the orbit while the Dopplergram oscillates on the
   *odd* ones, and its mean falls below the true field as γ grows.

5. **Correction to Part I: the inversion attenuates here, it does not amplify.**
   The `raw` estimator (no table) gives a *larger* amplitude than the full
   pipeline, 413 vs 161 m/s. The 6-point sampling non-linearity remains the
   origin of the residual, but the quiet look-up table reduces it rather than
   magnifying it. The `delta` estimator (236 m/s) shows the effect is not a
   filter-shape artefact either: an ideal 6-point instrument still produces it.

### What is still calibration, not prediction

- `k_c`, `k_d`, `k_s` are still phenomenological. This is the part to replace
  with a Milne-Eddington synthesis.
- The triplet is a superposition of components on a given absorption profile:
  no line saturation, no coupling of the source function to the field. At
  2500 G both already matter. It is nonetheless strictly better than the rigid
  shift.
- `A_ORB = 3000 m/s` and the sinusoidal `OBS_VR(t)` are representative, not
  SDO's actual ephemeris.

What *is* now measured rather than assumed: the six transmission profiles, the
quiet-Sun line, the phase→velocity scale with its `pv1 = dvtune·(N−1)`, and the
inversion table.
