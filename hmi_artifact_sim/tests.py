"""
Validation tests.  Run:  python -m hmi_artifact_sim.tests
Every test prints its measured number so the result is auditable, not just
pass/fail.
"""

import numpy as np

from . import config as cfg
from .profiles import (gaussian_profile, quiet_profile, umbral_profile,
                       wilson_parameters, zeeman_shift, zeeman_shift_velocity,
                       make_umbral_profile)
from .pipeline import (filter_wavelengths, sample_six_points, cog_velocity,
                       fourier_coefficients, velocity_from_profile,
                       dense_window_velocity, LookupTable)
from .artifact import measure_artifact

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def main():
    print("=" * 78)
    print("VALIDATION TESTS")
    print("=" * 78)

    # -- constants ---------------------------------------------------------
    check("DVDL matches Couvidat", abs(cfg.DVDL - 48562.4) < 0.1,
          f"DVDL = {cfg.DVDL:.4f} m/s/A")
    check("T_PERIOD = 0.4128 A", abs(cfg.T_PERIOD - 0.4128) < 1e-12,
          f"T = {cfg.T_PERIOD:.6f} A")
    check("Unambiguous range ~ +-10 km/s", 9000 < cfg.V_UNAMBIGUOUS < 11000,
          f"+-{cfg.V_UNAMBIGUOUS:.1f} m/s")

    # -- filter grid -------------------------------------------------------
    lam = filter_wavelengths()
    expected = cfg.LAMBDA0 + np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5]) * cfg.DLAMBDA
    check("Filter positions (decreasing wavelength)",
          np.allclose(lam, expected), f"offsets/dl = {(lam - cfg.LAMBDA0)/cfg.DLAMBDA}")

    # -- Zeeman order of magnitude ----------------------------------------
    dlB = zeeman_shift(2500.0)
    check("Zeeman shift at 2500 G is comparable to sigma",
          0.5 * cfg.SIGMA_QUIET < dlB < 3 * cfg.SIGMA_QUIET,
          f"d(lambda)_B = {dlB*1000:.2f} mA = {dlB/cfg.SIGMA_QUIET:.2f} sigma"
          f"  ({zeeman_shift_velocity(2500.0):.0f} m/s equivalent)")

    # -- B = 0 and z_W = 0 collapse to the quiet profile --------------------
    lam_test = np.linspace(cfg.LAMBDA0 - 0.3, cfg.LAMBDA0 + 0.3, 101)
    same = all(np.allclose(umbral_profile(lam_test, 0.0, 0.0, pol),
                           quiet_profile(lam_test)) for pol in ('LCP', 'RCP'))
    check("umbral_profile(z_W=0, B=0) == quiet profile", same)

    # -- MODULE 2 MANDATORY TEST: quiet identity ---------------------------
    v_true = np.linspace(-6000, 6000, 2401)
    v_meas = velocity_from_profile(quiet_profile, v_true)
    resid = v_meas - v_true
    check("Doppler sign is correct (not inverted)",
          np.corrcoef(v_true, v_meas)[0, 1] > 0.99,
          f"corr = {np.corrcoef(v_true, v_meas)[0,1]:.6f}")
    check("Quiet 6-point response is near-identity",
          np.max(np.abs(resid)) < 100.0,
          f"max |v_app - v_true| = {np.max(np.abs(resid)):.2f} m/s "
          f"(this residual is the 6-point non-linearity the LUT corrects)")

    # -- physical-truth estimator IS the exact identity ---------------------
    v_cog = cog_velocity(quiet_profile, v_true)
    check("Centre-of-gravity (physical truth) estimator is the identity",
          np.max(np.abs(v_cog - v_true)) < 1e-3,
          f"max |v_cog - v_true| = {np.max(np.abs(v_cog - v_true)):.2e} m/s")

    # -- decomposition of the quiet non-linearity: aliasing vs leakage ------
    v_dense = dense_window_velocity(quiet_profile, v_true)
    leak = v_dense - v_true
    alias = v_meas - v_dense
    check("Non-linearity splits into 6-point aliasing + window leakage",
          np.max(np.abs(alias)) > 1.0,
          f"max |leakage| = {np.max(np.abs(leak)):.2f} m/s, "
          f"max |aliasing| = {np.max(np.abs(alias)):.2f} m/s; at v=2 km/s "
          f"leakage {np.interp(2000, v_true, leak):.2f} vs aliasing "
          f"{np.interp(2000, v_true, alias):.2f} m/s")

    # -- the deformation carries NO real velocity signal --------------------
    prof_L = make_umbral_profile(600.0, 2500.0, 'LCP')
    prof_R = make_umbral_profile(600.0, 2500.0, 'RCP')
    vs = np.linspace(-3000, 3000, 61)
    v_phys = 0.5 * (cog_velocity(prof_L, vs) + cog_velocity(prof_R, vs))
    check("Deformed profile carries zero real velocity bias",
          np.max(np.abs(v_phys - vs)) < 1e-2,
          f"max |v_cog(LCP,RCP mean) - v_shift| = {np.max(np.abs(v_phys - vs)):.2e} m/s "
          f"(z_W=600 km, B=2500 G)")

    # -- LUT inverts the quiet response ------------------------------------
    lut = LookupTable()
    v_corr = lut(v_meas)
    check("Look-up table inverts the quiet response",
          np.max(np.abs(v_corr - v_true)) < 1.0,
          f"max |LUT(v_app) - v_true| = {np.max(np.abs(v_corr - v_true)):.3e} m/s; "
          f"extrapolations = {lut.n_extrapolated}/{lut.n_calls}")

    # -- LUT grid resolution floor (spec §9) -------------------------------
    step = (cfg.LUT_V_MAX - cfg.LUT_V_MIN) / (cfg.LUT_N - 1)
    check("LUT grid step known and small vs expected artifact",
          step < 50.0, f"grid step = {step:.2f} m/s per point")

    # -- Ic and Id do not affect the measured velocity (analytic result) ----
    p1 = lambda l: gaussian_profile(l, 1.0, 0.55, 0.060)
    p2 = lambda l: gaussian_profile(l, 0.30, 0.90, 0.060)
    d = np.max(np.abs(velocity_from_profile(p1, v_true) -
                      velocity_from_profile(p2, v_true)))
    check("Measured velocity is independent of Ic and Id",
          d < 1e-8,
          f"max difference = {d:.2e} m/s (continuum cancels in the 1st harmonic; "
          f"only sigma matters)")

    # -- EXPERIMENT A: no deformation -> no artifact ------------------------
    lut_a = LookupTable()
    vt = np.linspace(-2000, 2000, 41)
    ov = np.linspace(-4000, 4000, 41)
    VT, OV = np.meshgrid(vt, ov, indexing='ij')
    art, _, blos = measure_artifact(0.0, 0.0, VT, OV, lookup=lut_a)
    check("EXPERIMENT A: |artifact| < 1 m/s with z_W=0, B=0",
          np.max(np.abs(art)) < 1.0,
          f"max |artifact| = {np.max(np.abs(art)):.3e} m/s over "
          f"v_true in [-2,2] km/s x obs_vr in [-4,4] km/s")
    check("EXPERIMENT A: B_los == 0 when B = 0",
          np.max(np.abs(blos)) < 1e-9, f"max |B_los| = {np.max(np.abs(blos)):.2e} G")

    # -- B_los calibration recovers the input field ------------------------
    lut_b = LookupTable()
    _, _, b_meas = measure_artifact(0.0, 1000.0, 0.0, 0.0, lookup=lut_b)
    check("B_los calibration recovers a weak input field",
          abs(float(b_meas) - 1000.0) / 1000.0 < 0.05,
          f"input 1000 G -> measured {float(b_meas):.1f} G")

    # -- LCP/RCP antisymmetry at zero shift --------------------------------
    lut_c = LookupTable()
    art0, _, _ = measure_artifact(600.0, 2500.0, 0.0, 0.0, lookup=lut_c)
    check("Artifact vanishes identically at zero total shift (structural)",
          abs(float(art0)) < 1e-6,
          f"artifact(z_W=600, B=2500, v_shift=0) = {float(art0):.2e} m/s")

    print("-" * 78)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed")
    return n_fail


if __name__ == '__main__':
    raise SystemExit(main())
