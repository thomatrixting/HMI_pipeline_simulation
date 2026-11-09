"""
Modules 2-3 -- the MDI-like observables pipeline and the quiet look-up table.

The estimator is Couvidat's first Fourier harmonic of the sampled profile
(Eqs. 10-13) converted to a velocity through Eq. 6.  Everything is written for
a general number of samples N so that the *same* estimator can be run with
N = 6 (the real HMI instrument, aliased) and with N = 1024 (an "ideal
instrument", free of aliasing).  The difference between the two is exactly the
6-point sampling geometry, which is the robust part of the argument.

--- Note on atan vs atan2 (spec §3.3 / §9) -------------------------------------
The spec asks for atan2(b1, a1).  For an *absorption* line the first harmonic
carries an intrinsic pi phase (a1 < 0 at line centre), so atan2(b1, a1) returns
pi instead of 0 for an unshifted line and the identity test of Experiment A
fails by a constant DVDL*T/2 = 10023 m/s.  Couvidat's Eq. 6 uses arctan(b1/a1),
whose (-pi/2, pi/2) branch silently removes that offset.  We keep the quadrant
safety of atan2 and remove the absorption pi explicitly by evaluating
atan2(-b1, -a1), which is atan2(b1, a1) shifted by pi and correctly wrapped.
This was found by running Experiment A first, exactly as §6 prescribes.
"""

import numpy as np
from scipy.interpolate import interp1d

from . import config as cfg


# ---------------------------------------------------------------------------
# 3.1 Sampling
# ---------------------------------------------------------------------------
def filter_wavelengths(n=cfg.N_FILTERS, period=cfg.T_PERIOD, lambda0=cfg.LAMBDA0):
    """
    Sampling positions in order of *decreasing* wavelength (Couvidat: I0..I5).

    For n = 6 this is lambda0 + (2.5 - j)*DLAMBDA, j = 0..5.
    """
    j = np.arange(n)
    return lambda0 + ((n - 1) / 2.0 - j) * (period / n)


def _phases(n=cfg.N_FILTERS, harmonic=1):
    j = np.arange(n)
    return 2.0 * np.pi * harmonic * ((n - 1) / 2.0 - j) / n


def sample_points(profile_func, v, n=cfg.N_FILTERS):
    """
    Intensities measured by the n filters when the line is Doppler-shifted by v.

    A positive v (receding) red-shifts the line, so the rest-frame profile is
    evaluated at lambda_j - v/DVDL.  Broadcasts over an array of velocities:
    the returned array has shape (..., n).
    """
    lam_j = filter_wavelengths(n)
    dlam_v = np.asarray(v, dtype=float) / cfg.DVDL
    return profile_func(lam_j - dlam_v[..., None])


def sample_six_points(profile_func, v):
    """[I0, ..., I5] for the real 6-filter HMI sampling."""
    return sample_points(profile_func, v, cfg.N_FILTERS)


# ---------------------------------------------------------------------------
# 3.2 Discrete Fourier coefficients (Couvidat Eqs. 10-13)
# ---------------------------------------------------------------------------
def fourier_coefficients(I, n=None):
    """Return (a1, b1, a2, b2) from the sampled intensities."""
    I = np.asarray(I, dtype=float)
    n = I.shape[-1] if n is None else n
    p1, p2 = _phases(n, 1), _phases(n, 2)
    f = 2.0 / n
    a1 = f * np.sum(I * np.cos(p1), axis=-1)
    b1 = f * np.sum(I * np.sin(p1), axis=-1)
    a2 = f * np.sum(I * np.cos(p2), axis=-1)
    b2 = f * np.sum(I * np.sin(p2), axis=-1)
    return a1, b1, a2, b2


# ---------------------------------------------------------------------------
# 3.3 Velocity from the first coefficient (Couvidat Eq. 6)
# ---------------------------------------------------------------------------
def velocity_from_coefficients(a1, b1):
    """v = DVDL * (T/2pi) * phase, with the absorption-line pi removed."""
    phase = np.arctan2(-b1, -a1)
    return cfg.DVDL * (cfg.T_PERIOD / (2.0 * np.pi)) * phase


def velocity_from_profile(profile_func, v_true, n=cfg.N_FILTERS):
    """Full forward chain: sample at v_true -> Fourier -> apparent velocity."""
    I = sample_points(profile_func, v_true, n)
    a1, b1, _, _ = fourier_coefficients(I, n)
    return velocity_from_coefficients(a1, b1)


def dense_window_velocity(profile_func, v_true, n=cfg.N_IDEAL):
    """
    The same Fourier estimator, densely sampled inside the *same* T-wide
    window.  Aliasing is gone but the finite window is not, so this isolates
    the leakage (truncation) part of the instrumental non-linearity.
    Aliasing = velocity_from_profile(N=6) - dense_window_velocity.
    """
    return velocity_from_profile(profile_func, v_true, n=n)


def cog_velocity(profile_func, v_true, half_width=cfg.COG_HALF_WIDTH,
                 n=cfg.COG_N):
    """
    Physical-truth reference: the centre-of-gravity Doppler shift of the line,
    integrated over a window wide enough to contain the whole profile.

        v_cog = DVDL * int (Ic - I) (lam - lam0) dlam / int (Ic - I) dlam

    This is what an ideal spectrograph would report for the same profile, and
    it is exact for a Gaussian.  It defines the *real* velocity content of a
    deformed profile, against which the pipeline artifact is measured.
    """
    v_true = np.asarray(v_true, dtype=float)
    lam_s = cfg.LAMBDA0 + np.linspace(-half_width, half_width, n)
    I = profile_func(lam_s - (v_true[..., None] / cfg.DVDL))
    Ic = np.max(I, axis=-1, keepdims=True)
    w = Ic - I
    x = lam_s - cfg.LAMBDA0
    num = np.trapezoid(w * x, x=lam_s, axis=-1)
    den = np.trapezoid(w, x=lam_s, axis=-1)
    return cfg.DVDL * num / den


# Backwards-compatible alias used by artifact.py
def ideal_velocity_from_profile(profile_func, v_true, n=cfg.N_IDEAL):
    return dense_window_velocity(profile_func, v_true, n=n)


# ---------------------------------------------------------------------------
# 4. The quiet look-up table
# ---------------------------------------------------------------------------
class LookupTable:
    """
    Inverse of the quiet-Sun forward response: v_apparent -> v_true.

    Also records how often it is asked to extrapolate, which is physically
    meaningful (Couvidat: in active regions the correction "becomes a basic
    extrapolation").
    """

    def __init__(self, profile_func=None, v_min=cfg.LUT_V_MIN,
                 v_max=cfg.LUT_V_MAX, n=cfg.LUT_N):
        from .profiles import quiet_profile
        profile_func = quiet_profile if profile_func is None else profile_func
        self.v_true_grid = np.linspace(v_min, v_max, n)
        self.v_apparent_grid = velocity_from_profile(profile_func, self.v_true_grid)
        order = np.argsort(self.v_apparent_grid)
        self._x = self.v_apparent_grid[order]
        self._y = self.v_true_grid[order]
        self._f = interp1d(self._x, self._y, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
        self.range = (self._x.min(), self._x.max())
        self.n_extrapolated = 0
        self.n_calls = 0

    def __call__(self, v_apparent):
        v_apparent = np.asarray(v_apparent, dtype=float)
        lo, hi = self.range
        out = (v_apparent < lo) | (v_apparent > hi)
        self.n_calls += int(np.asarray(v_apparent).size)
        self.n_extrapolated += int(np.count_nonzero(out))
        return self._f(v_apparent)

    def derivative(self, v_apparent=None):
        """dv_true/dv_apparent -- the local amplification factor of the table."""
        if v_apparent is None:
            v_apparent = self._x
        v_apparent = np.asarray(v_apparent, dtype=float)
        h = 1.0  # m/s
        return (self._f(v_apparent + h) - self._f(v_apparent - h)) / (2.0 * h)


def build_lookup_table(**kw):
    return LookupTable(**kw)
