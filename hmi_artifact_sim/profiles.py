"""
Module 1 -- synthetic Fe I 6173 line profiles.

Two ingredients, kept separable so each can be switched off independently:

  (a) Wilson depression : modifies (Ic, Id, sigma) of the Gaussian.
  (b) Zeeman splitting  : displaces the line by +/- d(lambda)_B for LCP/RCP.

With z_W = 0 and B = 0 the umbral profile reduces *exactly* to the quiet one.
"""

import numpy as np

from . import config as cfg


# ---------------------------------------------------------------------------
# 2.1 Quiet Gaussian profile (Couvidat Eq. 5)
# ---------------------------------------------------------------------------
def gaussian_profile(lam, Ic=cfg.IC_QUIET, Id=cfg.ID_QUIET,
                     sigma=cfg.SIGMA_QUIET, lambda0=cfg.LAMBDA0):
    """I(lam) = Ic - Id * exp(-(lam - lambda0)^2 / sigma^2)."""
    lam = np.asarray(lam, dtype=float)
    return Ic - Id * np.exp(-((lam - lambda0) ** 2) / sigma ** 2)


def quiet_profile(lam):
    """The quiet-Sun profile with the default configuration parameters."""
    return gaussian_profile(lam)


# ---------------------------------------------------------------------------
# 2.2a Wilson depression -> deformed (Ic, Id, sigma)
# ---------------------------------------------------------------------------
def wilson_parameters(z_W, k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S):
    """Phenomenological scaling of the profile parameters with z_W [km]."""
    Ic = cfg.IC_QUIET * (1.0 - k_c * z_W)
    Id = cfg.ID_QUIET * (1.0 - k_d * z_W)
    sigma = cfg.SIGMA_QUIET * (1.0 - k_s * z_W)
    return Ic, Id, sigma


# ---------------------------------------------------------------------------
# 2.2b Zeeman splitting
# ---------------------------------------------------------------------------
def zeeman_shift(B_gauss, lande_g=cfg.LANDE_G, lambda0=cfg.LAMBDA0):
    """Longitudinal Zeeman displacement d(lambda)_B in Angstrom."""
    return cfg.ZEEMAN_C * lande_g * lambda0 ** 2 * B_gauss


def zeeman_shift_velocity(B_gauss, **kw):
    """The same shift expressed as an equivalent Doppler velocity [m/s]."""
    return cfg.DVDL * zeeman_shift(B_gauss, **kw)


# ---------------------------------------------------------------------------
# Combined umbral profile
# ---------------------------------------------------------------------------
def umbral_profile(lam, z_W=0.0, B_gauss=0.0, polarization='LCP',
                   k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S):
    """
    Deformed profile seen by one circular polarisation.

    LCP sees the line centred at lambda0 + d(lambda)_B, RCP at lambda0 - d(lambda)_B.
    For B = 0 both coincide; for z_W = 0 the shape is the quiet one.
    """
    Ic, Id, sigma = wilson_parameters(z_W, k_c, k_d, k_s)
    dlB = zeeman_shift(B_gauss)
    if polarization.upper() == 'LCP':
        centre = cfg.LAMBDA0 + dlB
    elif polarization.upper() == 'RCP':
        centre = cfg.LAMBDA0 - dlB
    else:
        raise ValueError("polarization must be 'LCP' or 'RCP'")
    return gaussian_profile(lam, Ic, Id, sigma, centre)


def make_umbral_profile(z_W=0.0, B_gauss=0.0, polarization='LCP', **kw):
    """Return a callable lam -> I(lam) for the given umbral configuration."""
    def _p(lam):
        return umbral_profile(lam, z_W, B_gauss, polarization, **kw)
    return _p
