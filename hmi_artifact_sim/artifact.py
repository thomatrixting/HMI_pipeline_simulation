"""
Module 4 -- the artifact experiment.

The pipeline measures the apparent velocity of each circular polarisation on
the *deformed* umbral profile and then corrects it with the *quiet* look-up
table.  That mismatch is the whole mechanism.
"""

import numpy as np

from . import config as cfg
from .pipeline import (LookupTable, velocity_from_profile,
                       ideal_velocity_from_profile)
from .profiles import make_umbral_profile


def measure_artifact(z_W=0.0, B_gauss=0.0, v_true=0.0, obs_vr=0.0,
                     lookup=None, k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S,
                     ideal=False):
    """
    Returns (artifact_velocity, V_doppler, B_los).

    Parameters are broadcast, so v_true / obs_vr may be arrays.
    If `ideal` is True the 6-point sampling is replaced by dense sampling,
    giving the answer a perfect instrument would return for the same profile.
    """
    lookup = LookupTable() if lookup is None else lookup

    v_shift = np.asarray(v_true, dtype=float) + np.asarray(obs_vr, dtype=float)

    kw = dict(k_c=k_c, k_d=k_d, k_s=k_s)
    prof_L = make_umbral_profile(z_W, B_gauss, 'LCP', **kw)
    prof_R = make_umbral_profile(z_W, B_gauss, 'RCP', **kw)

    if ideal:
        vapp_L = ideal_velocity_from_profile(prof_L, v_shift)
        vapp_R = ideal_velocity_from_profile(prof_R, v_shift)
        vcorr_L, vcorr_R = vapp_L, vapp_R
    else:
        vapp_L = velocity_from_profile(prof_L, v_shift)
        vapp_R = velocity_from_profile(prof_R, v_shift)
        vcorr_L = lookup(vapp_L)
        vcorr_R = lookup(vapp_R)

    V_doppler = 0.5 * (vcorr_L + vcorr_R)
    B_los = (vcorr_L - vcorr_R) * cfg.KM

    # The real HMI pipeline removes the known spacecraft radial velocity
    # OBS_VR from the Doppler map, so the residual attributed to the Sun is
    #     artifact = V_doppler - (v_true + obs_vr).
    # (The spec writes `V_doppler - v_true`; with that literal definition
    # Experiment A would return obs_vr instead of 0, contradicting its own
    # acceptance criterion.  Subtracting the full applied shift is the reading
    # that makes Experiment A the identity test it is meant to be.)
    artifact_velocity = V_doppler - v_shift
    return artifact_velocity, V_doppler, B_los


def artifact_only(*args, **kw):
    return measure_artifact(*args, **kw)[0]
