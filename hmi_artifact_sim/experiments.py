"""
The three experiments of spec §6, plus the sensitivity analysis they imply.

Run:  python -m hmi_artifact_sim.experiments
Figures are written to hmi_artifact_sim/figures/ and every number quoted in the
README is printed to stdout.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as cfg
from .profiles import quiet_profile, make_umbral_profile, zeeman_shift, \
    zeeman_shift_velocity, wilson_parameters
from .pipeline import LookupTable, velocity_from_profile, \
    dense_window_velocity, cog_velocity, filter_wavelengths
from .artifact import measure_artifact

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')

# --- Design tokens (validated categorical palette; light surface) ----------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#dedcd6'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
SEQ = ['#c3d9f5', '#8fbaeb', '#5a98e0', '#2a78d6', '#1a4f92']   # single-hue blue
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_gray_orange', ['#1a4f92', '#2a78d6', '#8fbaeb', '#e8e8e6',
                         '#f5b48f', '#eb6834', '#a13c17'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'text.color': INK,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.grid': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'legend.frameon': False, 'lines.linewidth': 2.0, 'figure.dpi': 130,
})


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"    figure -> {os.path.relpath(path)}")


def _title(ax, title, subtitle=None):
    ax.set_title(title, loc='left', pad=22 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK2, va='bottom')


REPORT = {}


# ===========================================================================
# Experiment A -- validation
# ===========================================================================
def experiment_A(lut):
    print("\n" + "=" * 78)
    print("EXPERIMENT A -- validation (z_W = 0, B = 0)")
    print("=" * 78)

    v = np.linspace(-7000, 7000, 1401)
    v_app = velocity_from_profile(quiet_profile, v)
    v_dense = dense_window_velocity(quiet_profile, v)
    v_cog = cog_velocity(quiet_profile, v)
    v_corr = lut(v_app)

    leak = v_dense - v
    alias = v_app - v_dense

    grid = np.linspace(-3000, 3000, 61)
    VT, OV = np.meshgrid(grid, grid, indexing='ij')
    art, _, blos = measure_artifact(0.0, 0.0, VT, OV, lookup=lut)

    print(f"  max |v_apparent - v_true|            = {np.max(np.abs(v_app-v)):.2f} m/s")
    print(f"  peak magnitude of each component (they partly cancel):")
    print(f"     finite-window leakage             = {np.max(np.abs(leak)):.2f} m/s")
    print(f"     6-point aliasing                  = {np.max(np.abs(alias)):.2f} m/s")
    k2 = int(np.argmin(np.abs(v - 2000)))
    print(f"     at v = 2000 m/s: leakage {leak[k2]:+.2f}, aliasing {alias[k2]:+.2f} m/s"
          f"  -> aliasing dominates at ordinary velocities")
    print(f"  max |v_cog - v_true| (physical truth)= {np.max(np.abs(v_cog-v)):.2e} m/s")
    print(f"  max |artifact| after the LUT         = {np.max(np.abs(art)):.2e} m/s")
    print(f"  max |B_los|                          = {np.max(np.abs(blos)):.2e} G")
    print(f"  ACCEPTANCE 8.1 (|artifact| < 1 m/s)  : "
          f"{'PASS' if np.max(np.abs(art)) < 1.0 else 'FAIL'}")
    REPORT['A_max_artifact'] = float(np.max(np.abs(art)))
    REPORT['A_max_nonlinearity'] = float(np.max(np.abs(v_app - v)))
    REPORT['A_max_alias'] = float(np.max(np.abs(alias)))
    REPORT['A_max_leak'] = float(np.max(np.abs(leak)))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    ax.plot(v / 1e3, v / 1e3, color=GRID, lw=3, label='identity  $v_{app}=v_{true}$')
    ax.plot(v / 1e3, v_app / 1e3, color=C[0], label='6-point HMI estimator')
    ax.set_xlabel('true velocity  [km/s]'); ax.set_ylabel('apparent velocity  [km/s]')
    _title(ax, 'A1  Quiet response is the identity', 'to the eye — the error lives below')
    ax.legend(loc='upper left')

    ax = axes[1]
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(v / 1e3, v_app - v, color=C[0], label='total ($v_{app}-v_{true}$)')
    ax.plot(v / 1e3, alias, color=C[1], label='6-point aliasing')
    ax.plot(v / 1e3, leak, color=C[2], label='finite-window leakage')
    ax.set_xlabel('true velocity  [km/s]'); ax.set_ylabel('residual  [m/s]')
    _title(ax, 'A2  The instrumental non-linearity',
           'this is what the look-up table is built to undo')
    ax.legend(loc='lower left', fontsize=9)

    ax = axes[2]
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(v / 1e3, (v_corr - v) * 1e9, color=C[0])
    ax.set_xlabel('true velocity  [km/s]')
    ax.set_ylabel('artifact after LUT  [nm/s]')
    _title(ax, 'A3  After the quiet look-up table',
           f'max |artifact| = {np.max(np.abs(art)):.1e} m/s  —  clean')
    _save(fig, 'expA_validation.png')

    # -- the look-up table itself and its local gain -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.plot(lut._x / 1e3, lut._y / 1e3, color=C[0])
    ax.plot(lut._x / 1e3, lut._x / 1e3, color=GRID, lw=2, zorder=0)
    ax.set_xlabel('$v_{apparent}$  [km/s]'); ax.set_ylabel('$v_{true}$  [km/s]')
    _title(ax, 'Quiet look-up table', 'grey = no correction')
    ax = axes[1]
    ax.plot(lut._x / 1e3, lut.derivative(lut._x), color=C[0])
    ax.axhline(1.0, color=GRID, lw=2, zorder=0)
    ax.set_xlabel('$v_{apparent}$  [km/s]')
    ax.set_ylabel('$dv_{true}/dv_{apparent}$')
    _title(ax, 'Local gain of the inversion',
           'departures from 1 amplify any profile error')
    REPORT['lut_gain_range'] = (float(lut.derivative(lut._x).min()),
                                float(lut.derivative(lut._x).max()))
    print(f"  LUT local gain dv_true/dv_app ranges over "
          f"[{REPORT['lut_gain_range'][0]:.3f}, {REPORT['lut_gain_range'][1]:.3f}]")
    _save(fig, 'lookup_table.png')


# ===========================================================================
# Experiment B -- artifact vs Wilson depression
# ===========================================================================
def experiment_B(lut):
    print("\n" + "=" * 78)
    print("EXPERIMENT B -- artifact vs Wilson depression (B = 2500 G)")
    print("=" * 78)

    z = np.linspace(0.0, 800.0, 161)

    # B.0 -- the literal spec configuration: obs_vr = 0
    art0 = np.array([measure_artifact(zi, cfg.B_GAUSS, 0.0, 0.0, lookup=lut)[0]
                     for zi in z])
    print(f"  At obs_vr = 0 (the literal spec setting): max |artifact| = "
          f"{np.max(np.abs(art0)):.2e} m/s  -> IDENTICALLY ZERO")
    print("    The deformation is symmetric in wavelength and the 6 filters are")
    print("    symmetric about lambda0, so LCP and RCP responses are exact")
    print("    mirror images and cancel in V = (v_L + v_R)/2.  The artifact")
    print("    CANNOT exist without a non-zero total shift: it is switched on by")
    print("    OBS_VR itself.  This is a prediction, not a bug.")
    REPORT['B_zero_obsvr'] = float(np.max(np.abs(art0)))

    # B.1 -- at several OBS_VR operating points
    obs_levels = [0.0, 1000.0, 2000.0, 3000.0]
    curves = {}
    for ov in obs_levels:
        curves[ov] = np.array([measure_artifact(zi, cfg.B_GAUSS, 0.0, ov,
                                                lookup=lut)[0] for zi in z])

    ref = curves[3000.0]
    slope = np.polyfit(z, ref, 1)[0]
    print(f"\n  At OBS_VR = 3000 m/s (orbital maximum):")
    for zi in (0, 200, 400, 600, 800):
        k = int(np.argmin(np.abs(z - zi)))
        print(f"    z_W = {zi:3d} km -> artifact = {ref[k]:8.2f} m/s")
    print(f"    mean slope over 0-800 km = {slope:.3f} (m/s) per km")
    print(f"    monotonic increasing?      {bool(np.all(np.diff(ref) > 0))}")
    REPORT['B_slope'] = float(slope)
    REPORT['B_at_600'] = float(ref[int(np.argmin(np.abs(z - 600)))])
    REPORT['B_monotonic'] = bool(np.all(np.diff(ref) > 0))

    # B.2 -- decomposition: Zeeman only / Wilson only / both
    z_only = np.array([measure_artifact(zi, 0.0, 0.0, 3000.0, lookup=lut)[0]
                       for zi in z])                       # Wilson only
    b_only = measure_artifact(0.0, cfg.B_GAUSS, 0.0, 3000.0, lookup=lut)[0]

    # B.3 -- the physical (real) velocity content of the same profiles
    pL = make_umbral_profile(600.0, cfg.B_GAUSS, 'LCP')
    pR = make_umbral_profile(600.0, cfg.B_GAUSS, 'RCP')
    v_phys = 0.5 * (cog_velocity(pL, 3000.0) + cog_velocity(pR, 3000.0)) - 3000.0
    print(f"\n  AMPLIFICATION at z_W = 600 km, OBS_VR = 3000 m/s:")
    print(f"    real (centre-of-gravity) velocity bias of the deformed profile"
          f" = {float(v_phys):.2e} m/s")
    print(f"    artifact reported by the HMI pipeline                        "
          f" = {REPORT['B_at_600']:.2f} m/s")
    print(f"    -> the deformation carries NO real velocity; the artifact is")
    print(f"       100 % instrumental (amplification of a null signal).")
    REPORT['B_phys_bias'] = float(abs(v_phys))
    REPORT['B_wilson_only'] = float(z_only[int(np.argmin(np.abs(z - 600)))])
    REPORT['B_zeeman_only'] = float(b_only)
    print(f"    Zeeman alone (z_W = 0)          : {float(b_only):8.2f} m/s")
    print(f"    Wilson alone (B = 0, z_W = 600) : {REPORT['B_wilson_only']:8.2f} m/s")
    print(f"    both together                   : {REPORT['B_at_600']:8.2f} m/s")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.axhline(0, color=GRID, lw=1)
    for k, ov in enumerate(obs_levels):
        ax.plot(z, curves[ov], color=SEQ[k + 1],
                label=f'OBS_VR = {ov:.0f} m/s')
        if ov > 0:
            ax.annotate(f'{ov:.0f}', xy=(z[-1], curves[ov][-1]),
                        xytext=(6, 0), textcoords='offset points',
                        color=SEQ[k + 1], fontsize=9, va='center')
    ax.set_xlabel('Wilson depression  $z_W$  [km]')
    ax.set_ylabel('artifact velocity  [m/s]')
    _title(ax, 'B1  Artifact grows with Wilson depression',
           'and vanishes identically when OBS_VR = 0')
    ax.legend(loc='upper left', fontsize=9)

    ax = axes[1]
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(z, curves[3000.0], color=C[0], label='Wilson + Zeeman')
    ax.plot(z, z_only, color=C[1], label='Wilson only (B = 0)')
    ax.axhline(float(b_only), color=C[2], ls='--',
               label=f'Zeeman only ($z_W$=0): {float(b_only):.0f} m/s')
    ax.axhline(0, color=C[3], lw=2.5, ls=':',
               label='real velocity bias (centre of gravity)')
    ax.set_xlabel('Wilson depression  $z_W$  [km]')
    ax.set_ylabel('artifact velocity  [m/s]')
    _title(ax, 'B2  Where the artifact comes from',
           'OBS_VR = 3000 m/s, B = 2500 G')
    ax.legend(loc='upper left', fontsize=8.5)
    _save(fig, 'expB_wilson_sweep.png')

    # B.4 -- 2-D map over (z_W, OBS_VR)
    zz = np.linspace(0, 800, 81)
    oo = np.linspace(-4000, 4000, 161)
    M = np.zeros((zz.size, oo.size))
    for i, zi in enumerate(zz):
        M[i] = measure_artifact(zi, cfg.B_GAUSS, 0.0, oo, lookup=lut)[0]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    lim = np.max(np.abs(M))
    im = ax.pcolormesh(oo, zz, M, cmap=DIVERGING, vmin=-lim, vmax=lim,
                       shading='auto')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax); cb.set_label('artifact velocity  [m/s]', color=INK2)
    ax.set_xlabel('OBS_VR  [m/s]'); ax.set_ylabel('Wilson depression  $z_W$  [km]')
    _title(ax, 'B3  The artifact is odd in OBS_VR',
           'zero along the whole OBS_VR = 0 axis, for every $z_W$')
    _save(fig, 'expB_map.png')
    REPORT['B_map_max'] = float(lim)


# ===========================================================================
# Experiment C -- the 24 h signature
# ===========================================================================
def experiment_C(lut, lut_ext):
    print("\n" + "=" * 78)
    print("EXPERIMENT C -- 24 h temporal signature via OBS_VR")
    print("=" * 78)

    n = 2048
    t = np.linspace(0.0, 48.0, n, endpoint=False)
    obs = cfg.A_ORB * np.sin(2 * np.pi * t / cfg.P_ORB_HOURS)

    art, vdop, blos = measure_artifact(cfg.Z_W_REF, cfg.B_GAUSS, 0.0, obs,
                                       lookup=lut)
    art_e, _, blos_e = measure_artifact(cfg.Z_W_REF, cfg.B_GAUSS, 0.0, obs,
                                        lookup=lut_ext)

    def harmonics(y):
        F = np.fft.rfft(y - y.mean()) * 2.0 / y.size
        per = np.full(F.size, np.inf)
        per[1:] = 48.0 / np.arange(1, F.size)
        return per, np.abs(F)

    per, amp = harmonics(art)
    per_e, amp_e = harmonics(art_e)
    k24 = int(np.argmin(np.abs(per - 24.0)))
    kmax = int(np.argmax(amp))

    print(f"  Configuration: z_W = {cfg.Z_W_REF:.0f} km, B = {cfg.B_GAUSS:.0f} G,"
          f" A_orb = {cfg.A_ORB:.0f} m/s")
    print(f"\n  Spec look-up table (+-7000 m/s grid):")
    print(f"    amplitude (half peak-to-peak) = {0.5*(art.max()-art.min()):7.1f} m/s")
    print(f"    peak-to-peak                  = {art.max()-art.min():7.1f} m/s")
    print(f"    24 h harmonic amplitude       = {amp[k24]:7.1f} m/s")
    print(f"    dominant period               = {per[kmax]:7.2f} h")
    print(f"    LUT extrapolations            = {lut.n_extrapolated}/{lut.n_calls}"
          f"  ({100*lut.n_extrapolated/max(lut.n_calls,1):.0f} %)")
    print(f"\n  Extended look-up table (+-9500 m/s grid, no extrapolation):")
    print(f"    amplitude                     = {0.5*(art_e.max()-art_e.min()):7.1f} m/s")
    print(f"    peak-to-peak                  = {art_e.max()-art_e.min():7.1f} m/s")
    print(f"    24 h harmonic amplitude       = {amp_e[k24]:7.1f} m/s")
    print(f"\n  Observed residual to explain    = {cfg.V_OBSERVED_RESIDUAL:7.1f} m/s")
    print(f"  B_los contamination: {blos.min():.1f} .. {blos.max():.1f} G "
          f"(peak-to-peak {blos.max()-blos.min():.1f} G) around a true 2500 G")
    per_b, amp_b = harmonics(blos)
    kb = int(np.argmax(amp_b))
    print(f"  B_los dominant period           = {per_b[kb]:7.2f} h "
          f"(amplitude {amp_b[kb]:.1f} G)")
    print("    -> V is ODD in OBS_VR (24 h) but B_LOS is EVEN in OBS_VR (12 h):")
    print("       a falsifiable prediction that separates this mechanism from a")
    print("       real solar signal.")
    print("  Harmonic content of the 24 h signal (spec LUT):")
    for kk in range(1, 9):
        idx = int(round(48.0 / (24.0 / kk)))
        if idx < amp.size:
            print(f"    period {24.0/kk:5.2f} h : {amp[idx]:7.2f} m/s")
    REPORT['C_amp'] = float(0.5 * (art.max() - art.min()))
    REPORT['C_amp_ext'] = float(0.5 * (art_e.max() - art_e.min()))
    REPORT['C_p2p'] = float(art.max() - art.min())
    REPORT['C_p2p_ext'] = float(art_e.max() - art_e.min())
    REPORT['C_24h'] = float(amp[k24])
    REPORT['C_24h_ext'] = float(amp_e[k24])
    REPORT['C_dominant'] = float(per[kmax])
    REPORT['C_blos_p2p'] = float(blos.max() - blos.min())
    REPORT['C_blos_period'] = float(per_b[kb])
    REPORT['C_extrap_frac'] = float(lut.n_extrapolated / max(lut.n_calls, 1))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.6), sharex=False)

    ax = axes[0]
    for sgn in (+1, -1):
        ax.axhline(sgn * cfg.V_OBSERVED_RESIDUAL, color='#b9b5a7', lw=1.4,
                   ls=(0, (6, 4)), zorder=0)
    ax.text(48.4, cfg.V_OBSERVED_RESIDUAL, ' observed\n ±350 m/s',
            color=INK2, fontsize=8.5, va='center')
    ax.set_ylim(-430, 430)
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(t, art, color=C[0], label='spec LUT ($\\pm$7000 m/s)')
    ax.plot(t, art_e, color=C[1], label='extended LUT ($\\pm$9500 m/s)')
    ax.set_xlabel('time  [h]'); ax.set_ylabel('artifact velocity  [m/s]')
    ax.set_xticks(np.arange(0, 49, 6))
    _title(ax, 'C1  Spurious velocity over the umbra',
           f'$z_W$ = {cfg.Z_W_REF:.0f} km, B = {cfg.B_GAUSS:.0f} G, '
           f'true plasma velocity = 0')
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    ax = axes[1]
    ax.axhline(cfg.B_GAUSS, color=GRID, lw=2)
    ax.text(48.3, cfg.B_GAUSS, ' true\n 2500 G', color=INK2, fontsize=8.5,
            va='center')
    ax.plot(t, blos, color=C[2])
    ax.set_xlabel('time  [h]'); ax.set_ylabel('$B_{LOS}$  [G]')
    ax.set_xticks(np.arange(0, 49, 6))
    _title(ax, 'C2  The magnetogram is contaminated too',
           f'peak-to-peak {blos.max()-blos.min():.0f} G — but at '
           f'{per_b[kb]:.0f} h, twice the velocity frequency')

    ax = axes[2]
    sel = (per > 2.0) & (per < 60.0)
    ax.vlines(per[sel], 0, amp[sel], color=C[0], lw=2.5)
    ax.set_xscale('log')
    ax.set_xticks([3, 4, 6, 8, 12, 24, 48])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(2.5, 60)
    ax.set_xlabel('period  [h]'); ax.set_ylabel('amplitude  [m/s]')
    _title(ax, 'C3  Harmonic content',
           f'fundamental at {per[kmax]:.0f} h, inherited directly from OBS_VR')
    ax.annotate(f'{amp[k24]:.0f} m/s @ 24 h', xy=(24, amp[k24]),
                xytext=(-10, 8), textcoords='offset points', color=C[0],
                fontsize=9, ha='right')
    _save(fig, 'expC_24h_signature.png')


# ===========================================================================
# Sensitivity -- what would it take to reach 350 m/s?
# ===========================================================================
def sensitivity(lut_maker):
    print("\n" + "=" * 78)
    print("SENSITIVITY -- dependence on B and on the line-width narrowing k_s")
    print("=" * 78)

    lut = lut_maker()
    t = np.linspace(0.0, 48.0, 512, endpoint=False)
    obs = cfg.A_ORB * np.sin(2 * np.pi * t / cfg.P_ORB_HOURS)

    def amp_of(z_W, B, k_s):
        a, _, _ = measure_artifact(z_W, B, 0.0, obs, lookup=lut, k_s=k_s)
        return 0.5 * (a.max() - a.min())

    Bs = np.linspace(0.0, 4000.0, 161)
    ampB = np.array([amp_of(cfg.Z_W_REF, b, cfg.K_S) for b in Bs])

    narrow = np.linspace(0.0, 0.40, 81)            # fractional sigma reduction at 600 km
    ampS = np.array([amp_of(cfg.Z_W_REF, cfg.B_GAUSS, f / cfg.Z_W_REF)
                     for f in narrow])

    # Above this field the Zeeman-split polarisation leaves the unambiguous
    # +-10023 m/s range once the orbital shift is added, and the Fourier phase
    # wraps: the pipeline saturates instead of merely being biased.
    from .profiles import zeeman_shift_velocity as _zv
    B_wrap = (cfg.V_UNAMBIGUOUS - cfg.A_ORB) / (_zv(1.0))
    # keep a 10 % margin: near the threshold only the extreme orbital phases
    # wrap, so the curve is already transitional there.
    safe = Bs < 0.9 * B_wrap
    kbest = int(np.argmax(np.where(safe, ampB, -np.inf)))
    print(f"  Phase-wrap threshold: B = {B_wrap:.0f} G (above it "
          f"|OBS_VR| + v_Zeeman > {cfg.V_UNAMBIGUOUS:.0f} m/s and the")
    print("    first-harmonic phase wraps -- the pipeline saturates, and the")
    print("    steep rise of the curve beyond that point is not a bias but a")
    print("    breakdown of the 6-point algorithm.")
    print(f"  Well below the threshold (B < {0.9*B_wrap:.0f} G) the amplitude is NOT "
          f"monotone in B:")
    print(f"    it is structured, peaking at B = {Bs[kbest]:.0f} G "
          f"({ampB[kbest]:.0f} m/s), because the Zeeman splitting")
    print("    slides the two polarisations along a non-linearity that is "
          "periodic in velocity.")
    hit = narrow[np.argmax(ampS >= cfg.V_OBSERVED_RESIDUAL)] if np.any(
        ampS >= cfg.V_OBSERVED_RESIDUAL) else None
    print(f"  Default line narrowing at z_W = 600 km: "
          f"{cfg.K_S*cfg.Z_W_REF*100:.0f} %  -> {amp_of(cfg.Z_W_REF, cfg.B_GAUSS, cfg.K_S):.0f} m/s")
    if hit is not None:
        print(f"  Narrowing needed to reach {cfg.V_OBSERVED_RESIDUAL:.0f} m/s: "
              f"{hit*100:.0f} %")
    REPORT['S_B_peak'] = (float(Bs[kbest]), float(ampB[kbest]))
    REPORT['S_B_wrap'] = float(B_wrap)
    REPORT['S_narrow_for_350'] = None if hit is None else float(hit)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.axvspan(B_wrap, Bs[-1], color='#eae7dd', zorder=0)
    ax.text(B_wrap + 40, ampB.max() * 0.86, 'phase wrap:\nalgorithm saturates',
            color=INK2, fontsize=8.5)
    ax.axhline(cfg.V_OBSERVED_RESIDUAL, color=GRID, lw=2)
    ax.text(4050, cfg.V_OBSERVED_RESIDUAL, ' observed', color=INK2, fontsize=8.5,
            va='center')
    ax.plot(Bs, ampB, color=C[0])
    ax.axvline(cfg.B_GAUSS, color=C[1], ls='--', lw=1.5)
    ax.text(cfg.B_GAUSS, ax.get_ylim()[1] * 0.05, ' default 2500 G',
            color=C[1], fontsize=8.5)
    ax.set_xlabel('longitudinal field  B  [G]')
    ax.set_ylabel('24 h artifact amplitude  [m/s]')
    _title(ax, 'S1  Artifact vs magnetic field',
           'structured below the wrap threshold, unbounded above it')

    ax = axes[1]
    ax.axhline(cfg.V_OBSERVED_RESIDUAL, color=GRID, lw=2)
    ax.text(0.405, cfg.V_OBSERVED_RESIDUAL, ' observed', color=INK2,
            fontsize=8.5, va='center')
    ax.plot(narrow * 100, ampS, color=C[0])
    ax.axvline(cfg.K_S * cfg.Z_W_REF * 100, color=C[1], ls='--', lw=1.5)
    ax.text(cfg.K_S * cfg.Z_W_REF * 100, ax.get_ylim()[1] * 0.05,
            ' default', color=C[1], fontsize=8.5)
    ax.set_xlabel('line narrowing at $z_W$ = 600 km  [%]')
    ax.set_ylabel('24 h artifact amplitude  [m/s]')
    _title(ax, 'S2  Artifact vs line-width narrowing',
           'the only Wilson parameter the estimator can feel')
    _save(fig, 'sensitivity.png')


# ===========================================================================
def profiles_figure():
    lam = cfg.LAMBDA0 + np.linspace(-0.35, 0.35, 800)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(lam - cfg.LAMBDA0, quiet_profile(lam), color=C[0], label='quiet Sun')
    pL = make_umbral_profile(cfg.Z_W_REF, cfg.B_GAUSS, 'LCP')
    pR = make_umbral_profile(cfg.Z_W_REF, cfg.B_GAUSS, 'RCP')
    ax.plot(lam - cfg.LAMBDA0, pL(lam), color=C[1], label='umbra LCP')
    ax.plot(lam - cfg.LAMBDA0, pR(lam), color=C[2], label='umbra RCP')
    for x in filter_wavelengths() - cfg.LAMBDA0:
        ax.axvline(x, color=GRID, lw=1.2, zorder=0)
    ax.text(0.02, 0.06, 'vertical lines: the 6 HMI filter positions',
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.set_xlabel('$\\lambda - \\lambda_0$  [Å]'); ax.set_ylabel('intensity')
    _title(ax, 'The profiles the pipeline actually sees',
           f'$z_W$ = {cfg.Z_W_REF:.0f} km, B = {cfg.B_GAUSS:.0f} G '
           f'($\\Delta\\lambda_B$ = {zeeman_shift(cfg.B_GAUSS)*1000:.0f} mÅ '
           f'= {zeeman_shift(cfg.B_GAUSS)/cfg.SIGMA_QUIET:.1f}$\\sigma$)')
    ax.legend(loc='lower right', fontsize=9)
    _save(fig, 'profiles.png')


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("=" * 78)
    print("HMI OBS_VR ARTIFACT SIMULATION")
    print("=" * 78)
    Ic, Id, sg = wilson_parameters(cfg.Z_W_REF)
    print(f"  quiet   (Ic, Id, sigma) = ({cfg.IC_QUIET:.3f}, {cfg.ID_QUIET:.3f}, "
          f"{cfg.SIGMA_QUIET:.4f} A)")
    print(f"  umbra   (Ic, Id, sigma) = ({Ic:.3f}, {Id:.3f}, {sg:.4f} A)"
          f"   at z_W = {cfg.Z_W_REF:.0f} km")
    print(f"  Zeeman  d(lambda)_B     = {zeeman_shift(cfg.B_GAUSS)*1000:.2f} mA"
          f"  = {zeeman_shift_velocity(cfg.B_GAUSS):.0f} m/s equivalent"
          f"  at B = {cfg.B_GAUSS:.0f} G")
    print(f"  unambiguous velocity range = +-{cfg.V_UNAMBIGUOUS:.0f} m/s")

    profiles_figure()
    lut = LookupTable()
    experiment_A(lut)
    lut_b = LookupTable()
    experiment_B(lut_b)
    lut_c = LookupTable()
    lut_ext = LookupTable(v_min=cfg.LUT_V_MIN_EXT, v_max=cfg.LUT_V_MAX_EXT,
                          n=cfg.LUT_N_EXT)
    experiment_C(lut_c, lut_ext)
    sensitivity(LookupTable)
    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    return REPORT


if __name__ == '__main__':
    main()
