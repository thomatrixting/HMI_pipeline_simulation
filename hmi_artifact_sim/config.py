"""
Configuration for the HMI OBS_VR artifact simulation.

Every physical constant, instrumental constant and *phenomenological* parameter
lives here.  Nothing below is hardcoded inside the functions of the other
modules (acceptance criterion 8.4).

References
----------
Couvidat et al. (2011), "Calculation of Observables from HMI Data",
Solar Physics 275, 285.  Equation numbers below refer to that paper.
"""

# ---------------------------------------------------------------------------
# 1. Physical / instrumental constants (Couvidat 2011, §1 of the spec)
# ---------------------------------------------------------------------------
C_LIGHT = 299792458.0          # m/s, speed of light
LAMBDA0 = 6173.3433            # Angstrom, rest centre of the Fe I line
DVDL = C_LIGHT / LAMBDA0       # m/s per Angstrom  -> 48562.4...
N_FILTERS = 6                  # HMI samples the line at 6 positions
DLAMBDA = 0.0688               # Angstrom, filter separation (68.8 mA)
T_PERIOD = N_FILTERS * DLAMBDA # Angstrom = 0.4128 A, "period" of the profile
LANDE_G = 2.5                  # effective Lande factor of Fe I 6173

# Standard Zeeman constant: d(lambda)_B = ZEEMAN_C * g * lambda0[A]^2 * B[G]
ZEEMAN_C = 4.67e-13

# Couvidat Eq. 15 magnetic calibration constant, literal expression.
# Units: Gauss per (m/s of LCP-RCP velocity difference).
KM = 1.0 / (2.0 * 4.67e-5 * 0.000061733433 * 2.5 * 299792458.0)

# Unambiguous velocity range of the 6-point algorithm: one full phase turn of
# the first Fourier harmonic corresponds to T_PERIOD in wavelength.
V_UNAMBIGUOUS = DVDL * T_PERIOD / 2.0        # +-10023 m/s

# ---------------------------------------------------------------------------
# 2. Quiet-Sun Fe I 6173 profile (Couvidat Eq. 5)
#       I(lam) = Ic - Id * exp(-(lam - lam0)^2 / sigma^2)
# ---------------------------------------------------------------------------
IC_QUIET = 1.0       # normalised continuum
ID_QUIET = 0.55      # line depth (~55 % of the continuum)
SIGMA_QUIET = 0.060  # Angstrom; FWHM = 2*sqrt(ln2)*sigma ~ 0.10 A

# ---------------------------------------------------------------------------
# 3. Wilson depression -> phenomenological deformation of (Ic, Id, sigma)
#
#     Ic_umbra    = Ic_quiet    * (1 - K_C * z_W)
#     Id_umbra    = Id_quiet    * (1 - K_D * z_W)
#     sigma_umbra = sigma_quiet * (1 - K_S * z_W)
#
# Calibrated so that at the reference depth Z_W_REF = 600 km the continuum
# drops ~25 %, the line depth ~15 % and the width ~10 %.
# THESE ARE PHENOMENOLOGICAL AND ARE THE MAIN TUNING KNOBS OF THE MODEL.
# ---------------------------------------------------------------------------
Z_W_REF = 600.0              # km, reference Wilson depression
K_C = 0.25 / Z_W_REF         # per km, continuum darkening
K_D = 0.15 / Z_W_REF         # per km, line-depth reduction
K_S = 0.10 / Z_W_REF         # per km, line-width narrowing

# ---------------------------------------------------------------------------
# 4. Magnetic field and orbit
# ---------------------------------------------------------------------------
B_GAUSS = 2500.0             # G, longitudinal umbral field
A_ORB = 3000.0               # m/s, amplitude of the OBS_VR orbital modulation
P_ORB_HOURS = 24.0           # h, SDO geosynchronous orbital period

# ---------------------------------------------------------------------------
# 5. Look-up table grid (Couvidat uses 821 points)
# ---------------------------------------------------------------------------
LUT_V_MIN = -7000.0
LUT_V_MAX = 7000.0
LUT_N = 821
# Extended grid used only for the sensitivity check of §9 (extrapolation).
LUT_V_MIN_EXT = -9500.0
LUT_V_MAX_EXT = 9500.0
LUT_N_EXT = 1901

# Number of samples used by the "ideal instrument" reference estimator: the
# same Fourier phase estimator but densely sampled, i.e. free of the 6-point
# aliasing.  This is the benchmark against which the artifact is measured.
N_IDEAL = 1024

# Physical-truth (centre-of-gravity) reference estimator: window half width in
# Angstrom and number of integration nodes.  Wide enough to contain the whole
# Zeeman-split profile plus a 3000 m/s orbital shift.
COG_HALF_WIDTH = 1.2
COG_N = 8001

# Observed residual we are trying to explain, for reference in the plots.
V_OBSERVED_RESIDUAL = 350.0  # m/s
