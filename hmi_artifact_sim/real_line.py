"""
La linea Fe I 6173 REAL que usa el pipeline, quieta y deformada.

ATENCION, dato importante que solo se ve leyendo el fuente: aunque el repo del
JSOC trae `ReferenceFeLine.bin` (la linea de R. Ulrich), HMI_lookup.c
**tiene comentada su lectura** (lineas 236-240) y en su lugar construye la
linea de Sol quieto analiticamente (lineas 279-357): una pseudo-Voigt mas dos
gaussianas de ala que simulan la asimetria, con parametros por calibracion.

    templineref(lam) = Icg
                     - fdepthg * exp(-l^2) * [ 1 - a*(2/sqrt(pi))*(1/(2 l^2))
                       * ( (4l^2+3)(l^2+1) exp(-l^2) - (1/l^2)(2l^2+3) sinh(l^2) ) ]
                     + A1 * exp(-(lam - c1)^2 / w1^2)
                     + A2 * exp(-(lam - c2)^2 / w2^2)

con l = lam/fwidthg, y para |l| > 26.5 solo el continuo mas las dos gaussianas
(el sinh desborda).  Los parametros, por calibracion:

  cal 11 (cal=0, may 2010 - 18 ene 2012): Ic 1.0, fdepth 0.5625, fwidth 0.06415,
          a=+0.03, blends -0.015 @ -0.225/0.2  y +0.004 @ +0.150/0.22
  cal 12 (cal=1, desde 18 ene 2012):      Ic 1.0, fdepth 0.53,   fwidth 0.0615,
          a=+0.03, blends +0.010 @ -0.225/0.2  y +0.015 @ +0.100/0.25
  cal 13 (cal=2, desde 15 ene 2014):      Ic 1.0, fdepth 0.58,   fwidth 0.058,
          a=-0.09, blends +0.0074 @ -0.200/0.13 y +0.021 @ +0.050/0.18

Que calibracion uso cada LUT esta en su keyword COMMENT.  (La linea del .bin se
deja accesible en `reference_bin_line()` para comparar: su minimo cae en
-62 mA, no en el centro, y de ahi venia un sesgo de ~2.4 km/s cuando se la usa
por error.)

Encima de eso va el escalado centro-limbo de HMI_lookup.c:982-996,

    Id(mu)   = 0.41922611 + 0.24190794 * mu
    FWHM(mu) = 151.34559  - 58.521771  * mu   [mA]      mu = cos(theta)

que el pipeline aplica como RAZON respecto a mu = 1, de modo que a centro de
disco la Voigt se usa tal cual.  Por eso la profundidad propia de la Voigt
(0.640 en cal 13) no coincide con Id(1) = 0.661: estos coeficientes salen de
ajustar la linea de Ulrich y aqui solo fijan la variacion centro-limbo.

La deformacion umbral conserva el significado de profiles.py.  Si
L_q(lam) = Ic0 - D(lam),

    L_u(lam) = Ic0 * f_c  -  f_d * D((lam - centro) / f_s)

con (f_c, f_d, f_s) = (1-k_c z_W, 1-k_d z_W, 1-k_s z_W) y centro = +-Delta lam_B.
"""

import numpy as np

from . import config as cfg
from . import jsoc_data as jd
from .profiles import zeeman_shift

# HMI_lookup.c:279-295 y 307-308, 331-357.  Clave = etiqueta del COMMENT.
CALIBRATIONS = {
    11: dict(cal=0, Ic=1.0, fdepth=0.5625, fwidth=0.06415, a=0.03,
             blends=((-0.015, -0.225, 0.2), (+0.004, +0.150, 0.22))),
    12: dict(cal=1, Ic=1.0, fdepth=0.53, fwidth=0.0615, a=0.03,
             blends=((+0.010, -0.225, 0.2), (+0.015, +0.100, 0.25))),
    13: dict(cal=2, Ic=1.0, fdepth=0.58, fwidth=0.058, a=-0.09,
             blends=((+0.0074, -0.200, 0.13), (+0.021, +0.050, 0.18))),
}

VOIGT_L_MAX = 26.5   # mas alla de esto el sinh(l^2) desborda en C


def calibration_of(comment):
    """Extrae 11, 12 o 13 del keyword COMMENT de un registro de LUT/phasemap."""
    m = [c for c in (11, 12, 13) if str(comment).rstrip().endswith(str(c))]
    if len(m) != 1:
        raise ValueError(f'no puedo leer la calibracion de {comment!r}')
    return m[0]


def voigt_line(lam, calibration=13):
    """La pseudo-Voigt + 2 gaussianas de HMI_lookup.c, con continuo 1."""
    k = CALIBRATIONS[calibration]
    lam = np.asarray(lam, float)
    l = lam / k['fwidth']
    l2 = np.where(l == 0.0, np.finfo(float).tiny, l ** 2)

    core = np.full(lam.shape, k['Ic'], dtype=float)
    ok = np.abs(l) <= VOIGT_L_MAX
    e = np.exp(-l2[ok])
    bracket = 1.0 - k['a'] * 2.0 / np.sqrt(np.pi) * (1.0 / (2.0 * l2[ok])) * (
        (4.0 * l2[ok] + 3.0) * (l2[ok] + 1.0) * e
        - (1.0 / l2[ok]) * (2.0 * l2[ok] + 3.0) * np.sinh(l2[ok]))
    core[ok] = k['Ic'] - k['fdepth'] * e * bracket

    for A, c, w in k['blends']:
        core = core + A * np.exp(-((lam - c) ** 2) / w ** 2)
    return core


def reference_grid():
    """La rejilla de 7000 puntos de HMI_lookup.c:248: k/(n-1) - 0.5, sin cero."""
    n = jd.parse_hmiparam()['referencenlam']
    return np.arange(n, dtype=float) / (n - 1.0) - 0.5


def reference_bin_line():
    """
    (perfil, rejilla) de ReferenceFeLine.bin: la linea de Ulrich tal cual.
    NO es la que usa el pipeline (su lectura esta comentada); se deja para
    comparar.  Su minimo cae en lam = -62 mA.
    """
    return jd.load_reference_line()


class RealLine:
    """
    La linea del pipeline a un mu y una calibracion dados.

    lam_ref : rejilla propia, A relativos al centro de la linea
    prof    : intensidad, continuo ~1
    Ic, Id, fwhm : continuo, profundidad y FWHM nominal (mA) a ese mu
    """

    def __init__(self, mu=1.0, calibration=13):
        p = jd.parse_hmiparam()
        mc, fc = p['minimumCoeffs'], p['FWHMCoeffs']
        wl = reference_grid()
        templineref = voigt_line(wl, calibration)

        self.mu = float(mu)
        self.calibration = calibration
        self.Id = mc[0] + mc[1] * self.mu
        self.fwhm = fc[0] + fc[1] * self.mu

        # HMI_lookup.c:986-990, literal
        self.prof = 1.0 - (1.0 - templineref) * self.Id / mc.sum()
        self.lam_ref = wl * self.fwhm / fc.sum()
        self.Ic = float(self.prof.max())

    def __call__(self, lam):
        return np.interp(lam, self.lam_ref, self.prof,
                         left=self.prof[0], right=self.prof[-1])

    def depth_profile(self, x):
        """D(x) = Ic - L_q(x): la absorcion, con pico ~Id."""
        return self.Ic - self(x)

    def centre(self):
        """Posicion del minimo del perfil, en mA (debe ser ~0)."""
        return 1e3 * self.lam_ref[int(np.argmin(self.prof))]

    def measured_fwhm(self):
        """FWHM del perfil completo, en mA."""
        d = self.Ic - self.prof
        half = d.max() / 2.0
        k = int(np.argmax(d))
        lo, hi = k, k
        while lo > 0 and d[lo] > half:
            lo -= 1
        while hi < d.size - 1 and d[hi] > half:
            hi += 1
        return 1e3 * (self.lam_ref[hi] - self.lam_ref[lo])


_CACHE = {}


def quiet_line(mu=1.0, calibration=13):
    """La linea real quieta (cacheada)."""
    key = (round(float(mu), 6), calibration)
    if key not in _CACHE:
        _CACHE[key] = RealLine(*key)
    return _CACHE[key]


def wilson_factors(z_W, k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S):
    """(f_c, f_d, f_s): factores de continuo, profundidad y ancho."""
    return 1.0 - k_c * z_W, 1.0 - k_d * z_W, 1.0 - k_s * z_W


def umbral_line(z_W=0.0, B_gauss=0.0, polarization='LCP', mu=1.0,
                calibration=13, k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S):
    """
    Callable lam -> L(lam): la linea real deformada por Wilson y desplazada
    rigidamente por Zeeman.  Con z_W = 0 y B = 0 es exactamente quiet_line().
    """
    base = quiet_line(mu, calibration)
    f_c, f_d, f_s = wilson_factors(z_W, k_c, k_d, k_s)
    dlB = zeeman_shift(B_gauss)
    pol = polarization.upper()
    centre = {'LCP': +dlB, 'RCP': -dlB, 'I': 0.0, 'NONE': 0.0}.get(pol)
    if centre is None:
        raise ValueError("polarization debe ser 'LCP', 'RCP' o 'I'")

    def _line(lam):
        x = (np.asarray(lam, float) - centre) / f_s
        return base.Ic * f_c - f_d * base.depth_profile(x)

    return _line
