"""
El pipeline con los perfiles reales: integracion, Fourier e inversion.

Tres diferencias con `pipeline.py`, que es la version sintetica:

1. MUESTREO.  Ya no es un delta, `I_j = L(lam_j)`, sino la integral real

       I_j = sum_k T_j(lam_k) * L(lam_k - v*dlamdv)      (HMI_lookup.c:1117)

   sobre la rejilla de 32001 puntos del pipeline, con el mismo recorte de
   bordes (`maxshiftlam`) que hace el fuente oficial.

2. ESCALA DE VELOCIDAD.  HMI_lookup.c:541 usa

       dtune  = FSR_NB/2.5 = 0.06756 A      (no 68.8 mA)
       dvtune = dtune/dlamdv = 3281 m/s
       pv1    = dvtune*(N-1)                <- con el comentario literal del
       pv2    = pv1/2                          fuente: "WARNING THIS IS WRONG,
                                                SHOULD BE dvtune*N"

   es decir +-8.2 km/s de rango no ambiguo en el 1er Fourier, 5/6 de los
   +-10.0 km/s que supone la version sintetica.  Ese 5/6 entra directo en la
   pendiente de la inversion.

3. INVERSION.  Dos tablas: la que reconstruimos con los filtros reales
   (`ReconstructedLUT`) y la real descargada del JSOC (`JSOCLookupTable`),
   invertida como en Dopplergram.c:482-585 (busqueda ascendente, interpolacion
   lineal, saturacion recortada a vtest[0] / vtest[-1] y contada como SATVALS).

El signo de la fase, `atan2(-f1s, -f1c)` (HMI_lookup.c:1137), es exactamente el
que ya usaba `pipeline.velocity_from_coefficients`.
"""

import numpy as np

from . import jsoc_data as jd
from . import real_filters as rfl
from .real_line import quiet_line

N_FILTERS = rfl.N_FILTERS


# ---------------------------------------------------------------------------
# Escala de velocidad real
# ---------------------------------------------------------------------------
def velocity_scale(n=N_FILTERS):
    """(dtune [A], dvtune [m/s], pv1 [m/s], pv2 [m/s]) de HMI_lookup.c:414,541."""
    p = jd.parse_hmiparam()
    dtune = p['FSR'][0] / 2.5
    dvtune = dtune / p['dlamdv']
    pv1 = dvtune * (n - 1)          # el (N-1) del fuente, no N
    return dtune, dvtune, pv1, pv1 / 2.0


def _fourier_weights(n=N_FILTERS):
    a = rfl.TUNE_ANGLE[:n] * 2.0 * np.pi / n
    return np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)


def velocities_from_intensities(I, n=N_FILTERS):
    """
    (vel1, vel2) en m/s a partir de I[..., 6].

    vel2 se desenvuelve contra vel1 como en HMI_lookup.c:1140.
    """
    I = np.asarray(I, float)
    c1, s1, c2, s2 = _fourier_weights(n)
    _, _, pv1, pv2 = velocity_scale(n)
    f1c = I @ c1
    f1s = I @ s1
    f2c = I @ c2
    f2s = I @ s2
    vel1 = np.arctan2(-f1s, -f1c) * pv1 / (2.0 * np.pi)
    v2 = np.arctan2(-f2s, -f2c) * pv2 / (2.0 * np.pi)
    vel2 = np.mod(v2 - vel1 + 10.5 * pv2, pv2) - pv2 / 2.0 + vel1
    return vel1, vel2


# ---------------------------------------------------------------------------
# Integracion de la linea por los filtros reales
# ---------------------------------------------------------------------------
def edge_clip(filterset):
    """
    (lam, T) recortados como HMI_lookup.c:1117: se descartan `maxshiftlam`
    muestras en cada borde, donde maxshiftlam = round(vtest_max*dlamdv/dlam).
    """
    p = jd.parse_hmiparam()
    dlam = filterset.dlam
    m = int(round(jd.vtest_grid()[-1] * p['dlamdv'] / dlam))
    sl = slice(m, filterset.lam.size - m)
    return filterset.lam[sl], filterset.T[:, sl]


def intensities(filterset, line_func, v, chunk=128, clip=True):
    """
    I[..., 6] para una o muchas velocidades de desplazamiento `v` [m/s].

    Positivo = corrimiento al rojo (alejandose), la convencion de
    HMI_lookup.c:1113.
    """
    p = jd.parse_hmiparam()
    lam, T = edge_clip(filterset) if clip else (filterset.lam, filterset.T)
    Tt = np.ascontiguousarray(T.T)                      # (nlam, 6)
    v = np.atleast_1d(np.asarray(v, float))
    out = np.empty(v.shape + (T.shape[0],))
    flat = out.reshape(-1, T.shape[0])
    vf = v.reshape(-1)
    for a in range(0, vf.size, chunk):
        b = min(a + chunk, vf.size)
        L = line_func(lam[None, :] - (vf[a:b, None] * p['dlamdv']))
        flat[a:b] = L @ Tt
    return out if np.ndim(v) else out[0]


def apparent_velocity(filterset, line_func, v, **kw):
    """La velocidad aparente del 1er Fourier: la cadena forward completa."""
    return velocities_from_intensities(intensities(filterset, line_func, v, **kw))[0]


# ---------------------------------------------------------------------------
# Inversion: la interfaz comun de las dos tablas
# ---------------------------------------------------------------------------
class _TableBase:
    """
    Una curva v_apparent(vtest) que se invierte como en Dopplergram.c.

    `self.vtest` es la rejilla de 821 velocidades verdaderas y `self.vapp` la
    velocidad aparente que el algoritmo de Fourier devuelve en cada una.
    """

    vtest = None
    vapp = None
    label = '?'

    def _finalize(self):
        self.vmin = float(self.vapp[0])
        self.vmax = float(self.vapp[-1])
        self.n_calls = 0
        self.n_saturated = 0

    def __call__(self, v_apparent):
        """
        v_apparent -> v_true, con recorte y conteo de saturacion.

        Busqueda e interpolacion lineal entre los dos nodos que lo encierran;
        si cae fuera de [vmin, vmax] se recorta al extremo y se cuenta
        (equivalente a SATVALS de Dopplergram.c).
        """
        x = np.atleast_1d(np.asarray(v_apparent, float))
        sat = (x < self.vmin) | (x > self.vmax)
        self.n_calls += x.size
        self.n_saturated += int(np.count_nonzero(sat))
        xc = np.clip(x, self.vmin, self.vmax)
        out = np.interp(xc, self.vapp, self.vtest)
        out = np.where(x < self.vmin, self.vtest[0], out)
        out = np.where(x > self.vmax, self.vtest[-1], out)
        return out if np.ndim(v_apparent) else float(out[0])

    def gain(self, v_apparent=None):
        """dv_true/dv_apparent: el factor de amplificacion local de la tabla."""
        x = self.vapp if v_apparent is None else np.asarray(v_apparent, float)
        h = 1.0
        return (self(x + h) - self(x - h)) / (2.0 * h)

    def reset_counters(self):
        self.n_calls = 0
        self.n_saturated = 0


class ReconstructedLUT(_TableBase):
    """
    La LUT construida por nosotros con los filtros reales y la linea real
    quieta: repite paso por paso el bucle de HMI_lookup.c:1110-1143.
    """

    def __init__(self, filterset, mu=None):
        self.filterset = filterset
        self.mu = filterset.meta['mu'] if mu is None else mu
        self.vtest = jd.vtest_grid()
        self.vapp = apparent_velocity(filterset, quiet_line(self.mu), self.vtest)
        self.label = (f'reconstruida FSN {filterset.meta["fsn"]} '
                      f'({filterset.meta["row"]},{filterset.meta["col"]})')
        self._finalize()


class JSOCLookupTable(_TableBase):
    """La LUT real del JSOC, en el pixel (row, col) de un registro ya bajado."""

    _cache = {}

    def __init__(self, fsn, row=64, col=64, hcamid=2, nwl=6, harmonic=1):
        key = (fsn, hcamid, nwl)
        if key not in self._cache:
            self._cache[key] = jd.load_real_lut(fsn, hcamid=hcamid, nwl=nwl)
        lut, rec = self._cache[key]
        ntest = jd.parse_hmiparam()['ntest']
        off = 0 if harmonic == 1 else ntest
        self.record = rec
        self.vtest = jd.vtest_grid()
        self.vapp = np.asarray(lut[off:off + ntest, row, col], float)
        self.label = f'JSOC FSN {fsn} ({row},{col}) armonico {harmonic}'
        self._finalize()


def lut_slice(fsn, row=64, col=64, hcamid=2, nwl=6, harmonic=1):
    """El corte crudo de la LUT real, sin envolverlo en una clase."""
    return JSOCLookupTable(fsn, row, col, hcamid, nwl, harmonic).vapp
