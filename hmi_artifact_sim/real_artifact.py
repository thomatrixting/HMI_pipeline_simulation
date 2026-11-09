"""
El experimento del artefacto, con los perfiles reales.

Misma cadena que `artifact.py`, pero cada eslabon es el del pipeline:

    linea real deformada  ->  integral por los 6 filtros reales  ->  1er
    coeficiente de Fourier con pv1 = dvtune*(N-1)  ->  inversion con la LUT
    quieta (la real del JSOC o la reconstruida)  ->  V = (v_L+v_R)/2,
    B_los = (v_L-v_R)*Km

y la misma convencion de resta que documenta `artifact.py:48-56`:

    artefacto = V_medida - (v_true + obs_vr)

Ademas se calcula el artefacto RELATIVO al Sol quieto,

    artefacto_rel = artefacto(umbra) - artefacto(Sol quieto),

que es la cantidad comparable con el residual observado de ~350 m/s, porque
esas series ya tienen el Sol quieto sustraido.

Para aislar de donde sale el residual hay tres estimadores por cada
configuracion:

  'pipeline'  los 6 filtros reales + inversion con la LUT quieta  (lo que HMI hace)
  'raw'       los 6 filtros reales, SIN invertir                  (cuanto aporta la tabla)
  'delta'     muestreo delta en los picos de los filtros          (lo que cancelaria
                                                                   un instrumento ideal)
"""

import numpy as np

from . import config as cfg
from . import jsoc_data as jd
from . import real_filters as rfl
from . import real_pipeline as rp
from .pipeline import fourier_coefficients
from .real_line import calibration_of, quiet_line, umbral_line
from .zeeman import triplet_line


def _line(mode, gamma_deg, **kw):
    """La linea deformada segun el modo Zeeman pedido."""
    if mode == 'rigid':
        return umbral_line(**kw)
    if mode == 'triplet':
        return triplet_line(gamma_deg=gamma_deg, **kw)
    raise ValueError("zeeman debe ser 'rigid' o 'triplet'")


class RealPipeline:
    """
    El pipeline real de un pixel y una epoca.  Cachea los filtros y la tabla.

    Parametros
    ----------
    fsn, row, col : la epoca y el pixel
    lut : 'jsoc' (la tabla real descargada) o 'reconstructed' (la nuestra)
    calibration : 11, 12 o 13; por defecto la que declara el COMMENT de la LUT
    """

    def __init__(self, fsn, row=64, col=64, hcamid=2, lut='jsoc',
                 calibration=None, nlam=32001):
        self.fsn = fsn
        self.row, self.col = row, col
        self.filters = rfl.build_filters(fsn, row, col, hcamid=hcamid, nlam=nlam)
        self.mu = self.filters.meta['mu']

        rec = self.filters.meta['lut_record']
        self.calibration = (calibration if calibration is not None
                            else calibration_of(rec['COMMENT']))

        if lut == 'jsoc':
            self.lut = rp.JSOCLookupTable(fsn, row, col, hcamid=hcamid)
        elif lut == 'reconstructed':
            self.lut = rp.ReconstructedLUT(self.filters, mu=self.mu)
        else:
            raise ValueError("lut debe ser 'jsoc' o 'reconstructed'")
        self.lut_kind = lut

        # posiciones efectivas de muestreo, para el estimador 'delta'
        self.lam_j = self.filters.lam[self.filters.T.argmax(1)]
        self._delta_lut = None

    # ------------------------------------------------------------------
    def apparent(self, line_func, v):
        """Velocidad aparente del 1er Fourier con los filtros reales."""
        return rp.apparent_velocity(self.filters, line_func, v)

    def apparent_delta(self, line_func, v):
        """
        Lo mismo pero muestreando la linea como deltas en los picos de los
        filtros: el instrumento idealizado de la version sintetica.
        """
        p = jd.parse_hmiparam()
        v = np.atleast_1d(np.asarray(v, float))
        I = line_func(self.lam_j[None, :] - v[..., None] * p['dlamdv'])
        return rp.velocities_from_intensities(I)[0]

    def delta_lut(self):
        """La LUT equivalente del estimador delta (para invertirlo igual)."""
        if self._delta_lut is None:
            vt = jd.vtest_grid()
            lut = rp._TableBase()
            lut.vtest = vt
            lut.vapp = self.apparent_delta(quiet_line(self.mu, self.calibration), vt)
            lut.label = 'delta'
            lut._finalize()
            self._delta_lut = lut
        return self._delta_lut

    # ------------------------------------------------------------------
    def observables(self, z_W=0.0, B_gauss=0.0, v_true=0.0, obs_vr=0.0,
                    zeeman='rigid', gamma_deg=0.0, estimator='pipeline',
                    k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S, relative=True):
        """
        Devuelve un dict con las observables.  v_true y obs_vr se difunden,
        asi que pueden ser arrays.

        Claves: v_shift, v_lcp, v_rcp, V, B_los, artifact, artifact_rel,
                saturated, vapp_lcp, vapp_rcp.
        """
        v_shift = (np.atleast_1d(np.asarray(v_true, float))
                   + np.atleast_1d(np.asarray(obs_vr, float)))
        kw = dict(mu=self.mu, calibration=self.calibration,
                  k_c=k_c, k_d=k_d, k_s=k_s)

        fwd = self.apparent_delta if estimator == 'delta' else self.apparent
        if estimator == 'pipeline':
            table = self.lut
        elif estimator == 'delta':
            table = self.delta_lut()
        elif estimator == 'raw':
            table = None
        else:
            raise ValueError("estimator debe ser 'pipeline', 'raw' o 'delta'")

        def chain(line):
            va = fwd(line, v_shift)
            if table is None:
                return va, va
            return va, table(va)

        sat0 = table.n_saturated if table is not None else 0

        lL = _line(zeeman, gamma_deg, z_W=z_W, B_gauss=B_gauss,
                   polarization='LCP', **kw)
        lR = _line(zeeman, gamma_deg, z_W=z_W, B_gauss=B_gauss,
                   polarization='RCP', **kw)
        vappL, vL = chain(lL)
        vappR, vR = chain(lR)

        V = 0.5 * (vL + vR)
        out = dict(v_shift=v_shift, vapp_lcp=vappL, vapp_rcp=vappR,
                   v_lcp=vL, v_rcp=vR, V=V,
                   B_los=(vL - vR) * cfg.KM,
                   artifact=V - v_shift,
                   saturated=(0 if table is None
                              else table.n_saturated - sat0))

        if relative:
            qL = _line(zeeman, gamma_deg, z_W=0.0, B_gauss=0.0,
                       polarization='LCP', **kw)
            _, vq = chain(qL)
            out['V_quiet'] = vq
            out['artifact_rel'] = V - vq
        return out

    def artifact(self, *a, **kw):
        """Solo el artefacto absoluto."""
        return self.observables(*a, **kw)['artifact']

    def artifact_rel(self, *a, **kw):
        """Solo el artefacto relativo al Sol quieto."""
        return self.observables(*a, **kw)['artifact_rel']

    def __repr__(self):
        return (f'<RealPipeline FSN {self.fsn} ({self.row},{self.col}) '
                f'mu={self.mu:.3f} cal={self.calibration} lut={self.lut_kind}>')


def orbital_obs_vr(t_hours, amplitude=cfg.A_ORB, period=cfg.P_ORB_HOURS):
    """OBS_VR(t) sinusoidal: la modulacion orbital de SDO."""
    return amplitude * np.sin(2.0 * np.pi * np.asarray(t_hours, float) / period)
