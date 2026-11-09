"""
Los 6 perfiles de transmision REALES del sistema de filtros de HMI.

Es una traduccion linea por linea de HMI_lookup.c:1088-1094, el bloque que el
pipeline usa para construir la look-up table.  El tren optico es

    T_j(lam) = B(lam) * W(lam)                        <- blocker x front window
             * prod_{E=E2..E5} [1 + C_E cos(2 pi lam / FSR_E + phi_E)] / 2
             * [1 + C_NB cos(2 pi lam / FSR_NB + Phi_NB,j + phi_NB)] / 2
             * [1 + C_WB cos(2 pi lam / FSR_WB + Phi_WB,j + phi_WB)] / 2
             * [1 + C_E1 cos(2 pi lam / FSR_E1 - Phi_E1,j + phi_E1)] / 2

donde

  * B, W                 vienen medidos en HMIparam.h (blocker S/N 11,
                         front window S/N 3);
  * C_*, phi_E2..E5      son los .bin de calibracion de junio de 2009;
  * phi_NB, phi_WB, phi_E1 son el mapa de fases del dia, de
                         hmi.phasemaps_extended;
  * Phi_*,j = (HCM + offset_j) * 6 grados es la fase mecanica del motor de
                         cada Michelson en la posicion de sintonia j.

Los j van de 0 a 5 en orden de longitud de onda DECRECIENTE (I0 = ala roja).
"""

import numpy as np

from . import jsoc_data as jd

N_FILTERS = 6

# HMI_lookup.c:744-763, N = 6.  Offsets en pasos de HCM (6 grados cada paso).
# El offset j=5 de WB repite el de j=0: esta asi en el fuente oficial.
HCM_OFFSET_E1 = np.array([15, 9, 3, -3, -9, -15])
HCM_OFFSET_WB = np.array([-30, -18, -6, 6, 18, -30])
HCM_OFFSET_NB = np.array([0, 24, -12, 12, -24, 0])

# HMI_lookup.c:434-439, el peso de cada posicion en la suma de Fourier
TUNE_ANGLE = np.array([2.5, 1.5, 0.5, -0.5, -1.5, -2.5])


def hcm_phases(hcm_nb, hcm_wb, hcm_e1):
    """
    (Phi_NB, Phi_WB, Phi_E1), cada uno de 6 elementos en RADIANES.

    El `% 360` del fuente es irrelevante dentro de un coseno, pero lo
    reproducimos para que los numeros intermedios coincidan con los de C.
    """
    def ph(hcm, off):
        return np.deg2rad(((hcm + off) * 6) % 360)
    return ph(hcm_nb, HCM_OFFSET_NB), ph(hcm_wb, HCM_OFFSET_WB), \
        ph(hcm_e1, HCM_OFFSET_E1)


def wavelength_grid(nlam=32001):
    """
    La rejilla de longitud de onda del pipeline, centrada en la linea.

    dlam = dvtest*dlamdv = 4.942e-4 A, de modo que desplazar la linea una
    velocidad de prueba es desplazar el array un numero entero de muestras
    (HMI_lookup.c:1113).  nlam impar para que exista el cero exacto.
    """
    p = jd.parse_hmiparam()
    dlam = p['dvtest'] * p['dlamdv']
    return (np.arange(nlam, dtype=float) - (nlam - 1) / 2.0) * dlam, dlam


def prefilter(lam):
    """
    Blocker S/N 11 por front window S/N 3, interpolados a `lam`.

    HMI_lookup.c:879-893: el front window viene en nanometros (x10 - lam0), el
    blocker en Angstrom y re-centrado con CBLOCKER, y los dos en porcentaje.
    """
    p = jd.parse_hmiparam()
    lam_f = p['wavelengthf'] * 10.0 - p['lam0']
    lam_b = p['wavelengthb'] + p['centerblocker'] - p['lam0']
    w = np.interp(lam, lam_f, p['frontwindow'] / 100.0, left=0.0, right=0.0)
    b = np.interp(lam, lam_b, p['blocker'] / 100.0, left=0.0, right=0.0)
    return b * w


def lyot(lam, phase_nt_deg, contrast_nt):
    """
    La parte no sintonizable: prefiltro por los 4 elementos E2..E5.

    `phase_nt_deg` y `contrast_nt` son los 4 valores del pixel.  Se aplican los
    dos ajustes que trae el fuente oficial (HMI_lookup.c:1079-1080):
    phi_E4 += 0.4 rad y phi_E5 -= 1.1 rad.
    """
    p = jd.parse_hmiparam()
    fsr = p['FSR'][3:7]                      # E2, E3, E4, E5
    phi = np.deg2rad(np.asarray(phase_nt_deg, float)).copy()
    phi[2] += jd.PHASE_E4_SHIFT
    phi[3] += jd.PHASE_E5_SHIFT
    out = prefilter(lam)
    for e in range(4):
        out = out * (1.0 + contrast_nt[e] * np.cos(2 * np.pi / fsr[e] * lam
                                                   + phi[e])) / 2.0
    return out


def tunable(lam, lyot_profile, phase_nb, phase_wb, phase_e1,
            contrast_nb, contrast_wb, contrast_e1,
            hcm_nb, hcm_wb, hcm_e1):
    """
    T[6, nlam]: el Lyot por los tres Michelson sintonizables en las 6 posiciones.

    A las fases del mapa del dia se les suma la correccion OBS-CAL de
    HMI_lookup.c:1085-1087 (+1.573, +0.59, -3.27 grados).  Notese el signo
    MENOS de Phi_E1 en el coseno, tal como en el fuente.
    """
    p = jd.parse_hmiparam()
    fsr_nb, fsr_wb, fsr_e1 = p['FSR'][0], p['FSR'][1], p['FSR'][2]
    pnb = np.deg2rad(phase_nb + jd.PHASE_NB_OBSCAL)
    pwb = np.deg2rad(phase_wb + jd.PHASE_WB_OBSCAL)
    pe1 = np.deg2rad(phase_e1 + jd.PHASE_E1_OBSCAL)
    Phi_nb, Phi_wb, Phi_e1 = hcm_phases(hcm_nb, hcm_wb, hcm_e1)

    T = np.empty((N_FILTERS, lam.size))
    for j in range(N_FILTERS):
        T[j] = (lyot_profile
                * (1.0 + contrast_nb * np.cos(2 * np.pi / fsr_nb * lam
                                              + Phi_nb[j] + pnb)) / 2.0
                * (1.0 + contrast_wb * np.cos(2 * np.pi / fsr_wb * lam
                                              + Phi_wb[j] + pwb)) / 2.0
                * (1.0 + contrast_e1 * np.cos(2 * np.pi / fsr_e1 * lam
                                              - Phi_e1[j] + pe1)) / 2.0)
    return T


class FilterSet:
    """Los 6 filtros reales de un pixel y una epoca, con sus metadatos."""

    def __init__(self, lam, T, meta):
        self.lam = lam
        self.T = T
        self.meta = meta

    @property
    def dlam(self):
        return self.lam[1] - self.lam[0]

    def peaks(self):
        """
        Posicion del lobulo principal de cada filtro, en mA: el punto de
        muestreo efectivo.  (El centro de gravedad no sirve aqui: las alas del
        blocker se extienden +-8 A y lo dominan.)
        """
        return 1e3 * self.lam[self.T.argmax(1)]

    def centroid(self):
        """Centro de gravedad de cada filtro sobre toda la rejilla, en mA."""
        w = np.clip(self.T, 0, None)
        return 1e3 * (w * self.lam).sum(1) / w.sum(1)

    def fwhm(self):
        """FWHM del lobulo principal de cada filtro, en mA."""
        out = []
        for j in range(self.T.shape[0]):
            t = self.T[j]
            k = int(np.argmax(t))
            half = t[k] / 2.0
            lo = k
            while lo > 0 and t[lo] > half:
                lo -= 1
            hi = k
            while hi < t.size - 1 and t[hi] > half:
                hi += 1
            out.append(1e3 * (self.lam[hi] - self.lam[lo]))
        return np.array(out)


def build_filters(fsn, row=64, col=64, hcamid=2, nlam=32001, hcm=None, nx=128):
    """
    Construye los 6 filtros reales del pixel (row, col) para la epoca `fsn`.

    Por defecto toma las posiciones de sintonia del registro de
    hmi.lookup_corrected_expanded (HWL4/HWL2/HWL1POS), que es lo correcto para
    reproducir la LUT real; `hcm=(NB, WB, E1)` las fuerza a mano.
    """
    phases, pm_rec = jd.load_phasemap(fsn, hcamid=hcamid, nx=nx)
    phaseNT, contrastNT, contrastT = jd.load_calibration_bins(nx)

    if hcm is None:
        lut_recs = jd.jsoc_records(
            'hmi.lookup_corrected_expanded', fsn,
            keys='HCAMID,NWL,HWL1POS,HWL2POS,HWL3POS,HWL4POS,COMMENT')
        lr = jd._pick(lut_recs, HCAMID=hcamid, NWL=6)
        hcm = jd.hcm_from_lut_record(lr)
        hcm_src = 'lookup_corrected_expanded HWL4/2/1POS'
    else:
        lr = None
        hcm_src = 'forzado por el usuario'

    lam, dlam = wavelength_grid(nlam)
    ly = lyot(lam, phaseNT[:, row, col], contrastNT[:, row, col])
    T = tunable(lam, ly,
                phases[0, row, col], phases[1, row, col], phases[2, row, col],
                contrastT[0, row, col], contrastT[1, row, col],
                contrastT[2, row, col],
                *hcm)

    meta = dict(fsn=fsn, date=jd.EPOCHS.get(fsn), row=row, col=col,
                hcamid=hcamid, hcm=tuple(hcm), hcm_source=hcm_src, nlam=nlam,
                dlam=dlam, mu=mu_of_pixel(row, col, nx),
                phase_nb=phases[0, row, col], phase_wb=phases[1, row, col],
                phase_e1=phases[2, row, col],
                contrast_nb=contrastT[0, row, col],
                contrast_wb=contrastT[1, row, col],
                contrast_e1=contrastT[2, row, col],
                phasemap=pm_rec, lut_record=lr)
    return FilterSet(lam, T, meta)


def mu_of_pixel(row, col, nx=128):
    """
    mu = cos(theta) del pixel, exactamente como HMI_lookup.c:672-673:

        d[arcsec] = hypot(col - X0, row - Y0) * NOMINALSCALE * ratio
        mu        = cos(asin(d / solarradiustable))        si d <= R
                    0.0  (zona de amortiguacion)  o  NaN   si no

    con X0 = Y0 = (nx-1)/2, ratio = 4096/nx, NOMINALSCALE = 0.503 "/px y
    solarradiustable = 976.22" (el radio maximo del Sol visto por HMI).
    """
    p = jd.parse_hmiparam()
    ratio = 4096 // nx
    x0 = y0 = (nx - 1) / 2.0
    d = np.hypot(col - x0, row - y0) * p['NOMINALSCALE'] * ratio
    rsun = p['solarradiustable']
    if d <= rsun:
        return float(np.cos(np.arcsin(d / rsun)))
    if d <= (2047.5 - ratio) * p['NOMINALSCALE']:
        return 0.0            # BUFFERDISTANCE
    return np.nan             # WRONGDISTANCE


def pixel_for_mu(target_mu, nx=128):
    """El pixel (row, col) de la fila central cuyo mu mas se acerca al pedido."""
    cols = np.arange(nx)
    mus = np.array([mu_of_pixel(nx // 2, c, nx) for c in cols])
    ok = np.isfinite(mus)
    j = int(cols[ok][np.nanargmin(np.abs(mus[ok] - target_mu))])
    return nx // 2, j
