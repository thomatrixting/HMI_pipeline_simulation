"""
Descarga y cache de los insumos *reales* del pipeline HMI.

Nada de esto se inventa: todo sale de dos fuentes publicas y verificadas,

  (1) el repositorio oficial del JSOC, github.com/JSOC-SDP/proj
      -> lev1.5_hmi/libs/lev15/HMIparam.h      (FSR, blocker, front window,
                                                ntest, dvtest, dlamdv, lam0)
      -> lev1.5_hmi/apps/HMI_lookup.c          (el codigo que construye la LUT)
      -> lev1.5_hmi/apps/*.bin                 (fases y contrastes de calibracion)

  (2) las series DRMS publicas del JSOC, consultadas por el CGI `jsoc_info`
      -> hmi.phasemaps_extended[FSN]           (fases de NB, WB, E1)
      -> hmi.lookup_corrected_expanded[FSN]    (la look-up table real)

`hmi.phasemaps_corrected`, la serie que menciona la guia, NO es accesible desde
fuera de Stanford ("not a valid series accessible from hmidb2").
`hmi.phasemaps_extended` si lo es y trae las mismas fases mas los keywords
FSRNB..FSRE5, CBLOCKER, HCMNB/WB/E1, R_SUN, CRPIX1/2, OBS_VR, CROTA2.

Todo queda cacheado en `hmi_artifact_sim/data/`, asi que la primera ejecucion
baja ~325 MB y las siguientes no bajan nada.
"""

import json
import os
import pathlib
import re
import urllib.parse
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# Rutas y endpoints
# ---------------------------------------------------------------------------
CACHE = pathlib.Path(__file__).resolve().parent / 'data'

GITHUB_RAW = 'https://raw.githubusercontent.com/JSOC-SDP/proj/master/lev1.5_hmi/'
JSOC_INFO = 'http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info'
JSOC_SUMS = 'https://jsoc1.stanford.edu'

TIMEOUT = 300

# Los tres phasemaps/LUT que comparamos.  HCAMID = 2 es la camara lateral, la
# que produce los observables de 720 s.
EPOCHS = {
    23845068: '2011-06-22',
    65714753: '2013-12-18',
    104314890: '2016-04-05',
}

# ---------------------------------------------------------------------------
# Constantes que viven en HMI_lookup.c y no en HMIparam.h.  Se declaran aqui
# con su numero de linea, y `check_lookup_c_constants()` vuelve a leer el
# fuente oficial para confirmar que siguen siendo esas.
# ---------------------------------------------------------------------------
# HMI_lookup.c:268  minimum intensity (= Id con continuo 1) vs mu
MINIMUM_COEFFS = (0.41922611, 0.24190794)
# HMI_lookup.c:269  FWHM de la linea solar en mA vs mu
FWHM_COEFFS = (151.34559, -58.521771)
# HMI_lookup.c:1079-1080  desplazamiento de los elementos no sintonizables
PHASE_E4_SHIFT = 0.4     # rad
PHASE_E5_SHIFT = -1.1    # rad
# HMI_lookup.c:1085-1087  correccion OBS-CAL de los sintonizables, en grados
PHASE_NB_OBSCAL = 1.573
PHASE_WB_OBSCAL = 0.59
PHASE_E1_OBSCAL = -3.27


# ---------------------------------------------------------------------------
# Descarga con cache
# ---------------------------------------------------------------------------
def _get(url, dest):
    """Descarga `url` a `dest` si no esta ya; devuelve el Path."""
    dest = pathlib.Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def fetch_github(relpath):
    """Un archivo de lev1.5_hmi/ del repo oficial del JSOC -> Path local."""
    return _get(GITHUB_RAW + relpath, CACHE / 'jsoc_proj' / relpath)


def fetch_segment(sums_path):
    """Un segmento DRMS servido desde /SUM*/... -> Path local."""
    name = sums_path.strip('/').replace('/', '_')
    return _get(JSOC_SUMS + sums_path, CACHE / 'segments' / name)


def jsoc_records(series, fsn, keys='**ALL**'):
    """
    Consulta `series[fsn]` por el CGI jsoc_info y devuelve una lista de dicts,
    uno por registro, con los keywords pedidos mas {'segments': {nombre: ruta}}.
    """
    cache = CACHE / 'queries' / f'{series}_{fsn}.json'
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        q = urllib.parse.urlencode({
            'op': 'rs_list', 'ds': f'{series}[{fsn}]',
            'key': keys, 'seg': '**ALL**',
        })
        with urllib.request.urlopen(f'{JSOC_INFO}?{q}', timeout=TIMEOUT) as r:
            payload = json.loads(r.read().decode())
        if payload.get('status') != 0:
            raise RuntimeError(f'jsoc_info: {payload.get("error", payload)}')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))

    kv = {k['name']: k['values'] for k in payload.get('keywords', [])}
    sg = {s['name']: s['values'] for s in payload.get('segments', [])}
    out = []
    for i in range(payload['count']):
        rec = {k: v[i] for k, v in kv.items()}
        rec['segments'] = {k: v[i] for k, v in sg.items()}
        out.append(rec)
    return out


def _pick(records, **conds):
    """El unico registro que cumple todas las condiciones (como strings)."""
    hits = [r for r in records
            if all(str(r.get(k)) == str(v) for k, v in conds.items())]
    if len(hits) != 1:
        raise RuntimeError(f'esperaba 1 registro con {conds}, hay {len(hits)}')
    return hits[0]


# ---------------------------------------------------------------------------
# HMIparam.h: parser de los arrays y escalares de C
# ---------------------------------------------------------------------------
def _strip_c_comments(src):
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)


def _c_array(src, name):
    m = re.search(re.escape(name) + r'\s*\[[^\]]*\]\s*=\s*\{(.*?)\}', src, re.S)
    if m is None:
        raise KeyError(name)
    return np.array([float(x) for x in m.group(1).replace('\n', '').split(',')
                     if x.strip()])


def _c_scalar(src, name):
    m = re.search(re.escape(name) + r'\s*=\s*([-+0-9.eE]+)', src)
    if m is None:
        raise KeyError(name)
    return float(m.group(1).rstrip('.eE') or m.group(1))


_HMIPARAM = None


def parse_hmiparam():
    """
    Lee HMIparam.h del repo oficial y devuelve sus constantes.  Es un parser,
    no una re-escritura: los numeros son literalmente los del archivo.
    """
    global _HMIPARAM
    if _HMIPARAM is not None:
        return _HMIPARAM
    src = _strip_c_comments(fetch_github('libs/lev15/HMIparam.h').read_text())
    p = {
        'FSR': _c_array(src, 'FSR'),                    # [NB,WB,E1,E2,E3,E4,E5]
        'blocker': _c_array(src, 'blockerd'),           # 201 pts, en %
        'wavelengthb': _c_array(src, 'wavelengthbd'),   # 201 pts, Angstrom
        'frontwindow': _c_array(src, 'frontwindowd'),   # 401 pts, en %
        'wavelengthf': _c_array(src, 'wavelengthd'),    # 401 pts, nanometros
        'centerblocker': _c_scalar(src, 'centerblocker'),
        'lam0': _c_scalar(src, 'lam0'),
        'ntest': int(_c_scalar(src, 'ntest')),
        'dvtest': _c_scalar(src, 'dvtest'),
        'dlamdv': _c_scalar(src, 'dlamdv'),
        'referencenlam': int(_c_scalar(src, 'referencenlam')),
        'nblocker': int(_c_scalar(src, 'nblocker')),
        'nfront': int(_c_scalar(src, 'nfront')),
        'solarradiustable': _c_scalar(src, 'solarradiustable'),
        'NOMINALSCALE': _c_scalar(src, 'NOMINALSCALE'),
    }
    p['minimumCoeffs'] = np.array(MINIMUM_COEFFS)
    p['FWHMCoeffs'] = np.array(FWHM_COEFFS)
    _HMIPARAM = p
    return p


def check_lookup_c_constants():
    """
    Vuelve a leer HMI_lookup.c y confirma que las constantes declaradas arriba
    siguen apareciendo tal cual en el fuente oficial.  Devuelve un dict
    {nombre: True/False}.
    """
    src = fetch_github('apps/HMI_lookup.c').read_text()
    def present(pat):
        return re.search(pat, src, re.M) is not None
    return {
        'minimumCoeffs': present(r'minimumCoeffs\[2\]\s*=\s*\{0\.41922611,0\.24190794\}'),
        'FWHMCoeffs': present(r'FWHMCoeffs\[2\]\s*=\s*\{151\.34559,-58\.521771\}'),
        'phaseE4+=0.4': present(r'^\s*phaseE4\+=0\.4;'),
        'phaseE5-=1.1': present(r'^\s*phaseE5-=1\.1;'),
        'phaseNB obs-cal': present(r'phaseNB\+=1\.573\*M_PI/180\.'),
        'phaseWB obs-cal': present(r'phaseWB\+=0\.59\*M_PI/180\.'),
        'phaseE1 obs-cal': present(r'phaseE1\+=-3\.27\*M_PI/180\.'),
        'pv1=dvtune*(N-1)': present(r'pv1\s*=\s*dvtune\*\(double\)\(N-1\)'),
        'atan2(-f1s,-f1c)': present(r'atan2\(-f1s,-f1c\)'),
        'mu=cos(asin(r/Rsun))': present(r'distance\[i\]=cos\(asin\('),
    }


# ---------------------------------------------------------------------------
# Binarios de calibracion (fases/contrastes que no cambian con la re-sintonia)
# ---------------------------------------------------------------------------
def load_calibration_bins(nx=128):
    """
    phaseNT[4,nx,nx]    fases de E2..E5, en grados
    contrastNT[4,nx,nx] contrastes de E2..E5
    contrastT[3,nx,nx]  contrastes de NB, WB, E1

    Orden IDL phase[nx,nx,n] -> C phase[elemento][fila][columna].
    """
    def rd(name, n):
        a = np.fromfile(fetch_github('apps/' + name), dtype='<f4')
        if a.size != n * nx * nx:
            raise RuntimeError(f'{name}: {a.size} floats, esperaba {n*nx*nx}')
        return a.reshape(n, nx, nx).astype(float)

    phaseNT = rd(f'non_tunable_phases_710660_June09_cal_{nx}_2.bin', 4)
    contrastNT = rd(f'non_tunable_contrasts_710660_June09_cal_{nx}_2.bin', 4)
    contrastT = rd(f'tunable_contrasts_710660_June09_cal_{nx}.bin', 3)
    return phaseNT, contrastNT, contrastT


# ---------------------------------------------------------------------------
# Linea Fe I de referencia (R. Ulrich, centro de disco, Sol quieto)
# ---------------------------------------------------------------------------
def load_reference_line():
    """
    (templineref, wavelengthref) con 7000 puntos.

    OJO: HMI_lookup.c:248 SOBREESCRIBE la rejilla leida de
    ReferenceWavelength.bin por una uniforme

        wavelengthref[k] = k/(referencenlam-1) - 0.5

    es decir [-0.5, +0.5] Angstrom sin cero exacto.  Replicamos ese override,
    porque es lo que realmente usa el pipeline al construir la LUT.
    """
    p = parse_hmiparam()
    n = p['referencenlam']
    line = np.fromfile(fetch_github('apps/ReferenceFeLine.bin'), dtype='<f4')
    if line.size != n:
        raise RuntimeError(f'ReferenceFeLine.bin: {line.size} != {n}')
    wl = np.arange(n, dtype=float) / (n - 1.0) - 0.5
    return line.astype(float), wl


# ---------------------------------------------------------------------------
# Segmentos DRMS
# ---------------------------------------------------------------------------
def _fits_data(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=float)
    raise RuntimeError(f'{path}: sin datos')


def load_phasemap(fsn, hcamid=2, nx=128):
    """
    Devuelve (phases, keywords) con phases[5, nx, nx] en GRADOS.
    Los planos 0,1,2 son NB, WB, E1 (los 3,4 son ancho y profundidad del ajuste).
    """
    recs = jsoc_records('hmi.phasemaps_extended', fsn,
                        keys='T_REC,HCAMID,NX,HCMNB,HCMWB,HCME1,HCMPOL,R_SUN,'
                             'CRPIX1,CRPIX2,CDELT1,FSRNB,FSRWB,FSRE1,FSRE2,'
                             'FSRE3,FSRE4,FSRE5,CBLOCKER,LW_MEAN,LD_MEAN,'
                             'CON_MEAN,OBS_VR,CROTA2,COMMENT')
    rec = _pick(recs, HCAMID=hcamid, NX=nx)
    data = _fits_data(fetch_segment(rec['segments']['phases']))
    # FITS NAXIS1=5 -> en numpy queda (nx, nx, 5): el eje de elementos ultimo
    if data.shape[-1] == 5:
        data = np.moveaxis(data, -1, 0)
    if data.shape != (5, nx, nx):
        raise RuntimeError(f'phases con forma {data.shape}')
    return data, rec


def load_real_lut(fsn, hcamid=2, nwl=6):
    """
    La look-up table real del pipeline.

    Devuelve (lut, keywords) con lut[2*ntest, nx, nx]:
      lut[0:ntest]      velocidad aparente del 1er coeficiente de Fourier
      lut[ntest:2*ntest] idem del 2o, ya desenvuelta contra la del 1o
    evaluada en vtest = dvtest*(i - (ntest-1)/2).
    """
    recs = jsoc_records('hmi.lookup_corrected_expanded', fsn,
                        keys='T_REC,HCAMID,NWL,HWL1POS,HWL2POS,HWL3POS,'
                             'HWL4POS,R_SUN,FSRNB,FSRWB,FSRE1,FSRE2,FSRE3,'
                             'FSRE4,FSRE5,CBLOCKER,OBS_VR,CROTA2,COMMENT')
    rec = _pick(recs, HCAMID=hcamid, NWL=nwl)
    data = _fits_data(fetch_segment(rec['segments']['lookup']))
    # FITS NAXIS1 = 2*ntest -> numpy (nx, nx, 2*ntest)
    if data.ndim == 3 and data.shape[0] != 2 * parse_hmiparam()['ntest']:
        data = np.moveaxis(data, -1, 0)
    return data, rec


def vtest_grid():
    """La rejilla de velocidades de prueba de la LUT: 821 puntos cada 24 m/s."""
    p = parse_hmiparam()
    n, dv = p['ntest'], p['dvtest']
    return dv * (np.arange(n, dtype=float) - (n - 1.0) / 2.0)


def hcm_from_lut_record(rec):
    """
    (HCMNB, HCMWB, HCME1) de un registro de lookup_corrected_expanded.

    HWL4POS = NB, HWL2POS = WB, HWL1POS = E1 (HWL3POS es el polarizador).
    Son estos y NO los HCMNB/HCMWB/HCME1 del phasemap los que hay que usar
    para reproducir la LUT: difieren en un paso de HCM (6 grados) en algunas
    epocas.
    """
    return int(rec['HWL4POS']), int(rec['HWL2POS']), int(rec['HWL1POS'])


def cache_report():
    """Lista lo que hay en el cache: (ruta relativa, bytes)."""
    out = []
    for p in sorted(CACHE.rglob('*')):
        if p.is_file() and not p.name.endswith('.part'):
            out.append((str(p.relative_to(CACHE)), p.stat().st_size))
    return out
