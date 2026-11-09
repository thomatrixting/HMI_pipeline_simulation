"""
Tests de la ruta con perfiles REALES.

Se ejecuta con
    python -m hmi_artifact_sim.tests_real
y necesita el cache de `hmi_artifact_sim/data/` (la primera vez descarga).

El test que decide todo es el 4: la LUT reconstruida contra la real del JSOC.
"""

import sys

import numpy as np

from . import config as cfg
from . import jsoc_data as jd
from . import real_artifact as ra
from . import real_filters as rfl
from . import real_line as rl
from . import real_pipeline as rp
from . import zeeman as zz

_FAIL = []


def check(ok, nombre, detalle=''):
    print(f'[{"PASS" if ok else "FAIL"}] {nombre}' + (f'  --  {detalle}' if detalle else ''))
    if not ok:
        _FAIL.append(nombre)


def main():
    print('=' * 78)
    print('TESTS CON PERFILES REALES')
    print('=' * 78)

    # --- 1. el parser de HMIparam.h -------------------------------------
    p = jd.parse_hmiparam()
    check(np.isclose(p['FSR'][0], 0.1689) and np.isclose(p['FSR'][6], 11.354),
          'HMIparam.h: FSR', f'{p["FSR"]}')
    check(p['ntest'] == 821 and np.isclose(p['dvtest'], 24.0),
          'HMIparam.h: ntest y dvtest', f'{p["ntest"]} velocidades cada {p["dvtest"]:.0f} m/s')
    check(np.isclose(p['dlamdv'], 2.059205672212074294e-5),
          'HMIparam.h: dlamdv', f'{p["dlamdv"]:.6e} A por (m/s)')
    check(p['blocker'].size == 201 and p['frontwindow'].size == 401,
          'HMIparam.h: blocker y front window',
          f'{p["blocker"].size} y {p["frontwindow"].size} puntos')
    check(np.allclose(p['minimumCoeffs'], (0.41922611, 0.24190794)),
          'coeficientes centro-limbo de profundidad')

    chk = jd.check_lookup_c_constants()
    check(all(chk.values()), 'constantes verificadas contra HMI_lookup.c',
          ', '.join(k for k, v in chk.items() if not v) or 'las 10 presentes')

    # --- 2. los filtros --------------------------------------------------
    fs = rfl.build_filters(23845068, 64, 64)
    picos, anchos = fs.peaks(), fs.fwhm()
    esperado = np.array([169, 101, 34, -34, -101, -169])
    check((fs.T >= 0).all(), 'transmision no negativa en toda la rejilla')
    check(np.all(np.diff(picos) < 0), 'orden j = 0..5 de rojo a azul',
          f'{np.round(picos, 1)} mA')
    check(np.abs(picos - esperado).max() < 10,
          'lobulos principales cerca de +-169/101/34 mA',
          f'desviacion maxima {np.abs(picos - esperado).max():.1f} mA')
    check(abs(anchos.mean() - 76) < 5, 'FWHM de los filtros ~76 mA',
          f'{anchos.mean():.1f} mA (LW_MEAN del dia: '
          f'{1e3*float(fs.meta["phasemap"]["LW_MEAN"]):.1f})')
    dtune = rp.velocity_scale()[0]
    check(abs(np.abs(np.diff(picos)).mean() - 1e3 * dtune) < 2,
          'separacion entre filtros ~ FSR_NB/2.5',
          f'{np.abs(np.diff(picos)).mean():.2f} contra {1e3*dtune:.2f} mA')

    # --- 3. la linea -----------------------------------------------------
    L = rl.quiet_line(1.0, 13)
    check(abs(L.centre()) < 1.0, 'la linea del pipeline esta centrada',
          f'minimo en {L.centre():+.2f} mA')
    bin_line, bin_wl = rl.reference_bin_line()
    check(abs(1e3 * bin_wl[bin_line.argmin()]) > 50,
          'ReferenceFeLine.bin NO esta centrado (por eso el pipeline no lo usa)',
          f'minimo en {1e3*bin_wl[bin_line.argmin()]:+.0f} mA')
    # A mu = 1 el factor de escala es exactamente 1: el pipeline usa la Voigt
    # tal cual.  Los minimumCoeffs/FWHMCoeffs solo entran como RAZON respecto a
    # mu = 1, asi que la profundidad absoluta de la Voigt (0.640) no tiene por
    # que coincidir con Id(1) = 0.661, que viene de ajustar la linea de Ulrich.
    check(np.allclose(L.prof, rl.voigt_line(rl.reference_grid(), 13)),
          'a mu = 1 no hay reescalado: la linea es la Voigt tal cual')
    check(np.allclose(L.lam_ref, rl.reference_grid()),
          'a mu = 1 la rejilla tampoco se reescala')
    L06 = rl.quiet_line(0.6, 13)
    d1 = L.Ic - L.prof.min()
    d06 = L06.Ic - L06.prof.min()
    check(abs(d06 / d1 - L06.Id / L.Id) < 1e-3,
          'la razon de profundidades sigue a minimumCoeffs(mu)',
          f'{d06/d1:.4f} contra {L06.Id/L.Id:.4f}')
    check(L06.Id < L.Id and L06.fwhm > L.fwhm,
          'centro-limbo: mas somera y mas ancha hacia el limbo',
          f'mu=1: Id={L.Id:.3f} FWHM={L.fwhm:.1f} | mu=0.6: Id={L06.Id:.3f} FWHM={L06.fwhm:.1f}')
    x = np.linspace(-0.5, 0.5, 601)
    check(np.allclose(rl.umbral_line(0, 0, 'LCP')(x), rl.quiet_line(1.0, 13)(x)),
          'umbral_line(z_W=0, B=0) == linea quieta')

    # --- 4. reproduccion de la LUT real ----------------------------------
    vt = jd.vtest_grid()
    peor = 0.0
    for fsn in jd.EPOCHS:
        lut_real, rec = jd.load_real_lut(fsn)
        cal = rl.calibration_of(rec['COMMENT'])
        f_ = rfl.build_filters(fsn, 64, 64)
        mia = rp.apparent_velocity(f_, rl.quiet_line(f_.meta['mu'], cal), vt)
        rms = float(np.sqrt(np.mean((mia - lut_real[:vt.size, 64, 64]) ** 2)))
        peor = max(peor, rms)
        check(rms < 25, f'LUT reconstruida == LUT real, FSN {fsn} (cal {cal})',
              f'rms = {rms:.1f} m/s')
        del lut_real
    check(peor < 25, 'las tres epocas dentro de 25 m/s', f'peor rms = {peor:.1f} m/s')

    # la calibracion correcta se identifica desde el residual
    lut_real, rec = jd.load_real_lut(23845068)
    real = np.asarray(lut_real[:vt.size, 64, 64], float)
    del lut_real
    rms_cal = {}
    for cal in (11, 12, 13):
        mia = rp.apparent_velocity(fs, rl.quiet_line(fs.meta['mu'], cal), vt)
        rms_cal[cal] = float(np.sqrt(np.mean((mia - real) ** 2)))
    check(min(rms_cal, key=rms_cal.get) == 11,
          'el residual identifica la calibracion declarada en COMMENT',
          '  '.join(f'cal {c}: {v:.1f}' for c, v in rms_cal.items()))

    # --- 5. experimento A ------------------------------------------------
    P = ra.RealPipeline(23845068, 64, 64)
    v_true = np.linspace(-4000, 4000, 41)
    peor_a = 0.0
    for ov in (-3000.0, 0.0, 3000.0):
        o = P.observables(0.0, 0.0, v_true, ov)
        peor_a = max(peor_a, float(np.abs(o['artifact']).max()))
        check(float(np.abs(o['B_los']).max()) == 0.0,
              f'B_los == 0 con B = 0 (OBS_VR = {ov:+.0f})')
    check(peor_a < p['dvtest'], 'EXP. A: |artefacto| por debajo del paso de la LUT',
          f'max {peor_a:.2f} m/s contra dvtest = {p["dvtest"]:.0f} m/s')

    # --- 6. la correccion Zeeman -----------------------------------------
    peor_z = max(abs(P.observables(0.0, b, 0.0, ov)['artifact_rel'][0])
                 for b in (0.0, 1000.0, 2000.0, 3000.0) for ov in (0.0, 3000.0))
    check(peor_z < 10, 'la media LCP/RCP cancela el Zeeman si el perfil es simetrico',
          f'max |artefacto| = {peor_z:.2f} m/s hasta 3000 G')
    o = P.observables(0.0, 1000.0, 0.0, 0.0)
    check(abs(o['B_los'][0] - 1000.0) < 5, 'B_los recupera el campo de entrada',
          f'1000 G -> {o["B_los"][0]:.1f} G')
    acoplado = P.observables(600.0, 2500.0, 0.0, 3000.0)['artifact_rel'][0]
    check(abs(acoplado) > 50, 'Zeeman x Wilson x orbita SI deja residual',
          f'{acoplado:+.1f} m/s (z_W=600, B=2500, OBS_VR=3000)')

    # --- 7. el triplete --------------------------------------------------
    for pol in ('LCP', 'RCP'):
        a = zz.triplet_line(600, 2500, 0.0, pol)(x)
        b = rl.umbral_line(600, 2500, pol)(x)
        check(np.allclose(a, b), f'triplete con gamma=0 == corrimiento rigido ({pol})',
              f'max dif {np.abs(a-b).max():.2e}')
    check(np.allclose(zz.triplet_line(600, 2500, 90.0, 'LCP')(x),
                      zz.triplet_line(600, 2500, 90.0, 'RCP')(x)),
          'con gamma=90 las dos polarizaciones coinciden')
    for g in (0, 30, 45, 60, 90):
        w_pi, w_s, _ = zz.component_weights(g)
        if not np.isclose(w_pi + 2 * w_s, 1.0):
            check(False, 'los pesos del triplete suman 1', f'gamma={g}')
            break
    else:
        check(True, 'los pesos del triplete suman 1 para todo gamma')
    o = P.observables(600, 2500, 0.0, 3000.0, zeeman='triplet', gamma_deg=45)
    check(abs(o['artifact_rel'][0]) > abs(acoplado),
          'el triplete da mas artefacto que el corrimiento rigido',
          f'{o["artifact_rel"][0]:+.1f} contra {acoplado:+.1f} m/s')

    # --- 8. coherencia con la version sintetica --------------------------
    check(abs(rp.velocity_scale()[2] / 2 - 8202) < 5,
          'pv1/2 = +-8.2 km/s (el (N-1) del fuente)',
          f'{rp.velocity_scale()[2]/2:.0f} m/s')
    check(abs(cfg.V_UNAMBIGUOUS - 10023) < 5,
          'la version sintetica sigue con +-10.0 km/s (sin tocar)',
          f'{cfg.V_UNAMBIGUOUS:.0f} m/s')

    print('-' * 78)
    print(f'{"TODOS OK" if not _FAIL else "FALLOS: " + ", ".join(_FAIL)}')
    return 1 if _FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
