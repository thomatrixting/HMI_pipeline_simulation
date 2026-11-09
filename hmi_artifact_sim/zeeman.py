"""
Apartado: el perfil Zeeman completo, en vez del corrimiento rigido.

Fe I 6173.3433 tiene g_eff = 2.5 y es un triplete Zeeman normal (J=1 -> J=0),
asi que se parte en tres componentes:

    sigma_azul  en  lam0 - Delta lam_B
    pi          en  lam0
    sigma_rojo  en  lam0 + Delta lam_B,     Delta lam_B = 4.67e-13 g lam0^2 B

Con el campo inclinado un angulo gamma respecto a la linea de vision, los
perfiles de absorcion de Stokes I y V son (triplete normal, linea opticamente
delgada; Landi Degl'Innocenti)

    phi_I = (1/2) phi_pi sin^2(gamma) + (1/4) (phi_b + phi_r) (1 + cos^2 gamma)
    phi_V = (1/2) (phi_r - phi_b) cos(gamma)

y las dos polarizaciones circulares que mide HMI son I +- V:

    LCP: phi_I + phi_V        RCP: phi_I - phi_V

Los pesos suman 1 exactamente, asi que la anchura equivalente se conserva.

Dos limites que sirven de test:

  * gamma = 0   ->  LCP = phi_r y RCP = phi_b: es EXACTAMENTE el corrimiento
                    rigido +-Delta lam_B de `real_line.umbral_line`.
  * gamma = 90  ->  phi_V = 0, LCP = RCP, y el magnetograma da cero: un campo
                    puramente transversal no produce senal longitudinal.

Limitacion honesta: esto es superposicion de componentes sobre un perfil de
absorcion dado, no una sintesis Milne-Eddington.  No incluye saturacion de la
linea ni el acoplamiento de la funcion fuente con el campo, que a 2500 G ya
importan.  Es, eso si, estrictamente mejor que el corrimiento rigido, porque
mantiene la componente pi en el centro, que es la que rompe la cancelacion
LCP/RCP.
"""

import numpy as np

from . import config as cfg
from .profiles import zeeman_shift
from .real_line import quiet_line, wilson_factors


def component_weights(gamma_deg):
    """
    (w_pi, w_sigma) para Stokes I y el peso de V, con gamma en grados.

    Devuelve (w_pi, w_sigma, w_V) tales que
        phi_I = w_pi*phi_pi + w_sigma*(phi_b + phi_r)
        phi_V = w_V*(phi_r - phi_b)
    """
    g = np.deg2rad(gamma_deg)
    w_pi = 0.5 * np.sin(g) ** 2
    w_sigma = 0.25 * (1.0 + np.cos(g) ** 2)
    w_V = 0.5 * np.cos(g)
    return w_pi, w_sigma, w_V


def triplet_line(z_W=0.0, B_gauss=0.0, gamma_deg=0.0, polarization='LCP',
                 mu=1.0, calibration=13,
                 k_c=cfg.K_C, k_d=cfg.K_D, k_s=cfg.K_S):
    """
    Callable lam -> L(lam): la linea real con triplete Zeeman completo y
    deformacion de Wilson, para 'LCP', 'RCP' o 'I' (Stokes I sin polarizar).
    """
    base = quiet_line(mu, calibration)
    f_c, f_d, f_s = wilson_factors(z_W, k_c, k_d, k_s)
    dlB = zeeman_shift(B_gauss)
    w_pi, w_sigma, w_V = component_weights(gamma_deg)

    pol = polarization.upper()
    if pol == 'LCP':
        sign = +1.0
    elif pol == 'RCP':
        sign = -1.0
    elif pol in ('I', 'NONE'):
        sign = 0.0
    else:
        raise ValueError("polarization debe ser 'LCP', 'RCP' o 'I'")

    def _line(lam):
        lam = np.asarray(lam, float)
        phi_p = base.depth_profile(lam / f_s)
        phi_b = base.depth_profile((lam + dlB) / f_s)
        phi_r = base.depth_profile((lam - dlB) / f_s)
        phi_I = w_pi * phi_p + w_sigma * (phi_b + phi_r)
        phi_V = w_V * (phi_r - phi_b)
        return base.Ic * f_c - f_d * (phi_I + sign * phi_V)

    return _line


def line_factory(mode='rigid'):
    """
    Devuelve la funcion que construye la linea segun el modo pedido:
    'rigid'   -> real_line.umbral_line  (corrimiento +-Delta lam_B)
    'triplet' -> triplet_line           (sigma+, pi, sigma- con inclinacion)

    Las dos aceptan los mismos argumentos con nombre, de modo que el resto del
    codigo no cambia; 'rigid' simplemente ignora gamma_deg.
    """
    if mode == 'rigid':
        from .real_line import umbral_line

        def _rigid(gamma_deg=0.0, **kw):
            return umbral_line(**kw)
        return _rigid
    if mode == 'triplet':
        return triplet_line
    raise ValueError("mode debe ser 'rigid' o 'triplet'")
