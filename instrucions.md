# Especificación para Claude Code: simulación del artefacto OBS_VR en umbras HMI

## 0. Contexto para el agente (leer primero)

Estás implementando un **modelo forward sintético del pipeline de velocidad Doppler de
SDO/HMI**, siguiendo el algoritmo MDI-like descrito en Couvidat et al. (2011),
"Calculation of Observables from HMI Data".

El objetivo científico es **demostrar cuantitativamente** que un residual oscilatorio de
~350 m/s observado sobre umbras de manchas solares (en datos con el Sol quieto ya
sustraído) es un **artefacto instrumental**, no una velocidad real. La hipótesis es que
el artefacto surge porque:

1. El pipeline invierte la velocidad usando una tabla (look-up table) calibrada con un
   perfil Fe I 6173 Å de **Sol quieto** (gaussiano).
2. Sobre la umbra, el perfil real está **deformado** por desdoblamiento Zeeman (campo
   magnético fuerte) y por la depresión de Wilson (capa τ=1 más profunda).
3. Al invertir un perfil deformado con una tabla quieta, se produce una **velocidad
   espuria**, que se **amplifica** por la no-linealidad de la inversión.
4. La firma temporal de ~24 h entra porque toda la operación se modula por OBS_VR(t),
   la velocidad radial orbital SDO–Sol.

**Resultado esperado que valida la hipótesis:** un sesgo de perfil que en velocidad real
vale ~1 m/s debe producir un artefacto de **decenas a centenas de m/s** tras pasar por la
inversión. Si el código reproduce ese factor de amplificación de ~2-3 órdenes de
magnitud, la hipótesis queda demostrada. Si sale ~1 m/s, la hipótesis es falsa y hay que
reportarlo honestamente.

**Importante:** NO uses ninguna tabla descargada de JSOC en esta versión. Todo se calcula
sintéticamente desde las ecuaciones de Couvidat. La tabla real de JSOC
(`hmi.lookup_corrected_expanded`) es una validación posterior opcional, fuera de alcance
aquí.

---

## 1. Constantes físicas y del instrumento (de Couvidat 2011)

Define estas constantes exactas en un módulo de configuración:

```
LAMBDA0      = 6173.3433          # Å, centro de la línea Fe I en reposo
DVDL         = 48562.4            # m/s por Å  (= c / LAMBDA0, con c en m/s)
                                  # Couvidat: 299792458.0 / 6173.3433
N_FILTERS    = 6                  # HMI muestrea la línea en 6 puntos
DLAMBDA      = 0.0688             # Å, separación entre filtros (68.8 mÅ)
T_PERIOD     = 6 * DLAMBDA        # Å = 0.4128 Å, "período" del perfil (412.8 mÅ)
LANDE_G      = 2.5                # factor de Landé efectivo de la línea
```

Posiciones de muestreo de los 6 filtros (índice j de 0 a 5, en orden de longitud de onda
**decreciente**, como especifica Couvidat: "I0 to I5 in order of decreasing wavelength").
Los filtros están centrados sobre la línea en la rejilla:

```
# offset del filtro j respecto al centro, en Å
lambda_j = LAMBDA0 + (2.5 - j) * DLAMBDA   para j = 0..5
# esto da: j=0 -> +2.5*dl, j=1 -> +1.5*dl, ..., j=5 -> -2.5*dl
```

---

## 2. Módulo 1 — Perfil de línea Fe I sintético

### 2.1 Perfil gaussiano (Sol quieto, Ec. 5 de Couvidat)

```
I(lambda) = Ic - Id * exp( -(lambda - lambda0)^2 / sigma^2 )
```

Parámetros de Sol quieto (usar como defaults; son valores estándar razonables para
Fe I 6173, ajustables):
```
Ic_quiet    = 1.0        # continuo normalizado
Id_quiet    = 0.55       # profundidad de línea (~55% del continuo para Fe I 6173 quieta)
sigma_quiet = 0.060      # Å, relacionado al ancho; FWHM = 2*sqrt(ln2)*sigma ~ 0.10 Å
lambda0     = LAMBDA0    # sin corrimiento en reposo
```

Implementa `gaussian_profile(lam, Ic, Id, sigma, lambda0)` que devuelve I(lam) para un
array `lam`.

### 2.2 Perfil umbral deformado

La deformación tiene DOS componentes. Implementa ambas de forma modular para poder
activarlas por separado y así aislar el efecto de cada una.

**(a) Depresión de Wilson → cambio de parámetros del perfil.**
La capa τ=1 más profunda en la umbra hace la línea más somera y ligeramente más angosta,
y baja el continuo. Modela esto como una modificación de (Ic, Id, sigma) parametrizada por
la profundidad de Wilson `z_W` en km:

```
# escalado lineal fenomenológico, calibrable:
Ic_umbra    = Ic_quiet * (1 - k_c * z_W)      # continuo más bajo (umbra es oscura)
Id_umbra    = Id_quiet * (1 - k_d * z_W)      # línea más somera
sigma_umbra = sigma_quiet * (1 - k_s * z_W)   # línea algo más angosta
```
Empieza con `k_c, k_d, k_s` pequeños (p.ej. tales que a z_W=600 km el continuo baje ~20-30%
y la profundidad ~15%). DEJA ESTOS COMO PARÁMETROS EXPUESTOS: el barrido sobre z_W es el
experimento central. **No hardcodees; ponlos en la config.**

**(b) Desdoblamiento Zeeman → línea partida en componentes.**
Un campo magnético longitudinal B parte la línea. Para el modo LCP/RCP de HMI, cada
polarización ve la línea desplazada por el corrimiento Zeeman ±Δλ_B:

```
# corrimiento Zeeman en Å para campo B (Gauss), longitudinal
# delta_lambda_B = 4.67e-13 * LANDE_G * lambda0^2[Å] * B[G]   (fórmula estándar)
# equivalente a la constante Km inversa de Couvidat Ec. 15
delta_lambda_B = 4.67e-13 * LANDE_G * (LAMBDA0**2) * B_gauss
```
Verifica el orden de magnitud: para B=2500 G debe dar del orden de ~0.05-0.07 Å (comparable
a sigma), que es justo el régimen donde el muestreo de 6 puntos se degrada.

El perfil que ve la polarización LCP es la línea centrada en `lambda0 + delta_lambda_B`; la
RCP en `lambda0 - delta_lambda_B`. (En el caso quieto B=0, ambas coinciden y todo se cancela
— eso debe caer del código automáticamente, es un buen test.)

Parámetro expuesto: `B_gauss` (default 2500 G).

Implementa `umbral_profile(lam, z_W, B_gauss, polarization)` que combina (a) y (b) y
devuelve el perfil deformado para la polarización 'LCP' o 'RCP'.

---

## 3. Módulo 2 — Muestreo de 6 puntos y coeficientes de Fourier

### 3.1 Muestreo

Dado un perfil `profile_func(lam)` y una velocidad Doppler `v` (m/s) que desplaza toda la
línea, calcula las 6 intensidades:

```
# el desplazamiento Doppler en Å correspondiente a v:
d_lambda_v = v / DVDL
# intensidad en cada filtro j:
I_j = profile_func(lambda_j - d_lambda_v)   para j = 0..5
```
(Restar el corrimiento porque una velocidad positiva —alejándose— corre la línea al rojo.)

Implementa `sample_six_points(profile_func, v)` que devuelve el array `[I0,...,I5]`.

### 3.2 Coeficientes de Fourier discretos (Ec. 10-13 de Couvidat)

```
a1 = (2/6) * sum_{j=0}^{5} I_j * cos(2*pi*(2.5 - j)/6)
b1 = (2/6) * sum_{j=0}^{5} I_j * sin(2*pi*(2.5 - j)/6)
a2 = (2/6) * sum_{j=0}^{5} I_j * cos(4*pi*(2.5 - j)/6)
b2 = (2/6) * sum_{j=0}^{5} I_j * sin(4*pi*(2.5 - j)/6)
```

### 3.3 Velocidad desde el primer coeficiente (Ec. 6)

```
v_measured = DVDL * (T_PERIOD / (2*pi)) * atan2(b1, a1)
```
**Usa `atan2(b1, a1)`, no `atan(b1/a1)`**, para manejar correctamente los cuadrantes.

Implementa `velocity_from_profile(profile_func, v_true)` que encadena:
muestrea en v_true → Fourier → devuelve v_measured. Para el perfil quieto sin deformación,
`velocity_from_profile` debe ser casi la identidad (v_measured ≈ v_true) en el rango de
velocidades quietas — ese es el **test de validación obligatorio del Módulo 2**.

---

## 4. Módulo 3 — La look-up table sintética (la inversa quieta)

Este es el corazón del mecanismo. La tabla del pipeline invierte `v_apparent → v_true`
suponiendo el **perfil quieto**. Constrúyela exactamente así:

```
# 1. define una rejilla de velocidades "verdaderas" de entrada, como Couvidat (821 puntos)
v_true_grid = linspace(-7000, 7000, 821)   # m/s, rango amplio tipo pipeline

# 2. para cada v_true, mide la v_apparent que el algoritmo de Fourier devuelve
#    usando el PERFIL QUIETO:
v_apparent_grid = [velocity_from_profile(quiet_profile, v) for v in v_true_grid]

# 3. la look-up table es la función inversa: dado v_apparent, devuelve v_true
#    implementa por interpolación (scipy.interpolate.interp1d o np.interp con inversión)
lookup_table = interp1d(v_apparent_grid, v_true_grid, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
```

Nota conceptual clave para el agente: `v_apparent_grid` NO es igual a `v_true_grid` porque
el muestreo de 6 puntos + Fourier discreto introduce no-linealidad (Couvidat: "the discrete
approximations to the Fourier coefficients are far from perfect"). Esa no-linealidad es la
que amplifica el artefacto. La tabla la corrige **para el perfil quieto**, pero no para el
deformado.

Guarda también `dLookup/dv_apparent` (derivada numérica de la tabla) — la pendiente local
es el factor de amplificación y hay que graficarla.

---

## 5. Módulo 4 — El experimento del artefacto (núcleo del paper)

Ahora se junta todo. El protocolo por cada punto del barrido:

```
def measure_artifact(z_W, B_gauss, v_true=0.0, obs_vr=0.0):
    # velocidad total que desplaza el perfil: la real del plasma + la orbital
    v_shift = v_true + obs_vr

    # perfil deformado, por polarización
    prof_LCP = lambda lam: umbral_profile(lam, z_W, B_gauss, 'LCP')
    prof_RCP = lambda lam: umbral_profile(lam, z_W, B_gauss, 'RCP')

    # el pipeline mide v_apparent de cada polarización sobre el perfil DEFORMADO
    vapp_LCP = velocity_from_profile(prof_LCP, v_shift)
    vapp_RCP = velocity_from_profile(prof_RCP, v_shift)

    # y las corrige con la tabla QUIETA (aquí está el error: tabla equivocada)
    vcorr_LCP = lookup_table(vapp_LCP)
    vcorr_RCP = lookup_table(vapp_RCP)

    # observables finales (Couvidat Ec. 14 y 15)
    V_doppler = 0.5 * (vcorr_LCP + vcorr_RCP)
    B_los     = (vcorr_LCP - vcorr_RCP) * Km   # Km de Couvidat Ec.15

    # el artefacto en velocidad es V_doppler menos la velocidad real de entrada
    artifact_velocity = V_doppler - v_true
    return artifact_velocity, V_doppler, B_los
```

Donde `Km = 1.0/(2.0*4.67e-5*0.000061733433*2.5*299792458.0)` (constante literal de
Couvidat Ec. 15).

---

## 6. Los tres experimentos a ejecutar y graficar

### Experimento A — validación (obligatorio, va primero)
Con `z_W=0, B_gauss=0`: confirma que `artifact_velocity ≈ 0` para todo `v_true` y todo
`obs_vr`. Si esto no da ~0, hay un bug; PARA y arréglalo antes de seguir. Grafica
`v_measured` vs `v_true` para el perfil quieto y muestra que es la identidad.

### Experimento B — magnitud del artefacto vs profundidad de Wilson
Barre `z_W` de 0 a 800 km (a `B_gauss=2500 G`, `v_true=0`, `obs_vr=0`). Grafica
`artifact_velocity` vs `z_W`. **Esta es la curva forward central.** Reporta la pendiente
(m/s por km). Compara la magnitud con los ~350 m/s observados: ¿el modelo alcanza esa
escala para z_W realistas (~600 km)?

También grafica el factor de amplificación: compara `artifact_velocity` (que puede ser
centenas de m/s) contra el sesgo de velocidad "geométrico" ingenuo de ~1 m/s. El cociente
es el resultado titular.

### Experimento C — firma temporal de 24 h vía OBS_VR
Fija `z_W=600 km, B_gauss=2500 G, v_true=0`. Genera una OBS_VR(t) sinusoidal representativa
de la órbita de SDO:
```
obs_vr(t) = A_orb * sin(2*pi*t / 24h)    con A_orb ~ 3000 m/s (amplitud orbital típica)
```
Barre t sobre 48 h y grafica `artifact_velocity(t)`. **Debe mostrar una oscilación de ~24 h**
cuya amplitud es tu predicción del residual. Compara esa amplitud con los ~350 m/s.
Reporta también el `B_los(t)` para verificar que el artefacto también contamina el
magnetograma (mecanismo de no-cancelación LCP/RCP).

---

## 7. Estructura de archivos que debe crear el agente

```
hmi_artifact_sim/
├── config.py           # todas las constantes de §1 y parámetros de perfil §2
├── profiles.py         # Módulo 1: gaussian_profile, umbral_profile
├── pipeline.py         # Módulos 2-3: sample_six_points, fourier coeffs,
│                       #   velocity_from_profile, build_lookup_table
├── artifact.py         # Módulo 4: measure_artifact
├── experiments.py      # los 3 experimentos §6, genera todas las figuras
├── figures/            # PNGs de salida
└── README.md           # cómo correr, qué significan los parámetros, resultados
```

Dependencias: solo `numpy`, `scipy`, `matplotlib`. Nada de sunpy/drms/astropy en esta
versión sintética.

---

## 8. Criterios de aceptación (cómo saber que está bien hecho)

1. Experimento A da |artifact| < ~1 m/s en todo el rango (validación de que el pipeline
   sintético es fiel y no introduce artefactos espurios por sí mismo).
2. La curva del Experimento B es monótona creciente en z_W y su pendiente/magnitud está
   documentada. Sea cual sea el resultado (llega o no a 350 m/s), se reporta con honestidad.
3. El Experimento C produce una oscilación con período exacto de 24 h (heredado de OBS_VR)
   y amplitud reportada.
4. Cada parámetro fenomenológico (k_c, k_d, k_s, B_gauss, A_orb) está en `config.py`,
   documentado, y NO hardcodeado dentro de funciones.
5. El README explica claramente cuáles resultados son predicciones físicas robustas
   (la amplificación por no-linealidad de la inversión, que es geometría del muestreo) y
   cuáles dependen de la calibración fenomenológica de la deformación del perfil
   (k_c, k_d, k_s), que es la parte a refinar después con síntesis Milne-Eddington o con
   la tabla real de JSOC.

---

## 9. Advertencias sobre errores probables

- **atan vs atan2:** usar atan2 siempre. atan(b1/a1) rompe en los cruces de cuadrante.
- **Signo del corrimiento Doppler:** verifica con el Experimento A. Si la identidad sale
  invertida (v_measured = -v_true), tienes un signo cambiado en `d_lambda_v`.
- **Rango de la rejilla de la lookup:** si `v_apparent` de un perfil deformado cae fuera del
  rango de `v_apparent_grid`, la interpolación extrapola. Documenta cuándo pasa; es
  físicamente significativo (Couvidat dice que en active regions la corrección "becomes a
  basic extrapolation").
- **No confundir el artefacto con ruido numérico:** el paso de la rejilla (821 puntos sobre
  14000 m/s ≈ 17 m/s/punto) pone un piso de resolución. Si el artefacto que buscas es menor
  que eso, refina la rejilla. Los efectos de centenas de m/s están muy por encima, así que
  no debería ser problema, pero verifícalo.
- La deformación de Wilson (§2.2a) es fenomenológica y calibrable; NO la presentes como
  derivada de primeros principios. La parte robusta del argumento es la amplificación por la
  no-linealidad de la inversión de 6 puntos, que sí es geometría pura del instrumento.
```
