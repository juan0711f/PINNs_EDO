# Control óptimo de terapias en un modelo EDO de glioblastoma con PINNs secuenciales

[![arXiv](https://img.shields.io/badge/arXiv-2608.18974-b31b1b.svg)](https://arxiv.org/abs/2608.18974)
[![Python](https://img.shields.io/badge/Python-3-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Licencia: Apache 2.0](https://img.shields.io/badge/licencia-Apache%202.0-blue.svg)](LICENSE)

Código de las simulaciones numéricas del artículo [*Modeling of an ODE-constrained optimization problem describing tumor dynamics, and numerical approximation via sequential physics-informed neural networks*](https://arxiv.org/abs/2608.18974), de J. J. Forero-Hernández y É. J. Villamizar-Roa. Está escrito en Python: las PINNs se entrenan con PyTorch y la referencia clásica se calcula con SciPy. Incluye un monitor en Matplotlib para revisar las corridas.

## Modelo y método

El modelo es un sistema de EDO para la densidad de células tumorales $u$ y la concentración de oxígeno $\sigma$ en $[0,T]$:

$$
u' = \rho(\sigma)u(\alpha-u) - \kappa c(t)u\sigma, \qquad u(0)=u_0,
$$

$$
\sigma' = P_{er}S_v(\beta-\sigma) - \frac{A_{ox}u\sigma}{k_{ox}+\sigma} + (1-s(t))S_c u, \qquad \sigma(0)=\sigma_0,
$$

con $\rho(\sigma)=\frac{\hat\rho}{\alpha}\left(\frac{\sigma}{\beta}+b\left(1-\frac{\sigma}{\beta}\right)\right)$. Los controles son la quimioterapia $c$ y la terapia antiangiogénica $s$. Toman valores en $[0,1]$ y la dosis total de quimioterapia cumple $\int_0^T c(t)dt \le c_{\max}$. El funcional penaliza $u^2$ y $(\sigma-\sigma_d)^2$ durante el tratamiento y en $t=T$, además de la cantidad de cada terapia.

Los controles se calculan con descenso de gradiente proyectado y búsqueda de Armijo. En cada iteración se aproximan con PINNs el estado, hacia adelante, y el adjunto, hacia atrás desde $T$, y con ellos se calcula el gradiente del funcional reducido. Los controles de prueba se truncan a $[0,1]$ y $c$ se proyecta sobre la restricción integral, buscando el multiplicador por bisección. Evaluar $J$ en cada control de prueba exige entrenar otra PINN para el estado. El algoritmo para cuando $J$ cambia menos de $10^{-6}$ en tres iteraciones seguidas, o al llegar a `--max-iter-optimo`.

Las PINNs son secuenciales en tiempo: el intervalo se divide en $M$ subintervalos y en cada uno se entrena una red distinta, totalmente conectada, con activación tanh y optimizador Adam. El valor de cada red al final de su subintervalo es el dato inicial de la siguiente; en el adjunto el recorrido va de $T$ a $0$. En la formulación SCS (*soft-constrained*) ese dato entra en la pérdida con un peso $\omega_{init}$. En la HCS (*hard-constrained*) la salida de la red $\mathcal{N}_\theta$ se escribe como $y(\tau_{m-1})+(t-\tau_{m-1})\mathcal{N}_\theta(t)$, así que el dato se cumple exactamente. Con $M=1$, SCS es la PINN usual. En cada subintervalo se conservan los pesos de menor pérdida, y las redes se entrenan desde cero en cada iteración.

En paralelo, el programa ejecuta el mismo algoritmo desde el mismo control inicial, pero con el estado y el adjunto resueltos por `scipy.integrate.solve_ivp` (Radau, `rtol=1e-8`, `atol=1e-12`). Esa es la referencia clásica con la que se comparan las PINNs. Cada rama se detiene por su cuenta y conserva desde ahí sus últimos valores; la corrida termina cuando paran las dos. Los controles se representan en una malla uniforme de `--n-iter` intervalos, donde también se calculan las integrales de $J$ con la regla del trapecio.

## Requisitos

- Python 3 con NumPy, SciPy y PyTorch. El entrenamiento corre en CPU.
- Matplotlib y PyQt6, solo para el monitor. Sin Qt, Matplotlib no puede crear las pestañas y el monitor no abre.

Se instalan con `python3 -m pip install numpy scipy torch matplotlib PyQt6`.

## Uso

Los comandos se ejecutan desde esta carpeta, porque la ruta a `results/` es relativa.

```bash
# Ayuda
python3 main.py --help

# Control óptimo con HCS y 4 subintervalos, prueba corta
python3 main.py --simulation prueba --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 4 --adj-subintervals 4 --state-epochs 500 --adj-epochs 500 --max-iter-optimo 3

# Ver la corrida
python3 monitor.py prueba
```

Cada corrida se guarda en `results/<simulación>/`. Si esa carpeta ya existe, el programa crea otra con la fecha y la hora en el nombre, así que nunca sobrescribe una corrida. Por cada iteración, la terminal muestra $J$ con PINNs y con la referencia clásica, su diferencia, $\int_0^T c$ y el paso de Armijo.

### Opciones

| Opción | Descripción |
| --- | --- |
| `--simulation <nombre>` | Nombre de la carpeta en `results/`. Por defecto `simulation`. |
| `--state-pinn-mode`, `--adj-pinn-mode` | Formulación del estado y del adjunto: `HCS_PINNs` o `SCS_PINNs`. Por defecto `SCS_PINNs`. |
| `--state-subintervals`, `--adj-subintervals` | Número de subintervalos $M$. Por defecto 1. |
| `--state-epochs`, `--adj-epochs` | Épocas por subintervalo. Por defecto 40000 y 7000. |
| `--state-hidden-depth`, `--adj-hidden-depth` | Capas ocultas. Por defecto 4 y 5. |
| `--state-hidden-width`, `--adj-hidden-width` | Neuronas por capa oculta. Por defecto 100 y 150. |
| `--state-n-collocation`, `--adj-n-collocation` | Puntos de colocación equiespaciados por subintervalo. Por defecto 250; si se piden menos de 32, se usan 32. |
| `--state-lr`, `--adj-lr` | Tasa de aprendizaje de Adam. Por defecto `1e-3`. |
| `--state-SCS-weight`, `--adj-SCS-weight` | Peso $\omega_{init}$ del dato inicial en SCS. Por defecto 10. |
| `--controls <perfil>` | Control inicial. `constant`: $c=c_{\max}/T$ y $s=0$. `start` y `end`: $c=1$ en un tramo de largo $c_{\max}$ al principio o al final del intervalo, y $s=1$ en ese tramo. `sinusoidal`: dos senos desfasados entre 0 y 1, con $c$ proyectado. Por defecto `constant`. |
| `--max-iter-optimo <int>` | Máximo de iteraciones de control. Por defecto 800. |
| `--save-every-n-epochs <int>` | Cada cuántas épocas se guarda un checkpoint de las redes. Por defecto 500. |
| `--T <real>`, `--n-iter <int>` | Tiempo final e intervalos de la malla temporal. Por defecto 2 y 250. |
| `--u0-inicial`, `--s0-inicial` | Datos iniciales $u_0$ y $\sigma_0$. Por defecto 0.6 y 1. |

Los parámetros del modelo, del funcional y de Armijo también se cambian por línea de comandos. Sus valores por defecto, en `cli.py`, son los del artículo, salvo $l_1$ y $l_2$: valen 1 y en el artículo se usó 10.

| Opción | Parámetro | Por defecto |
| --- | --- | --- |
| `--alpha`, `--beta` | $\alpha$, $\beta$ | 4, 1 |
| `--pmax`, `--b` | $\hat\rho$, $b$ | 3.5, 1 |
| `--psi` | $\kappa$ | 2 |
| `--aox`, `--kox` | $A_{ox}$, $k_{ox}$ | 0.15, 0.1 |
| `--per`, `--sv`, `--sc` | $P_{er}$, $S_v$, $S_c$ | 0.1, 1, 0.4 |
| `--k1` a `--k4` | $k_1$ a $k_4$ | 1 |
| `--l1`, `--l2` | $l_1$, $l_2$ | 1 |
| `--sigma-d`, `--sigma-dt` | $\sigma_d$, $\sigma_{dT}$ | 1, 1 |
| `--ct` | $c_{\max}$ | 0.75 |
| `--initial-step-size`, `--min-step-size` | paso inicial y paso mínimo de Armijo | 1, 1e-4 |
| `--step-reduction-factor`, `--max-step-reductions` | factor de reducción del paso y máximo de reducciones | 0.5, 2 |
| `--armijo-constant-c` | constante de Armijo $\eta$ | 1e-3 |
| `--stop-j-tol` | tolerancia del criterio de parada | 1e-6 |

Dos nombres pueden confundir: `--psi` es el $\kappa$ del artículo y `--ct` es $c_{\max}$, la cota de $\int_0^T c(t)dt$. Además, en el código los controles se llaman `C` y `S`, y los adjuntos $p_1$ y $p_2$ se llaman `w` y `z`.

### Salidas

```text
results/<simulación>/
├── run_info.json              redes, malla temporal e iteraciones de parada
├── control_history.npz        J, J1 a J6, controles, estados y adjuntos de cada iteración, con PINNs y clásicos
├── state_training/
│   └── iter_<k>.npz           pérdida de la PINN del estado en cada época
├── adjoint_training/
│   └── iter_<k>.npz           lo mismo para el adjunto
├── state_checkpoints/
│   └── iter_<k>/
│       └── epoch_<e>.npz      predicción de la red en la época e y solución clásica
└── adjoint_checkpoints/
    └── iter_<k>/
        └── epoch_<e>.npz
```

`J1` a `J6` son los términos del funcional en este orden: $u^2$, $(\sigma-\sigma_d)^2$, $u(T)^2$, $(\sigma(T)-\sigma_{dT})^2$, $c$ y $s$. Cada fila de la pérdida tiene cinco columnas: los residuos de las dos ecuaciones, los dos términos del dato inicial (nulos en HCS) y el total. Las épocas se cuentan seguidas a lo largo de los subintervalos, y se guarda checkpoint en las épocas 0 y 1, cada `--save-every-n-epochs` y en la de menor pérdida.

`control_history.npz` se reescribe al terminar cada iteración, así que una corrida interrumpida también se puede abrir. `run_info.json` no guarda los parámetros del modelo ni la formulación HCS o SCS, por eso conviene ponerlos en el nombre de la simulación.

### Monitor

```bash
python3 monitor.py exp1_HCS_4sub    # también acepta la ruta a la carpeta
```

Tiene tres pestañas en la barra de herramientas. *Control* muestra $J$ con PINNs y clásico, su diferencia y, para la iteración elegida, los controles $c$ y $s$ y sus errores. *State PINN* y *Adj PINN* comparan $u,\sigma$ y $w,z$ con la solución clásica y muestran la pérdida del entrenamiento con sus componentes. La iteración se elige con un clic sobre la curva de $J$ o escribiendo su número, y `Play` las recorre en orden.

## Reproducir los resultados del artículo

```bash
# Experimento 1: SCS y HCS, con 4 subintervalos y con 1
python3 main.py --simulation exp1_SCS_4sub --l1 10 --l2 10 --state-pinn-mode SCS_PINNs --adj-pinn-mode SCS_PINNs --state-subintervals 4 --state-epochs 10000 --adj-subintervals 4 --adj-epochs 10000
python3 main.py --simulation exp1_SCS_1sub --l1 10 --l2 10 --state-pinn-mode SCS_PINNs --adj-pinn-mode SCS_PINNs --state-subintervals 1 --state-epochs 40000 --adj-subintervals 1 --adj-epochs 40000
python3 main.py --simulation exp1_HCS_4sub --l1 10 --l2 10 --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 4 --state-epochs 10000 --adj-subintervals 4 --adj-epochs 10000
python3 main.py --simulation exp1_HCS_1sub --l1 10 --l2 10 --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 1 --state-epochs 40000 --adj-subintervals 1 --adj-epochs 40000

# Experimento 2: HCS con 4 subintervalos desde otros dos controles iniciales (el constante es exp1_HCS_4sub)
python3 main.py --simulation exp2_HCS_4sub_control2_start --controls start --l1 10 --l2 10 --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 4 --state-epochs 10000 --adj-subintervals 4 --adj-epochs 10000
python3 main.py --simulation exp2_HCS_4sub_control3_sinusoidal --controls sinusoidal --l1 10 --l2 10 --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 4 --state-epochs 10000 --adj-subintervals 4 --adj-epochs 10000
```

| Carpeta en `results/` | Experimento | J final | Iteración de parada | Tiempo |
| --- | --- | ---: | ---: | ---: |
| `exp1_SCS_4sub` | 1, SCS con 4 subintervalos | 15.878714 | 24 | 2.3 h |
| `exp1_SCS_1sub` | 1, SCS con 1 subintervalo | 15.623673 | 56 | 5.4 h |
| `exp1_HCS_4sub` | 1, HCS con 4 subintervalos; 2, control constante | 15.450080 | 79 | 6.9 h |
| `exp1_HCS_1sub` | 1, HCS con 1 subintervalo | 15.561244 | 60 | 5.7 h |
| `exp2_HCS_4sub_control2_start` | 2, control concentrado al inicio | 15.407012 | 153 | 15.1 h |
| `exp2_HCS_4sub_control3_sinusoidal` | 2, control sinusoidal | 15.385723 | 98 | 11.9 h |

$J$ y la iteración de parada son los de las PINNs. La referencia clásica da $J=15.350898$ en la iteración 118 con el control inicial constante, $15.364730$ en la 162 con el concentrado al inicio y $15.349983$ en la 116 con el sinusoidal. Los tiempos son de un MacBook Air M4 con 16 GB e incluyen la rama clásica. La salida completa de las seis corridas está en `results/`; si repites una, el programa crea otra carpeta en lugar de sobrescribirla.

Las carpetas se renombraron después de las corridas: `run_info.json` conserva el nombre con el que se lanzó cada una y `nombre_anterior.txt` indica la correspondencia. El programa fija la semilla `98765` y trabaja en `float64`. Con Python 3.14.7, PyTorch 2.13.0 y NumPy 2.5.2, las dos primeras iteraciones de `exp1_HCS_4sub` y `exp1_SCS_4sub` salen idénticas a las guardadas.

## Estructura

```text
.
├── main.py                         programa principal
├── cli.py                          opciones de línea de comandos
├── params.py                       validación de opciones y controles iniciales
├── control_algorithm.py            iteraciones de las ramas PINN y clásica, y criterio de parada
├── pinn_solver.py                  PINNs secuenciales del estado y del adjunto
├── classic_solver.py               estado y adjunto con solve_ivp (Radau)
├── save_simulations.py             archivos de salida
├── tool/
│   ├── pinn_tools.py               semilla, red, formulaciones SCS y HCS, derivada e interpolación
│   └── control_algorithm_tools.py  J por la regla del trapecio, gradiente, proyección de c y Armijo
├── monitor.py                      visor de resultados
├── console                         salida de la terminal de las corridas del Experimento 1
└── results/                        corridas del artículo
```

## Cita

Si usas este código, cita el artículo:

```bibtex
@misc{ForeroHernandez2026PINNs,
  title         = {Modeling of an {ODE}-constrained optimization problem describing tumor dynamics, and numerical approximation via sequential physics-informed neural networks},
  author        = {Forero-Hern{\'a}ndez, Juan J. and Villamizar-Roa, {\'E}lder J.},
  year          = {2026},
  eprint        = {2608.18974},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  doi           = {10.48550/arXiv.2608.18974},
  url           = {https://arxiv.org/abs/2608.18974}
}
```

## Licencia

Copyright 2026 Juan J. Forero-Hernández y Élder J. Villamizar-Roa.

El código se distribuye bajo la licencia Apache 2.0. El texto completo está en [LICENSE](LICENSE) y el aviso de copyright, en [NOTICE](NOTICE).
