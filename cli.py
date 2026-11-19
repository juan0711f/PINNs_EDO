import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="" \
    "En este script, resolvemos un problema de control óptimo para un modelo de crecimiento tumoral de ecuaciones diferenciales ordinarias (EDO) " \
    "utilizando redes neuronales basadas en la física (PINN). El script nos permite entrenar redes PINN por separado para las ecuaciones de estado " \
    "y adjuntas, y optimizar las variables de control de manera iterativa mediante un método de descenso de gradiente con Armijo.")

    # ========================================================================================================================
    # ============================================ Parámetros de PINNs =======================================================
    # ========================================================================================================================
    pinn_group = parser.add_argument_group("PINN Parameters")

    # Sistema de estados (Parámetros de PINNs)
    pinn_group.add_argument("--state-n-collocation",    type=int,   default=250,    help="Puntos de colocalización por subintervalo para los estados.")
    pinn_group.add_argument("--state-hidden-width",     type=int,   default=100,    help="Número de neuronas para cada capa oculta.")
    pinn_group.add_argument("--state-hidden-depth",     type=int,   default=4,      help="Número de capas ocultas para cada subintervalo de los estados.")
    pinn_group.add_argument("--state-subintervals",     type=int,   default=1,      help="Número de subintervalos de entrenamiento para los estados.")
    pinn_group.add_argument("--state-lr",               type=float, default=1e-3,   help="Tasa de aprendizaje del optimizador para la red PINN de estados.")
    pinn_group.add_argument("--state-SCS-weight",       type=float, default=10.0,   help="Peso del término de penalización del SCS para los estados.")

    pinn_group.add_argument("--state-epochs",           type=int,   default=40000,   help="Épocas por subintervalo para los estados.")

    pinn_group.add_argument("--state-pinn-mode",        choices=["HCS_PINNs", "SCS_PINNs"], default="SCS_PINNs", help="Modo de restricciones para los estados: restricciones estrictas (HCS) o restricciones suaves (SCS).")

    # Sistema adjunto (Parámetros de PINNs)
    pinn_group.add_argument("--adj-n-collocation",    type=int,   default=250,    help="Colocación de puntos por subintervalo para adjuntos.")
    pinn_group.add_argument("--adj-hidden-width",     type=int,   default=150,    help="Número de neuronas para cada capa oculta.")
    pinn_group.add_argument("--adj-hidden-depth",     type=int,   default=5,      help="Número de capas ocultas para cada subintervalo en el caso de las funciones adjuntas.")
    pinn_group.add_argument("--adj-subintervals",     type=int,   default=1,      help="Número de subintervalos de entrenamiento para los adjuntos.")
    pinn_group.add_argument("--adj-lr",               type=float, default=1e-3,   help="Tasa de aprendizaje del optimizador para PINN de adjuntos.")
    pinn_group.add_argument("--adj-SCS-weight",       type=float, default=10.0,   help="Peso del término de penalización SCS para los adjuntos.")

    pinn_group.add_argument("--adj-epochs",           type=int,   default=7000,   help="Epochs per subinterval for adjoints.")
    

    pinn_group.add_argument("--adj-pinn-mode",        choices=["HCS_PINNs", "SCS_PINNs"], default="SCS_PINNs", help="Modo de restricciones para las adjuntas: restricciones estrictas (HCS) o restricciones suaves (SCS).")

    # ========================================================================================================================
    # =========================================== Parámetros del modelo ======================================================
    # ========================================================================================================================
    model_group = parser.add_argument_group("Parámetros del modelo")

    # Ecuación tumoral (parámetros)
    model_group.add_argument("--alpha", type=float, default=4.0, help="Capacidad máxima del tumor.")
    model_group.add_argument("--psi",   type=float, default=2.0, help="Intensidad del efecto del control C(t) sobre el tumor.")
    model_group.add_argument("--pmax",  type=float, default=3.5, help="Tasa máxima de crecimiento del tumor asociado a Rho(sigma).")
    model_group.add_argument("--b",     type=float, default=1.0, help="Constante que interviene en la función rho(sigma).")

    # Ecuación de Sigma (parámetros)
    model_group.add_argument("--beta",  type=float, default=1.0,  help="Capacidad máxima de oxígeno.")
    model_group.add_argument("--aox",   type=float, default=0.15, help="Constante que interviene en la cinética de Michaelis-Menten.")
    model_group.add_argument("--kox",   type=float, default=0.1,  help="Constante que interviene en la cinética de Michaelis-Menten.")
    model_group.add_argument("--per",   type=float, default=0.1,  help="Permeabilidad vascular.")
    model_group.add_argument("--sv",    type=float, default=1.0,  help="Densidad vascular.")
    model_group.add_argument("--sc",    type=float, default=0.4,  help="Tasa de suministro de nutrientes por parte del tumor (angiogénesis).")

    # ========================================================================================================================
    # =================================== Parámetros del problema de optimización ============================================
    # ========================================================================================================================
    opti_group = parser.add_argument_group("Parámetros del problema de optimización")
    
    # Ponderaciones de la función de costo
    opti_group.add_argument("--k1", type=float, default=1.0, help="Peso de la variable de estado u en la función de costo.")
    opti_group.add_argument("--k2", type=float, default=1.0, help="Peso de la variable de estado sigma en el funcional de costo.")
    opti_group.add_argument("--k3", type=float, default=1.0, help="Peso del control C en la función de costo.")
    opti_group.add_argument("--k4", type=float, default=1.0, help="Peso del control S en la función de costo.")
    opti_group.add_argument("--l1", type=float, default=1.0, help="Peso del control S en la función de costo.")
    opti_group.add_argument("--l2", type=float, default=1.0, help="Peso cuadrático terminal para sigma(T)-sigma_dT.")
     
    # Valores deseados en la función de costo
    opti_group.add_argument("--sigma-d",    type=float, default=1.0, help="Parámetro sigma deseado en el rango [0, T] del modelo.")
    opti_group.add_argument("--sigma-dt",   type=float, default=1.0, help="Parámetro sigma deseado en el momento final T.")

    # Valores relacionados con el control de C
    opti_group.add_argument("--ct", type=float, default=0.75, help="Target integral value for C.")
    opti_group.add_argument("--controls", "--control", dest="control", choices=["start", "constant", "sinusoidal", "end"], default="constant",
        help="Perfil de control inicial: «constant» (uniforme), «sinusoidal» (oscilatorio), «start» (alto al principio) o «end» (alto al final).")

    # Valores para el descenso por gradiente con Armijo
    opti_group.add_argument("--max-iter-optimo",        type=int,   default=800,    help="Número máximo de iteraciones en el problema de control.")
    opti_group.add_argument("--initial-step-size",      type=float, default=1,      help="Primer paso en una reducción tipo Armijo.")
    opti_group.add_argument("--min-step-size",          type=float, default=1e-4,   help="Límite inferior del tamaño del paso en la reducción iterativa.")
    opti_group.add_argument("--step-reduction-factor",  type=float, default=0.5,    help="Factor multiplicativo para la reducción de pasos.")
    opti_group.add_argument("--max-step-reductions",    type=int,   default=2,      help="Número máximo de reducciones de paso por iteración.")
    opti_group.add_argument("--armijo-constant-c",      type=float, default=1e-3,   help="Constante de Armijo.")
    opti_group.add_argument("--stop-j-tol",             type=float, default=1e-6,   help="Tolerancia de convergencia para |J_k - J_{k-1}|.")

    # ========================================================================================================================
    # ===================================== Configuración para guardar resultados ============================================
    # ========================================================================================================================
    output_group = parser.add_argument_group("Result saving settings")
    output_group.add_argument("--simulation",           type=str, default="simulation", help="Nombre de la simulación que se utilizará al crear la carpeta de resultados.")
    output_group.add_argument("--save-every-n-epochs",  type=int, default=500,          help="Guarda el estado de los controles cada N épocas durante el entrenamiento de PINN.")

    # ========================================================================================================================
    # ================================= Parámetros de discretización temporal ==========================================
    # ========================================================================================================================
    init_time_group = parser.add_argument_group("Estado inicial y parámetros de discretización temporal")

    init_time_group.add_argument("--u0-inicial",    type=float, default=0.6, help="Condición inicial del estado u(0).")
    init_time_group.add_argument("--s0-inicial",    type=float, default=1.0, help="Condición inicial del estado sigma(0).")
    init_time_group.add_argument("--T",             type=float, default=2.0, help="Tiempo final.")
    init_time_group.add_argument("--n-iter",        type=int,   default=250, help="Número de intervalos de discretización temporal.")

    args = parser.parse_args()
    return args
