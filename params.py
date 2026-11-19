import numpy as np

def validateCLI(value, condition, message):
    if not condition(value):
        raise ValueError(message)
    return value


def build_initial_controls(control_type, CT, T, nIter):
    n_points = nIter + 1
    dt = T / nIter

    if control_type == "constant":
        C0 = np.full(n_points, CT / T, dtype=float)
        S0 = np.zeros(n_points, dtype=float)
        return C0.tolist(), S0.tolist()

    if control_type == "sinusoidal":
        from tool.control_algorithm_tools import project_c_integral_constraint

        t_grid_local = np.linspace(0.0, T, n_points)
        C0 = 0.5 * (1.0 + np.sin(2.0 * np.pi * t_grid_local / T))
        S0 = 0.5 * (1.0 + np.sin(2.0 * np.pi * t_grid_local / T + 0.5 * np.pi))

        C0 = np.clip(C0, 0.0, 1.0)
        S0 = np.clip(S0, 0.0, 1.0)

        project_c_integral_constraint(C0)
        C0 = np.clip(C0, 0.0, 1.0)
        return C0.tolist(), S0.tolist()

    if control_type not in ["start", "end"]:
        raise ValueError("El control debe ser uno de los siguientes: 'start', 'constant', 'sinusoidal', o 'end'.")


    # Si control_type es "start" o "end", se deben construir de manera que se satisfaga la restricción integral.
    weights = np.full(n_points, dt, dtype=float)
    weights[0] = 0.5 * dt
    weights[-1] = 0.5 * dt

    C0 = np.zeros(n_points, dtype=float)
    remaining = float(CT)
    if control_type == "start":
        indices = range(n_points)
    else:
        indices = range(n_points - 1, -1, -1)

    for idx in indices:
        if remaining <= 0.0:
            break
        wi = weights[idx]
        if remaining >= wi:
            C0[idx] = 1.0
            remaining -= wi
        else:
            C0[idx] = remaining / wi
            remaining = 0.0

    if remaining > 1e-12:
        raise ValueError("CT es demasiado grande para los controles con el valor actual de T.")

    C0 = np.clip(C0, 0.0, 1.0)
    S0 = (C0 > 0.0).astype(float)
    return C0.tolist(), S0.tolist()

def apply_cli_args(args):
    global state_epochs, state_n_collocation, state_hidden_width, state_hidden_depth
    global state_subintervals, state_lr, state_pinn_mode
    global adj_epochs, adj_n_collocation, adj_hidden_width, adj_hidden_depth
    global adj_subintervals, adj_lr, adj_pinn_mode
    global max_iter_optimo, stop_j_tol
    global simulation, save_every_n_epochs

    global alpha, beta, psi
    global Aox, kox, Per, Sv, Sc
    global pmax, b
    global k1, k2, k3, k4, l1, l2
    global DesiredSigma, DesiredSigmaT
    global CT
    global T, nIter
    global u0_inicial, s0_inicial
    global initial_step_size, min_step_size
    global step_reduction_factor, max_step_reductions, armijo_constant_c
    global dt, t_grid
    global C0, S0
    global control
    global state_SCS_weight, adj_SCS_weight
    
    state_epochs =        validateCLI(args.state_epochs,        lambda v: v >= 1,  "state_epochs debe ser >= 1.")
    state_n_collocation = validateCLI(args.state_n_collocation, lambda v: v >= 1,  "state_n_collocation debe ser >= 1.")
    state_hidden_width =  validateCLI(args.state_hidden_width,  lambda v: v >= 1,  "state_hidden_width debe ser >= 1.")
    state_hidden_depth =  validateCLI(args.state_hidden_depth,  lambda v: v >= 1,  "state_hidden_depth debe ser >= 1.")
    state_subintervals =  validateCLI(args.state_subintervals,  lambda v: v >= 1,  "state_subintervals debe ser >= 1.")
    state_lr =            validateCLI(args.state_lr,            lambda v: v > 0.0, "state_lr debe ser mayor que 0.")
    state_pinn_mode =     validateCLI(args.state_pinn_mode,     lambda v: v in ["HCS_PINNs", "SCS_PINNs"], "state_pinn_mode debe ser 'HCS_PINNs' o 'SCS_PINNs'.")
    state_SCS_weight =    validateCLI(args.state_SCS_weight,    lambda v: v > 0.0, "state_SCS_weight debe ser > 0.")

    adj_epochs =        validateCLI(args.adj_epochs,        lambda v: v >= 1,  "adj_epochs debe ser >= 1.")
    adj_n_collocation = validateCLI(args.adj_n_collocation, lambda v: v >= 1,  "adj_n_collocation debe ser >= 1.")
    adj_hidden_width =  validateCLI(args.adj_hidden_width,  lambda v: v >= 1,  "adj_hidden_width debe ser >= 1.")
    adj_hidden_depth =  validateCLI(args.adj_hidden_depth,  lambda v: v >= 1,  "adj_hidden_depth debe ser >= 1.")
    adj_subintervals =  validateCLI(args.adj_subintervals,  lambda v: v >= 1,  "adj_subintervals debe ser >= 1.")
    adj_lr =            validateCLI(args.adj_lr,            lambda v: v > 0.0, "adj_lr debe ser > 0.")
    adj_pinn_mode =     validateCLI(args.adj_pinn_mode,     lambda v: v in ["HCS_PINNs", "SCS_PINNs"], "adj_pinn_mode debe ser 'HCS_PINNs' o 'SCS_PINNs'.")
    adj_SCS_weight =    validateCLI(args.adj_SCS_weight,    lambda v: v > 0.0, "adj_SCS_weight debe ser > 0.")

    alpha = validateCLI(args.alpha, lambda v: v > 0.0, "alpha debe ser > 0.")
    beta =  validateCLI(args.beta,  lambda v: v > 0.0, "beta debe ser  > 0.")
    psi =   validateCLI(args.psi,   lambda v: v > 0.0, "psi debe ser  > 0.")
    Aox =   validateCLI(args.aox,   lambda v: v > 0.0, "Aox debe ser  > 0.")
    kox =   validateCLI(args.kox,   lambda v: v > 0.0, "kox debe ser  > 0.")
    Per =   validateCLI(args.per,   lambda v: v > 0.0, "Per debe ser  > 0.")
    Sv =    validateCLI(args.sv,    lambda v: v > 0.0, "Sv debe ser  > 0.")
    Sc =    validateCLI(args.sc,    lambda v: v > 0.0, "Sc debe ser  > 0.")
    pmax =  validateCLI(args.pmax,  lambda v: v > 0.0, "pmax debe ser  > 0.")
    b =     validateCLI(args.b,     lambda v: v > 0.0, "b debe ser  > 0.")

    DesiredSigma =  validateCLI(args.sigma_d,   lambda v: v >= 0.0, "DesiredSigma no puede ser negativo.")
    DesiredSigmaT = validateCLI(args.sigma_dt,  lambda v: v >= 0.0, "DesiredSigmaT no puede ser negativo.")

    k1 = validateCLI(args.k1, lambda v: v >= 0.0, "k1 no puede ser negativo.")
    k2 = validateCLI(args.k2, lambda v: v >= 0.0, "k2 no puede ser negativo.")
    k3 = validateCLI(args.k3, lambda v: v >= 0.0, "k3 no puede ser negativo.")
    k4 = validateCLI(args.k4, lambda v: v >= 0.0, "k4 no puede ser negativo.")
    l1 = validateCLI(args.l1, lambda v: v >= 0.0, "l1 no puede ser negativo.")
    l2 = validateCLI(args.l2, lambda v: v >= 0.0, "l2 no puede ser negativo.")

    CT = validateCLI(args.ct, lambda v: v > 0.0, "CT debe ser > 0.")

    T =     validateCLI(args.T,      lambda v: v > 0, "T debe ser > 0.")
    CT =    validateCLI(CT,          lambda v: v <= T, "CT debe ser < T.")
    nIter = validateCLI(args.n_iter, lambda v: v >= 1, "nIter debe ser >= 1.")

    u0_inicial = validateCLI(args.u0_inicial, lambda v: v > 0.0, "u0_inicial debe ser > 0.")
    s0_inicial = validateCLI(args.s0_inicial, lambda v: v > 0.0, "s0_inicial debe ser > 0.")

    initial_step_size =     validateCLI(args.initial_step_size,     lambda v: v > 0.0,                      "initial_step_size debe ser > 0.")
    min_step_size =         validateCLI(args.min_step_size,         lambda v: 0.0 < v <= initial_step_size, "min_step_size debe ser > 0 y no puede ser más grande que initial_step_size.")
    step_reduction_factor = validateCLI(args.step_reduction_factor, lambda v: 0.0 < v < 1.0,                "step_reduction_factor debe estar en (0, 1).")
    max_step_reductions =   validateCLI(args.max_step_reductions,   lambda v: v >= 0,                       "max_step_reductions debe ser >= 0.")
    armijo_constant_c =     validateCLI(args.armijo_constant_c,     lambda v: 0.0 < v < 1.0,                "armijo_constant_cdebe estar en (0, 1).")
    max_iter_optimo =       validateCLI(args.max_iter_optimo,       lambda v: v >= 1,                       "max_iter_optimo debe ser >= 1.")
    stop_j_tol =            validateCLI(args.stop_j_tol,            lambda v: v > 0.0,                      "stop_j_tol debe ser > 0.")

    control =           validateCLI(args.control,           lambda v: v in ["start", "constant", "sinusoidal", "end"],      "El control debe ser uno de los siguientes: 'start', 'constant', 'sinusoidal', o 'end'.")
    simulation =        validateCLI(args.simulation,        lambda v: isinstance(v, str) and v.strip() != "", "simulation debe ser una cadena no vacía.")
    save_every_n_epochs = validateCLI(args.save_every_n_epochs, lambda v: v >= 1,                                  "save_every_n_epochs debe ser >= 1.")

    dt = T / nIter
    n_points = nIter + 1
    t_grid = np.linspace(0.0, T, n_points)

    C0, S0 = build_initial_controls(control_type=control, CT=CT, T=T, nIter=nIter)
