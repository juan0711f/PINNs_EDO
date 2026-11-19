import numpy as np
from scipy.integrate import solve_ivp
import params

""" Resuelve el sistema de estados (u, sigma) utilizando la función solve_ivp. """
def states_solve_classic(Cvec, Svec):

    Cvec = np.asarray(Cvec)
    Svec = np.asarray(Svec)

    def c_t(t):
        return np.interp(t, params.t_grid, Cvec)

    def s_t(t):
        return np.interp(t, params.t_grid, Svec)

    def state_system(t, y):
        u, sigma = y
        c = c_t(t)
        s = s_t(t)
        frac = params.Aox * sigma / (params.kox + sigma)
        du = (params.pmax / params.alpha) * ((sigma * (1 / params.beta)) + params.b * (1.0 - sigma * (1 / params.beta))) * u * (params.alpha - u) - params.psi * c * u * sigma
        dsigma = (-u * frac + params.Per * params.Sv * (params.beta - sigma) + (1.0 - s) * params.Sc * u)
        return [du, dsigma]

    sol = solve_ivp(
        state_system,
        [0.0, params.T],
        [params.u0_inicial, params.s0_inicial],
        method="Radau",
        t_eval=params.t_grid,
        rtol=1e-8,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp fail: {sol.message}")
    return {"u_classic": np.asarray(sol.y[0]), "s_classic": np.asarray(sol.y[1])}

""" Resuelve el sistema adjunto (w, z) utilizando la función solve_ivp. """
def adjoint_solve_classic(uVec, sVec, Cvec, Svec):

    uVec = np.asarray(uVec)
    sVec = np.asarray(sVec)
    Cvec = np.asarray(Cvec)
    Svec = np.asarray(Svec)

    def adjoint_system(t, y):
        w, z = y
        u = np.interp(t, params.t_grid, uVec)
        sigma = np.interp(t, params.t_grid, sVec)
        c_ctrl = np.interp(t, params.t_grid, Cvec)
        s_ctrl = np.interp(t, params.t_grid, Svec)
        rhs_w = -(params.k1 * u + (params.pmax/params.alpha)*(sigma/params.beta + params.b*(1.0-sigma/params.beta))*(params.alpha - 2*u)*w - params.psi*c_ctrl*sigma*w - (params.Aox*sigma / (params.kox + sigma))*z + (1-s_ctrl)*params.Sc*z)
        rhs_z = -(params.k2*(sigma - params.DesiredSigma) + (params.pmax/params.alpha)*(1/params.beta - params.b/params.beta)*u*(params.alpha - u)*w - params.psi*c_ctrl*u*w - params.Per*params.Sv*z - (params.Aox*u*params.kox/((params.kox+sigma)**2))*z)
        return [rhs_w, rhs_z]

    w_terminal = float(params.l1 * uVec[-1])
    z_terminal = float(params.l2 * (sVec[-1] - params.DesiredSigmaT))
    t_eval_desc = params.t_grid[::-1]

    sol = solve_ivp(
        adjoint_system,
        [float(params.T), 0.0],
        [w_terminal, z_terminal],
        method="Radau",
        t_eval=t_eval_desc,
        rtol=1e-8,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp adjunto fail: {sol.message}")

    return np.asarray(sol.y[0][::-1]), np.asarray(sol.y[1][::-1])
