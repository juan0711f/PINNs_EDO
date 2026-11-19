import numpy as np
import params

""" Aplica la regla trapezoidal sobre la malla temporal para calcular una integral """
def integral_trapz(vec):
    return params.dt * 0.5 * (vec[0] + 2.0 * np.sum(vec[1:-1]) + vec[-1])

""" Proyección del control C (restricción integral) """
def project_c_integral_constraint(Cvec):
    Cvec[:] = np.clip(Cvec, 0.0, 1.0)

    # Ponderaciones de la regla trapezoidal para la integral discreta.
    w = np.zeros_like(Cvec)
    w[0] = 0.5 * params.dt
    w[-1] = 0.5 * params.dt
    w[1:-1] = params.dt

    def phi(gamma):
        cproj = np.clip(Cvec - gamma * w, 0.0, 1.0)
        return float(np.dot(w, cproj) - params.CT)

    if phi(0.0) <= 0.0:
        return float(np.dot(w, Cvec))

    # Construcción del intervalo [gamma_low, gamma_high] con cambio de signo.
    gamma_low = 0.0
    gamma_high = 1.0
    while phi(gamma_high) > 0.0:
        gamma_high *= 2.0

    # Bisección para resolver phi(gamma) = 0.
    for _ in range(60):
        gamma_mid = 0.5 * (gamma_low + gamma_high)
        if phi(gamma_mid) > 0.0:
            gamma_low = gamma_mid
        else:
            gamma_high = gamma_mid

    Cvec[:] = np.clip(Cvec - gamma_high * w, 0.0, 1.0)

    return float(np.dot(w, Cvec))

""" Calcula los gradientes dJ/dC y dJ/dS """
def compute_gradients(uVec, sVec, wVec, zVec):
    dU = params.dt * (params.k3 - params.psi * uVec * sVec * wVec)
    dS = params.dt * (params.k4 - params.Sc * uVec * zVec)
    dU[0] *= 0.5
    dU[-1] *= 0.5
    dS[0] *= 0.5
    dS[-1] *= 0.5
    return dU, dS

def build_iteration_log_line(it, pinn_control_step, classic_control_step):
    J_pinn = pinn_control_step["J_current"]
    J_classic = classic_control_step["J_current"]
    J_error = np.abs(J_pinn - J_classic)
    step_size = pinn_control_step["step_size"]
    step_reductions_used = pinn_control_step["step_reductions_used"]
    IntC = pinn_control_step["IntC_current"]
    return (f"Iter {it:03d} | J_pinn={J_pinn:.6f} "
        f"| J_solve={J_classic:.6f} | |Jp-Js|={J_error:.3e} "
        f"| IntC={IntC:.4f} "
        f"| step_size={step_size:.3e} | reductions={step_reductions_used} ")


def evaluate_functional_with_states(uVec, sVec, Cvec, Svec):
    J1 = 0.5 * params.k1 * integral_trapz(uVec ** 2)
    J2 = 0.5 * params.k2 * integral_trapz((sVec - params.DesiredSigma) ** 2)
    J3 = 0.5 * params.l1 * (uVec[-1] ** 2)
    J4 = 0.5 * params.l2 * ((sVec[-1] - params.DesiredSigmaT) ** 2)
    J5 = params.k3 * integral_trapz(Cvec)
    J6 = params.k4 * integral_trapz(Svec)

    J = J1 + J2 + J3 + J4 + J5 + J6
    return J, {"J1": J1, "J2": J2, "J3": J3, "J4": J4, "J5": J5, "J6": J6}



def armijo_update_controls(J_current, J_terms_list, Cvec, Svec, dC, dS, mode, control_iteration):
    squared_grad_norm = np.dot(dC, dC) + np.dot(dS, dS)

    IntC_current = integral_trapz(Cvec)

    step_reductions_used = 0
    last_candidate_J = J_current
    J_terms = J_terms_list
    Cnew = Cvec
    Snew = Svec
    step_size = max(params.initial_step_size, params.min_step_size)
    changed_controls = False

    if (not np.isfinite(squared_grad_norm)) or squared_grad_norm == 0.0:
        result = None
        #return {"C_new": Cvec, "S_new": Svec, "IntC_new": IntC, "result": None, "J_terms": J_terms,"J_current": J_current, "step_size": step_size, "step_reductions_used": step_reductions_used, "last_candidate_J": last_candidate_J, "changed_controls": changed_controls}
        return {"C_new": Cvec, "S_new": Svec, "IntC_new": IntC_current, "IntC_current": IntC_current, "J_current": J_current, "J_terms_new": J_terms_list, "J_terms_current": J_terms_list, "step_size": 0, "step_reductions_used": 0, "J_new": last_candidate_J, "changed_controls": changed_controls, "result": result}
    else:
        for k in range(params.max_step_reductions + 1):
            C_candidate = np.clip(Cvec - step_size * dC, 0.0, 1.0)
            S_candidate = np.clip(Svec - step_size * dS, 0.0, 1.0)

            project_c_integral_constraint(C_candidate)

            if (np.array_equal(C_candidate, Cvec) and np.array_equal(S_candidate, Svec)):
                result=None
                #return {"C": Cvec, "S": Svec, "IntC": IntC, "result": None, "J_current": J_current, "J_terms": None, "step_size": step_size, "step_reductions_used": step_reductions_used, "last_candidate_J": last_candidate_J, "changed_controls": changed_controls}
                return {"C_new": Cvec, "S_new": Svec, "IntC_new": IntC_current, "IntC_current": IntC_current, "J_current": J_current, "J_terms_new": J_terms_list, "J_terms_current": J_terms_list, "step_size": 0, "step_reductions_used": 0, "J_new": last_candidate_J, "changed_controls": changed_controls, "result": result}
            else:
                # Calcula las variables de estado para los nuevos controles
                if mode == "pinn":
                    from pinn_solver import train_pinn_in_subintervals

                    pinn_result = train_pinn_in_subintervals(problem_type="state", Cvec=C_candidate, Svec=S_candidate, control_iteration=control_iteration)
                    result = pinn_result
                    u = pinn_result["u_pinn"]
                    sigma = pinn_result["s_pinn"]
                    
                    J, J_terms = evaluate_functional_with_states(u, sigma, Cvec=C_candidate, Svec=S_candidate)
                else: # mode == "classic"
                    from classic_solver import states_solve_classic

                    classic_result = states_solve_classic(Cvec=C_candidate, Svec=S_candidate)
                    result=classic_result
                    
                    u = classic_result["u_classic"]
                    sigma = classic_result["s_classic"]

                    J, J_terms = evaluate_functional_with_states(u, sigma, Cvec=C_candidate, Svec=S_candidate)
                
                last_candidate_J = J
                if J <= J_current - params.armijo_constant_c * step_size * squared_grad_norm:
                    Cnew = C_candidate
                    Snew = S_candidate
                    changed_controls = True
                    step_reductions_used = k
                    break

            # Para no exceder la cantidad máxima de pasos de reducción.
            if k >= params.max_step_reductions:
                step_reductions_used = k
                break

            # Si la siguiente reducción superara el umbral mínimo de paso, se detiene aquí.
            next_step_size = step_size * params.step_reduction_factor
            if next_step_size < params.min_step_size:
                step_reductions_used = k
                break

            step_size = next_step_size
            step_reductions_used = k + 1
        IntC = integral_trapz(Cnew)

        # - Result son los nuevos estados de la siguiente interación.
        # - J_new, J_terms_new, es el funcional evaluado en los controles de la siguiente ieración
        # - changed_controls indica si hay un cambio en los controles para la siguiente iteración.
        #    Así, si no lo hay, se cargan los estados de la anterior iteración
        return {"C_new": Cnew, "S_new": Snew, "IntC_new": IntC, "IntC_current": IntC_current, "J_current": J_current, "J_terms_new": J_terms, "J_terms_current": J_terms_list, "step_size": step_size, "step_reductions_used": step_reductions_used, "J_new": last_candidate_J, "changed_controls": changed_controls, "result": result}
