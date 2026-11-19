import json

import numpy as np

import params
from tool.control_algorithm_tools import (armijo_update_controls, build_iteration_log_line, compute_gradients, evaluate_functional_with_states)


def compute_control_iteration_step(C, S, uVec, sVec, wVec, zVec, mode, control_iteration, J=None, J_terms=None):
    dC, dS = compute_gradients(uVec, sVec, wVec, zVec)  # Paso 3. Calcular los gradientes
    if J is None and J_terms is None:
        J, J_terms = evaluate_functional_with_states(uVec, sVec, C, S)  # Paso 5. Calcular la función sobre los estados calculados
    result = armijo_update_controls(J, J_terms, C, S, dC, dS, mode=mode, control_iteration=control_iteration)
    return result


def control_algorithm(save_folder=None):
    from classic_solver import adjoint_solve_classic, states_solve_classic
    from pinn_solver import train_pinn_in_subintervals

    J_history_pinn = []
    J_history_classic = []

    C = params.C0
    S = params.S0

    pinn_cache = None
    classic_cache = None

    pinn_iter_stop = None
    classic_iter_stop = None

    wVec = None
    zVec = None
    w_classic = None
    z_classic = None

    for it in range(0, params.max_iter_optimo + 1):
        if save_folder is not None:
            save_folder.register_control_iteration_start(it)

        # ==================== Solución clásica - Algoritmo de control ====================
        # Acá se resuelve el problema de control óptimo únicamente usando el solver clásico.

        if it == 0:
            classic_result = states_solve_classic(params.C0, params.S0)
        else:
            classic_result = classic_cache["result"]

        u_classic = classic_result["u_classic"]
        s_classic = classic_result["s_classic"]

        if classic_iter_stop is None:
            w_classic, z_classic = adjoint_solve_classic(u_classic, s_classic, params.C0, params.S0)
            if classic_cache is not None:
                classic_control_step = compute_control_iteration_step(C=params.C0, S=params.S0, uVec=u_classic, sVec=s_classic, wVec=w_classic, zVec=z_classic, mode="classic", control_iteration=it, J=classic_cache["J_new"], J_terms=classic_cache["J_terms_new"])
            else:
                classic_control_step = compute_control_iteration_step(C=params.C0, S=params.S0, uVec=u_classic, sVec=s_classic, wVec=w_classic, zVec=z_classic, mode="classic", control_iteration=it)
        else:
            classic_control_step = classic_cache

        # En la primera iteración, si Armijo no acepta controles nuevos,
        # se conserva el resultado ya calculado para los controles actuales. (inicialización de la caché)
        if it == 0 and classic_control_step["changed_controls"] is False:
            classic_control_step["result"] = classic_result
            classic_control_step["J_new"] = classic_control_step["J_current"]
            classic_control_step["J_terms_new"] = classic_control_step["J_terms_current"]

        # Guarda en caché los controles, estados y funcional aceptados que se reutilizarán
        # en la siguiente iteración; en it=0 inicializa la caché aunque no haya cambio.
        if (classic_iter_stop is None) and (classic_control_step["changed_controls"] or it == 0):
            classic_cache = dict(classic_control_step)
            classic_cache["J_current"] = classic_cache["J_new"]
            classic_cache["J_terms_current"] = classic_cache["J_terms_new"]
            classic_cache["IntC_current"] = classic_cache["IntC_new"]

        # ==================== PINN solve - Control Algorithm ===========================
        # Acá se resuelve el problema de control óptimo únicamente usando PINNs.
        
        if it == 0:
            state_pinn_result = train_pinn_in_subintervals(problem_type="state", Cvec=C, Svec=S, control_iteration=it)
        else:
            state_pinn_result = pinn_cache["result"]

        uVec = state_pinn_result["u_pinn"]
        sVec = state_pinn_result["s_pinn"]

        # =============== Guardado del entrenamiento de las variables de estado ===================
        if save_folder is not None:
            for checkpoint_row in state_pinn_result["checkpoint_rows"]:
                    save_folder.save_checkpoint(problem_type="state", control_iteration=it, epoch=checkpoint_row["epoch"], total_epochs=checkpoint_row["total_epochs"], loss_value=checkpoint_row["loss_value"], var1_pred=checkpoint_row["var1_pred"], var2_pred=checkpoint_row["var2_pred"], var1_ref=u_classic, var2_ref=s_classic, J_epoch=checkpoint_row.get("J_epoch", np.nan), is_best_epoch=checkpoint_row.get("is_best_epoch", False))
            save_folder.finalize_training("state", it, state_pinn_result["loss_history"])
        # =============== fin del guardado ===================

        # ========= Se resuelve el sistema adjunto y se guarda =====
        if pinn_iter_stop is None:
            adjoint_pinn_result = train_pinn_in_subintervals(problem_type="adjoint", Cvec=C, Svec=S, uVec=uVec, sVec=sVec, control_iteration=it)

            wVec = adjoint_pinn_result["w_pinn"]
            zVec = adjoint_pinn_result["z_pinn"]

            if save_folder is not None:
                for checkpoint_row in adjoint_pinn_result["checkpoint_rows"]:
                    save_folder.save_checkpoint(problem_type="adjoint", control_iteration=it, epoch=checkpoint_row["epoch"], total_epochs=checkpoint_row["total_epochs"], loss_value=checkpoint_row["loss_value"], var1_pred=checkpoint_row["var1_pred"], var2_pred=checkpoint_row["var2_pred"], var1_ref=w_classic, var2_ref=z_classic, J_epoch=checkpoint_row.get("J_epoch", np.nan), is_best_epoch=checkpoint_row.get("is_best_epoch", False))
                save_folder.finalize_training("adjoint", it, adjoint_pinn_result["loss_history"])
        # ========= Fin del guardado de adjuntos =====

        if pinn_iter_stop is None:
            if pinn_cache is not None:
                pinn_control_step = compute_control_iteration_step(C=C, S=S, uVec=uVec, sVec=sVec, wVec=wVec, zVec=zVec, mode="pinn", control_iteration=it, J=pinn_cache["J_new"], J_terms=pinn_cache["J_terms_new"])
            else:
                pinn_control_step = compute_control_iteration_step(C=C, S=S, uVec=uVec, sVec=sVec, wVec=wVec, zVec=zVec, mode="pinn", control_iteration=it)
        else:
            pinn_control_step = pinn_cache

        # En la primera iteración, si Armijo no acepta controles nuevos,
        # se conserva el resultado ya calculado para los controles actuales. (inicialización de la caché)
        if it == 0 and pinn_control_step["changed_controls"] is False:
            pinn_control_step["result"] = state_pinn_result
            pinn_control_step["J_new"] = pinn_control_step["J_current"]
            pinn_control_step["J_terms_new"] = pinn_control_step["J_terms_current"]

        # Guarda en caché los controles, estados y funcional aceptados que se reutilizarán
        # en la siguiente iteración; en it=0 inicializa la caché aunque no haya cambio.
        if (pinn_iter_stop is None) and (pinn_control_step["changed_controls"] or it == 0):
            pinn_cache = dict(pinn_control_step)
            pinn_cache["J_current"] = pinn_cache["J_new"]
            pinn_cache["J_terms_current"] = pinn_cache["J_terms_new"]
            pinn_cache["IntC_current"] = pinn_cache["IntC_new"]

        print(build_iteration_log_line(it=it, pinn_control_step=pinn_control_step, classic_control_step=classic_control_step))

        J_history_pinn.append(pinn_control_step["J_current"])
        J_history_classic.append(classic_control_step["J_current"])

        if save_folder is not None:
            save_folder.register_control_iteration_end(iteration=it, J_pinn=pinn_control_step["J_current"], J_classic=classic_control_step["J_current"], J_terms_pinn=pinn_control_step["J_terms_current"], J_terms_classic=classic_control_step["J_terms_current"], Cvec_pinn=C, Svec_pinn=S, uVec_pinn=uVec, sVec_pinn=sVec, wVec_pinn=wVec, zVec_pinn=zVec, Cvec_classic=params.C0, Svec_classic=params.S0, uVec_classic=u_classic, sVec_classic=s_classic, wVec_classic=w_classic, zVec_classic=z_classic, IntC_pinn=pinn_control_step["IntC_current"], IntC_classic=classic_control_step["IntC_current"])

        # Actualizar los controles para la próxima iteración
        if classic_iter_stop is None:
            params.C0 = classic_control_step["C_new"]
            params.S0 = classic_control_step["S_new"]

        if pinn_iter_stop is None:
            C = pinn_control_step["C_new"]
            S = pinn_control_step["S_new"]

        # Criterio de parada
        if len(J_history_classic) >= 4 and classic_iter_stop is None:
            if np.all(np.abs(np.diff(J_history_classic[-4:])) < params.stop_j_tol):
                classic_iter_stop = it
        if len(J_history_pinn) >= 4 and pinn_iter_stop is None:
            if np.all(np.abs(np.diff(J_history_pinn[-4:])) < params.stop_j_tol):
                pinn_iter_stop = it

        # Guardar la iteración de parada para "classic" y "pinn" en run_info.json
        if save_folder is not None:
            run_info = {}
            if save_folder.run_info_path.exists():
                with save_folder.run_info_path.open("r", encoding="utf-8") as file:
                    run_info = json.load(file)

            run_info["stop_j_tol"] = float(params.stop_j_tol)
            run_info["classic_stop_iteration"] = classic_iter_stop
            run_info["pinn_stop_iteration"] = pinn_iter_stop

            with save_folder.run_info_path.open("w", encoding="utf-8") as file:
                json.dump(run_info, file, indent=2, ensure_ascii=False)

        if (pinn_iter_stop is not None) and (classic_iter_stop is not None):
            stop_msg = (
                f"Stopping criterion. PINN stop iter={pinn_iter_stop}, "
                f"classic stop iter={classic_iter_stop}."
            )
            print(stop_msg)
            return
