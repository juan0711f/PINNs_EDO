import numpy as np
import torch

import params
from tool.pinn_tools import (build_neural_network, evaluate_in_collocation_points, model_outputs_subinterval, time_derivative)

torch.set_default_dtype(torch.float64)


def rhs_state_torch(t, u, sigma, t_grid, c_ctrl_vals, s_ctrl_vals):
    c_ctrl = evaluate_in_collocation_points(t, t_grid, c_ctrl_vals)
    s_ctrl = evaluate_in_collocation_points(t, t_grid, s_ctrl_vals)
    rhs_u = (
        (params.pmax / params.alpha)
        * (sigma / params.beta + params.b * (1.0 - sigma / params.beta))
        * u
        * (params.alpha - u)
        - params.psi * c_ctrl * u * sigma
    )
    rhs_sigma = (
        -u * (params.Aox * sigma / (params.kox + sigma + 1e-15))
        + params.Per * params.Sv * (params.beta - sigma)
        + (1.0 - s_ctrl) * params.Sc * u
    )
    return rhs_u, rhs_sigma


def rhs_adjoint_torch(t, w, z, t_grid, u_vals, s_vals, c_ctrl_vals, s_ctrl_vals):
    c_ctrl = evaluate_in_collocation_points(t, t_grid, c_ctrl_vals)
    s_ctrl = evaluate_in_collocation_points(t, t_grid, s_ctrl_vals)
    u = evaluate_in_collocation_points(t, t_grid, u_vals)
    sigma = evaluate_in_collocation_points(t, t_grid, s_vals)

    rhs_w = -(
        params.k1 * u
        + (params.pmax / params.alpha) * (sigma / params.beta + params.b * (1.0 - sigma / params.beta)) * (params.alpha - 2 * u) * w
        - params.psi * c_ctrl * sigma * w
        - (params.Aox * sigma / (params.kox + sigma)) * z
        + (1 - s_ctrl) * params.Sc * z
    )
    rhs_z = -(
        params.k2 * (sigma - params.DesiredSigma)
        + (params.pmax / params.alpha) * (1 / params.beta - params.b / params.beta) * u * (params.alpha - u) * w
        - params.psi * c_ctrl * u * w
        - params.Per * params.Sv * z
        - (params.Aox * u * params.kox / ((params.kox + sigma) ** 2)) * z
    )
    return rhs_w, rhs_z


def train_pinn_in_subintervals(problem_type, Cvec, Svec, uVec=None, sVec=None, control_iteration=None):
    if problem_type == "state":
        pinn_mode = params.state_pinn_mode
        pinn_hidden_width = params.state_hidden_width
        pinn_hidden_depth = params.state_hidden_depth
        pinn_SCS_weight = params.state_SCS_weight
        pinn_lr = params.state_lr
        pinn_subintervals = params.state_subintervals
        pinn_cant_collocation_points = max(32, params.state_n_collocation)

        subintervals = np.linspace(0, params.T, pinn_subintervals + 1)

        pinn_epochs = params.state_epochs

        var1_init = torch.tensor(params.u0_inicial)
        var2_init = torch.tensor(params.s0_inicial)
        u_vals = None
        s_vals = None
    elif problem_type == "adjoint":
        pinn_mode = params.adj_pinn_mode
        pinn_hidden_width = params.adj_hidden_width
        pinn_hidden_depth = params.adj_hidden_depth
        pinn_SCS_weight = params.adj_SCS_weight
        pinn_lr = params.adj_lr
        pinn_subintervals = params.adj_subintervals
        pinn_cant_collocation_points = max(32, params.adj_n_collocation)

        subintervals = np.linspace(params.T, 0, pinn_subintervals + 1)

        pinn_epochs = params.adj_epochs

        var1_init = torch.tensor(params.l1 * float(uVec[-1]))
        var2_init = torch.tensor(params.l2 * float(sVec[-1] - params.DesiredSigmaT))

        u_vals = torch.tensor(uVec)
        s_vals = torch.tensor(sVec)
    else:
        raise ValueError("problem_type debe ser 'state' o 'adjoint'.")

    t_grid = torch.tensor(params.t_grid)
    c_control_vals = torch.tensor(Cvec)
    s_control_vals = torch.tensor(Svec)

    var1_global = np.full(params.t_grid.shape, np.nan)
    var2_global = np.full(params.t_grid.shape, np.nan)

    loss_history = []
    epoch_counter = 0
    total_epochs = pinn_epochs * pinn_subintervals
    checkpoint_rows = []
    best_checkpoint_loss = float("inf")
    best_checkpoint_row = None

    for subinterval_index in range(pinn_subintervals):
        interval_start = subintervals[subinterval_index]
        interval_end = subintervals[subinterval_index + 1]

        t_colollocation = torch.linspace(interval_start, interval_end, steps=pinn_cant_collocation_points).unsqueeze(1)
        t_colollocation.requires_grad_(True)

        # Identificación de los índices de la malla temporal (donde se definen los puntos de control) que corresponden a este subintervalo.
        if interval_start <= interval_end:
            subinterval_ids = np.where((params.t_grid >= interval_start) & (params.t_grid <= interval_end))[0]
        else:
            subinterval_ids = np.where((params.t_grid >= interval_end) & (params.t_grid <= interval_start))[0]

        t_init = torch.tensor(interval_start)
        t_end = torch.tensor(interval_end)
        t_end_point = torch.tensor([[interval_end]])

        model = build_neural_network(width=pinn_hidden_width, depth=pinn_hidden_depth, out_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=pinn_lr)

        minn_loss_value = float("inf")
        weights_of_min_loss = None

        for _ in range(1, pinn_epochs + 1):
            optimizer.zero_grad()
            zero_term = torch.zeros((), dtype=torch.float64)
            loss_res_1 = zero_term
            loss_res_2 = zero_term
            loss_ic_1 = zero_term
            loss_ic_2 = zero_term

            var1_col, var2_col = model_outputs_subinterval(model, t_colollocation, pinn_mode=pinn_mode, var_init_list=[var1_init, var2_init], t_init=t_init, t_end=0)

            dvar1_dt = time_derivative(var1_col, t_colollocation)
            dvar2_dt = time_derivative(var2_col, t_colollocation)

            if problem_type == "state":
                rhs_var1, rhs_var2 = rhs_state_torch(t_colollocation, var1_col, var2_col, t_grid=t_grid, c_ctrl_vals=c_control_vals, s_ctrl_vals=s_control_vals)
            else:
                rhs_var1, rhs_var2 = rhs_adjoint_torch(t_colollocation, var1_col, var2_col, t_grid=t_grid, u_vals=u_vals, s_vals=s_vals, c_ctrl_vals=c_control_vals, s_ctrl_vals=s_control_vals)

            loss_res_1 = torch.mean((dvar1_dt - rhs_var1) ** 2)
            loss_res_2 = torch.mean((dvar2_dt - rhs_var2) ** 2)
            loss = loss_res_1 + loss_res_2

            if pinn_mode == "SCS_PINNs":
                var1_init_pred = var1_col[0]
                var2_init_pred = var2_col[0]
                loss_ic_1 = pinn_SCS_weight * torch.mean((var1_init_pred - var1_init) ** 2)
                loss_ic_2 = pinn_SCS_weight * torch.mean((var2_init_pred - var2_init) ** 2)
                loss += loss_ic_1 + loss_ic_2

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            loss_history.append(
                [
                    float(loss_res_1.detach().cpu().item()),
                    float(loss_res_2.detach().cpu().item()),
                    float(loss_ic_1.detach().cpu().item()),
                    float(loss_ic_2.detach().cpu().item()),
                    loss_value,
                ]
            )

            if loss_value < minn_loss_value:
                minn_loss_value = loss_value
                weights_of_min_loss = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

            if loss_value < best_checkpoint_loss:
                best_checkpoint_loss = loss_value
                var1_pred_best = np.full(params.t_grid.shape, np.nan)
                var2_pred_best = np.full(params.t_grid.shape, np.nan)
                t_eval_best = torch.tensor(params.t_grid[subinterval_ids]).unsqueeze(1)
                with torch.no_grad():
                    var1_eval_best, var2_eval_best = model_outputs_subinterval(model, t_eval_best, pinn_mode=pinn_mode, var_init_list=[var1_init, var2_init], t_init=t_init, t_end=0)
                var1_pred_best[subinterval_ids] = var1_eval_best.detach().cpu().numpy().reshape(-1)
                var2_pred_best[subinterval_ids] = var2_eval_best.detach().cpu().numpy().reshape(-1)
                best_checkpoint_row = {"problem_type": problem_type, "control_iteration": control_iteration, "epoch": epoch_counter, "total_epochs": total_epochs, "loss_value": loss_value, "var1_pred": var1_pred_best, "var2_pred": var2_pred_best, "J_epoch": np.nan, "is_best_epoch": True}

            # Guardar este entrenamiento
            if epoch_counter == 1 or epoch_counter % params.save_every_n_epochs == 0 or epoch_counter == total_epochs:
                var1_pred_epoch = np.full(params.t_grid.shape, np.nan)
                var2_pred_epoch = np.full(params.t_grid.shape, np.nan)

                t_eval = torch.tensor(params.t_grid[subinterval_ids]).unsqueeze(1)
                with torch.no_grad():
                    var1_eval, var2_eval = model_outputs_subinterval(model, t_eval, pinn_mode=pinn_mode, var_init_list=[var1_init, var2_init], t_init=t_init, t_end=0)

                var1_pred_epoch[subinterval_ids] = var1_eval.detach().cpu().numpy().reshape(-1)
                var2_pred_epoch[subinterval_ids] = var2_eval.detach().cpu().numpy().reshape(-1)

                checkpoint_row = {"problem_type": problem_type, "control_iteration": control_iteration, "epoch": epoch_counter, "total_epochs": total_epochs, "loss_value": loss_value, "var1_pred": var1_pred_epoch, "var2_pred": var2_pred_epoch, "J_epoch": np.nan, "is_best_epoch": False}
                checkpoint_rows.append(checkpoint_row)

            epoch_counter += 1

        if weights_of_min_loss is not None:
            model.load_state_dict(weights_of_min_loss)

        # La predicción final para el subintervalo se almacena en los nodos de t_grid.
        if subinterval_ids.size > 0:
            t_subinterval_t = torch.tensor(params.t_grid[subinterval_ids]).unsqueeze(1)
            with torch.no_grad():
                var1_subinterval_final, var2_subinterval_final = model_outputs_subinterval(
                    model,
                    t_subinterval_t,
                    pinn_mode=pinn_mode,
                    var_init_list=[var1_init, var2_init],
                    t_init=t_init,
                    t_end=t_end,
                )
            # Las predicciones finales para el subintervalo se almacenan en las variables de resultado globales (en los índices correspondientes).
            var1_global[subinterval_ids] = var1_subinterval_final.squeeze(1).cpu().numpy()
            var2_global[subinterval_ids] = var2_subinterval_final.squeeze(1).cpu().numpy()

        # Realiza la evaluación al final del subintervalo para obtener los datos iniciales del siguiente subintervalo.
        with torch.no_grad():
            var1_next, var2_next = model_outputs_subinterval(model, t_end_point, pinn_mode=pinn_mode, var_init_list=[var1_init, var2_init], t_init=t_init, t_end=t_end)

        # Actualiza los datos iniciales para el siguiente subintervalo.
        var1_init = float(var1_next.item())
        var2_init = float(var2_next.item())

    var1_final = np.asarray(var1_global)
    var2_final = np.asarray(var2_global)
    loss_history_arr = np.asarray(loss_history, dtype=np.float64)

    if best_checkpoint_row is not None:
        checkpoint_rows = [row for row in checkpoint_rows if int(row["epoch"]) != int(best_checkpoint_row["epoch"])]
        checkpoint_rows.append(best_checkpoint_row)
        checkpoint_rows.sort(key=lambda row: int(row["epoch"]))

    if problem_type == "state":
        return {"u_pinn": var1_final, "s_pinn": var2_final, "loss_history": loss_history_arr, "n_subintervals": pinn_subintervals, "checkpoint_rows": checkpoint_rows}

    return {"w_pinn": var1_final, "z_pinn": var2_final, "loss_history": loss_history_arr, "n_subintervals": pinn_subintervals, "checkpoint_rows": checkpoint_rows}
