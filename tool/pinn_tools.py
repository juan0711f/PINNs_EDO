import random

import numpy as np
import torch
import torch.nn as nn

import params


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)

def evaluate_in_collocation_points(t, t_grid, variable_values):
    insertion_index = torch.searchsorted(t_grid, t, right=False)
    previous_index = torch.clamp(insertion_index - 1, 0, t_grid.numel() - 2)
    next_index = previous_index + 1

    previous_time = t_grid[previous_index]
    next_time = t_grid[next_index]
    control_previous = variable_values[previous_index]
    control_next = variable_values[next_index]

    control_value = control_previous + ((t - previous_time) / (next_time - previous_time)) * (control_next - control_previous)

    return control_value

def build_neural_network(width=96, depth=4, out_dim=2):
    layers = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), nn.Tanh()])
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)

def time_derivative(y, t):
    return torch.autograd.grad(
        y,
        t,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]

def model_outputs_subinterval(model, t, pinn_mode, var_init_list, t_init, t_end):
    out = model(t)

    if out.shape[1] != len(var_init_list):
        raise ValueError("La salida de la red neuronal debe coincidir con len(var_init_list)")

    vars_out = []

    for i, var_init in enumerate(var_init_list):
        if pinn_mode == "HCS_PINNs":
            if i <= 1:
                var_i = var_init + (t - t_init) * out[:, i:i+1]
            else:
                var_i = var_init + (t-t_end)*out[:, i:i+1]
        elif pinn_mode == "SCS_PINNs":
            var_i = out[:, i:i+1]
        vars_out.append(var_i)
    return vars_out
