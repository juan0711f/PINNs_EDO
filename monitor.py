#!/usr/bin/env python3

"""
Monitor grafico para revisar las simulaciones guardadas de control optimo.

El monitor tiene tres pestañas:
1. Control, donde se comparan el funcional y los controles obtenidos con PINNs
   y con el metodo clasico.
2. State PINN, donde se revisa el entrenamiento de las variables de estado.
3. Adj PINN, donde se revisa el entrenamiento de las variables adjuntas.

Los datos se leen desde control_history.npz, run_info.json y las carpetas con
los historiales de entrenamiento. Al seleccionar una iteracion
se actualizan las tres pestañas mostrando así la misma iteracion.
"""

import argparse                         # Para leer la simulacion desde la terminal
import json                             # Permite leer la informacion general de la ejecucion
from pathlib import Path                # Maneja las rutas de resultados

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np                                  # Para leer y organizar los datos numericos
from matplotlib.backend_tools import ToolBase       # Permite crear las pestañas en la barra superior
from matplotlib.patches import Rectangle            # Fondo del panel de controles
from matplotlib.widgets import Button, Slider, TextBox

plt.rcParams["toolbar"] = "toolmanager"             # Se necesita para agregar las tres pestañas


# ====================================================================================================
# =================================== Definicion de clases ============================================
# ====================================================================================================

""" Ventana principal del monitor """
class MonitorApp:
    def __init__(self, simulation):
        self.run_dir = self.resolve_run_dir(simulation)

        # El archivo principal contiene una fila por cada iteracion del control.
        control_path = self.run_dir / "control_history.npz"
        self.control_data = np.load(control_path, allow_pickle=True)

        self.iterations = [int(x) for x in self.control_data["iterations"].tolist()]
        if not self.iterations:
            raise SystemExit("No hay iteraciones guardadas en control_history.npz")

        self.tgrid = np.asarray(self.control_data["t_grid"])
        self.run_info = {}
        run_info_path = self.run_dir / "run_info.json"
        if run_info_path.exists():
            with run_info_path.open("r", encoding="utf-8") as file:
                self.run_info = json.load(file)

        # Se guardan por separado las perdidas del estado y del adjunto.
        self.state_loss_by_iteration = {}
        self.adjoint_loss_by_iteration = {}
        self.state_loss_components_by_iteration = {}
        self.adjoint_loss_components_by_iteration = {}
        self.state_loss_component_count = 0
        self.adjoint_loss_component_count = 0
        self.load_saved_training()
        self.state_loss_component_indices = self.detect_active_component_indices(self.state_loss_components_by_iteration, self.state_loss_component_count)
        self.adjoint_loss_component_indices = self.detect_active_component_indices(self.adjoint_loss_components_by_iteration, self.adjoint_loss_component_count)
        self.state_loss_labels = self.build_loss_component_labels("state", self.state_loss_component_indices)
        self.adjoint_loss_labels = self.build_loss_component_labels("adjoint", self.adjoint_loss_component_indices)

        # Datos utilizados por los controles de reproduccion de la pagina principal.
        self.selected_index = None
        self.playing = False
        self.play_timer = None
        self.play_speed = 1.0

        self.backend_ready()
        plt.ion()

        self.fig = plt.figure(figsize=(13.5, 10.5))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Monitor control optimo - " + self.run_dir.name)

        self.pages = {}
        self.active_page = "control"

        self.build_control_page()
        self.build_state_page()
        self.build_adjoint_page()
        self.setup_tabs()
        self.show_page("control")

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.select_iteration(len(self.iterations) - 1)

    """ Encuentra la carpeta de una simulacion a partir de su nombre o su ruta """
    def resolve_run_dir(self, simulation):
        path = Path(simulation)
        if path.exists():
            return path.resolve()
        default_path = Path("results") / simulation
        if default_path.exists():
            return default_path.resolve()
        raise SystemExit(f"No se encontró la simulación: {simulation}")

    """ Intenta usar un backend que permita trabajar con la ventana interactiva """
    def backend_ready(self):
        backend_name = mpl.get_backend().lower()
        if "qtagg" not in backend_name:
            try:
                plt.switch_backend("QtAgg")
            except Exception:
                if backend_name.endswith("agg"):
                    try:
                        plt.switch_backend("MacOSX")
                    except Exception:
                        pass

    # ====================================================================================================
    # =============================== Lectura de datos guardados =========================================
    # ====================================================================================================

    """ Lee los historiales de entrenamiento disponibles del estado y del adjunto """
    def load_saved_training(self):
        state_training_dir = self.run_dir / "state_training"
        adjoint_training_dir = self.run_dir / "adjoint_training"

        if state_training_dir.exists():
            for file in sorted(state_training_dir.glob("iter_*.npz")):
                iteration = int(file.stem.split("_")[1])
                data = np.load(file, allow_pickle=True)
                loss_data = self.extract_loss_history_data(np.asarray(data["loss_history"]))
                self.state_loss_by_iteration[iteration] = loss_data["total"]
                self.state_loss_components_by_iteration[iteration] = loss_data["components"]
                if loss_data["components"].ndim == 2:
                    self.state_loss_component_count = max(self.state_loss_component_count, int(loss_data["components"].shape[1]))

        if adjoint_training_dir.exists():
            for file in sorted(adjoint_training_dir.glob("iter_*.npz")):
                iteration = int(file.stem.split("_")[1])
                data = np.load(file, allow_pickle=True)
                loss_data = self.extract_loss_history_data(np.asarray(data["loss_history"]))
                self.adjoint_loss_by_iteration[iteration] = loss_data["total"]
                self.adjoint_loss_components_by_iteration[iteration] = loss_data["components"]
                if loss_data["components"].ndim == 2:
                    self.adjoint_loss_component_count = max(
                        self.adjoint_loss_component_count,
                        int(loss_data["components"].shape[1]))

    """ Busca la iteracion de entrenamiento mas cercana que tenga datos guardados """
    def resolve_training_source_iteration(self, iteration, loss_by_iteration):
        full_loss = loss_by_iteration.get(iteration)
        if full_loss is not None and np.asarray(full_loss).size > 0:
            return iteration, full_loss

        candidate_iterations = sorted(loss_by_iteration.keys())
        if not candidate_iterations:
            return iteration, None

        previous_candidates = [it for it in candidate_iterations if it <= iteration]
        if previous_candidates:
            source_iteration = previous_candidates[-1]
        else:
            source_iteration = candidate_iterations[0]

        return source_iteration, loss_by_iteration.get(source_iteration)

    """ Devuelve solamente la perdida total de un historial """
    def extract_total_loss_history(self, loss_history):
        return self.extract_loss_history_data(loss_history)["total"]

    """ Separa la perdida total y sus componentes cuando vienen en una misma matriz """
    def extract_loss_history_data(self, loss_history):
        loss_arr = np.asarray(loss_history, dtype=np.float64)
        if loss_arr.size == 0:
            empty = np.asarray([], dtype=np.float64)
            return {"total": empty, "components": np.empty((0, 0), dtype=np.float64)}

        if loss_arr.ndim == 1:
            total = np.asarray(loss_arr.reshape(-1), dtype=np.float64)
            return {"total": total, "components": np.empty((total.size, 0), dtype=np.float64)}

        if loss_arr.ndim == 2:
            if loss_arr.shape[1] == 0:
                empty = np.asarray([], dtype=np.float64)
                return {"total": empty, "components": np.empty((loss_arr.shape[0], 0), dtype=np.float64)}
            if loss_arr.shape[1] == 1:
                total = np.asarray(loss_arr[:, 0], dtype=np.float64)
                return {"total": total, "components": np.empty((loss_arr.shape[0], 0), dtype=np.float64)}

            components = np.asarray(loss_arr[:, :-1], dtype=np.float64)
            total = np.asarray(loss_arr[:, -1], dtype=np.float64)
            return {"total": total, "components": components}

        total = np.asarray(loss_arr.reshape(-1), dtype=np.float64)
        return {"total": total, "components": np.empty((total.size, 0), dtype=np.float64)}

    """ Encuentra las componentes de la perdida que realmente fueron utilizadas """
    def detect_active_component_indices(self, components_by_iteration, max_component_count):
        if max_component_count <= 0:
            return []

        active_by_idx = [False for _ in range(max_component_count)]
        for component_history in components_by_iteration.values():
            components = np.asarray(component_history, dtype=np.float64)
            if components.size == 0:
                continue
            if components.ndim == 1:
                components = components.reshape(-1, 1)
            elif components.ndim != 2:
                components = np.asarray(components.reshape(-1, 1), dtype=np.float64)

            n_components = min(max_component_count, components.shape[1])
            for comp_idx in range(n_components):
                if np.any(components[:, comp_idx] > 0.0):
                    active_by_idx[comp_idx] = True

        return [comp_idx for comp_idx, is_active in enumerate(active_by_idx) if is_active]

    """ Coloca los nombres de las componentes del estado o del adjunto """
    def build_loss_component_labels(self, problem_type, component_indices):
        if not component_indices:
            return []

        if problem_type == "state":
            base_labels = ["loss_res_u", "loss_res_sigma", "loss_ic_u", "loss_ic_sigma"]
        else:
            base_labels = ["loss_res_w", "loss_res_z", "loss_ic_w", "loss_ic_z"]

        labels = []
        for comp_idx in component_indices:
            if comp_idx < len(base_labels):
                labels.append(base_labels[comp_idx])
            else:
                labels.append(f"loss_comp_{comp_idx + 1}")
        return labels

    """ Actualiza las lineas de cada componente de la funcion de perdida """
    def set_component_loss_lines(self, lines, component_indices, component_history):
        if not lines:
            return

        components = np.asarray(component_history, dtype=np.float64)
        if components.size == 0:
            for line in lines:
                line.set_data([], [])
            return

        if components.ndim == 1:
            components = components.reshape(-1, 1)
        elif components.ndim != 2:
            components = np.asarray(components.reshape(-1, 1), dtype=np.float64)

        epochs_all = np.arange(1, components.shape[0] + 1)
        for line_idx, line in enumerate(lines):
            if line_idx >= len(component_indices):
                line.set_data([], [])
                continue

            comp_idx = int(component_indices[line_idx])
            if comp_idx >= components.shape[1]:
                line.set_data([], [])
                continue

            comp_values = np.asarray(components[:, comp_idx], dtype=np.float64)
            if np.any(comp_values > 0.0):
                line.set_data(epochs_all, np.maximum(comp_values, 1e-30))
            else:
                line.set_data([], [])

    # ====================================================================================================
    # =============================== Manejo de las pestañas =============================================
    # ====================================================================================================

    """ Registra los ejes y textos que pertenecen a una pagina """
    def register_page(self, name, axes, artists):
        self.pages[name] = {"axes": list(axes), "artists": list(artists)}
        for ax in self.pages[name]["axes"]:
            ax.set_visible(False)
        for artist in self.pages[name]["artists"]:
            artist.set_visible(False)

    """ Muestra una pagina y oculta los elementos de las otras dos """
    def show_page(self, name):
        self.active_page = name
        for page_name, page in self.pages.items():
            visible = page_name == name
            for ax in page["axes"]:
                ax.set_visible(visible)
            for artist in page["artists"]:
                artist.set_visible(visible)
        self.refresh()

    """ Crea los tres botones que funcionan como pestañas en la barra superior """
    def setup_tabs(self):
        manager = self.fig.canvas.manager
        toolmanager = manager.toolmanager
        toolbar = manager.toolbar

        for tool_name in ["StatePINN", "AdjPINN", "Control"]:
            if tool_name in toolmanager.tools:
                try:
                    toolmanager.remove_tool(tool_name)
                except Exception:
                    pass

        page_data = [
            ("StatePINN", "state", "State PINN"),
            ("AdjPINN", "adjoint", "Adj PINN"),
            ("Control", "control", "Control")]

        for tool_name, page_name, description in page_data:
            class TabTool(ToolBase):
                pass

            TabTool.description = description

            def trigger(tool_self, sender, event, data=None, selected_page=page_name):
                self.show_page(selected_page)

            TabTool.trigger = trigger
            toolmanager.add_tool(tool_name, TabTool)
            toolbar.add_tool(tool_name, "navigation")

    # ====================================================================================================
    # =============================== Pagina del control =================================================
    # ====================================================================================================

    """ Crea las graficas y los controles de la pagina principal """
    def build_control_page(self):
        # Panel superior con los mismos controles utilizados en el monitor de FreeFem.
        self.control_panel_ax = self.fig.add_axes([0.08, 0.90, 0.84, 0.055])
        self.control_panel_ax.axis("off")
        self.control_panel_ax.set_title("Controles", loc="left", fontsize=11, color="#111827", pad=4)
        self.control_panel_ax.add_patch(Rectangle((0, 0), 1, 1, transform=self.control_panel_ax.transAxes, facecolor="#f8fafc", edgecolor="#d9e2ec"))

        # Los ejes pequeños contienen los botones, el slider y la caja de iteracion.
        box = self.control_panel_ax.get_position()
        control_positions = {
            "play": (0.02, 0.25, 0.075, 0.5),
            "restart": (0.11, 0.25, 0.09, 0.5),
            "speed": (0.25, 0.32, 0.20, 0.36),
            "iteration": (0.56, 0.25, 0.10, 0.5)}
        self.control_widget_axes = {}
        for name, (x_pos, y_pos, width, height) in control_positions.items():
            self.control_widget_axes[name] = self.fig.add_axes([box.x0 + x_pos * box.width, box.y0 + y_pos * box.height, width * box.width, height * box.height])

        self.play_button = Button(self.control_widget_axes["play"], "Play", color="#e8eef8", hovercolor="#d8e2f2")
        self.restart_button = Button(self.control_widget_axes["restart"], "Inicio", color="#e7f6ef", hovercolor="#d1f0e5")
        self.speed_slider = Slider(self.control_widget_axes["speed"], "vel", 0.5, 3.0, valinit=1.0, valstep=0.1, color="#a5b4fc")
        self.iteration_box = TextBox(self.control_widget_axes["iteration"], "iter", initial="", color="#f8fafc", hovercolor="#eef2f7")

        self.play_button.on_clicked(self.toggle_playback)
        self.restart_button.on_clicked(lambda event: self.select_iteration(0))
        self.speed_slider.on_changed(self.change_playback_speed)
        self.iteration_box.on_submit(self.submit_iteration)

        # Las posiciones de las seis graficas se conservan para no cambiar las pestañas.
        self.ax_j = self.fig.add_axes([0.08, 0.74, 0.84, 0.14])
        self.ax_j_err = self.fig.add_axes([0.08, 0.57, 0.84, 0.10])
        self.ax_c = self.fig.add_axes([0.08, 0.36, 0.38, 0.16])
        self.ax_s = self.fig.add_axes([0.54, 0.36, 0.38, 0.16])
        self.ax_c_err = self.fig.add_axes([0.08, 0.14, 0.38, 0.16])
        self.ax_s_err = self.fig.add_axes([0.54, 0.14, 0.38, 0.16])

        # Lineas del funcional y puntos que indican la iteracion seleccionada.
        self.line_j_pinn, = self.ax_j.plot([], [], "--o", label="J PINN", color="#ff7f0e", linewidth=1.7, markersize=4)
        self.line_j_classic, = self.ax_j.plot([], [], "-o", label="J clásico", color="#1f77b4", linewidth=1.7, markersize=4)
        self.line_j_err, = self.ax_j_err.plot([], [], "-o", label="|J_PINN - J_clásico|", color="#d62728", linewidth=1.6, markersize=4)

        self.line_j_sel_pinn, = self.ax_j.plot([], [], "o", color="#ff7f0e", markersize=9, markeredgecolor="black", markeredgewidth=1.0)
        self.line_j_sel_classic, = self.ax_j.plot([], [], "o", color="#1f77b4", markersize=9, markeredgecolor="black", markeredgewidth=1.0)
        self.line_j_sel_err, = self.ax_j_err.plot([], [], "o", color="#ffeb3b", markersize=9, markeredgecolor="#b91c1c", markeredgewidth=1.3)

        self.line_j_pinn.set_picker(5)
        self.line_j_classic.set_picker(5)
        self.line_j_err.set_picker(5)

        # Controles de la iteracion y sus errores absolutos.
        self.line_c_classic, = self.ax_c.plot(self.tgrid, np.zeros_like(self.tgrid), "-o", label="Ctrl C clásico", color="#1f77b4", linewidth=1.5, markersize=3)
        self.line_c_pinn, = self.ax_c.plot(self.tgrid, np.zeros_like(self.tgrid), "--o", label="Ctrl C PINN", color="#ff7f0e", linewidth=1.5, markersize=3)
        self.line_s_classic, = self.ax_s.plot(self.tgrid, np.zeros_like(self.tgrid), "-o", label="Ctrl S clásico", color="#1f77b4", linewidth=1.5, markersize=3)
        self.line_s_pinn, = self.ax_s.plot(self.tgrid, np.zeros_like(self.tgrid), "--o", label="Ctrl S PINN", color="#ff7f0e", linewidth=1.5, markersize=3)
        self.line_c_err, = self.ax_c_err.plot(self.tgrid, np.zeros_like(self.tgrid), label="|C_pinn - C_clásico|", color="#d62728", linewidth=1.6)
        self.line_s_err, = self.ax_s_err.plot(self.tgrid, np.zeros_like(self.tgrid), label="|S_pinn - S_clásico|", color="#9467bd", linewidth=1.6)

        self.ax_j.set_xlabel("Iteración")
        self.ax_j.set_ylabel("J")
        self.ax_j.set_title("Funcional de costo: PINN vs clásico")
        self.ax_j.ticklabel_format(axis="y", style="plain", useOffset=False)

        self.ax_j_err.set_xlabel("Iteración")
        self.ax_j_err.set_ylabel("|ΔJ|")
        self.ax_j_err.set_title("Error funcional")

        self.ax_c.set_ylabel("C")
        self.ax_s.set_ylabel("S")
        self.ax_s.set_xlabel("t")
        self.ax_c_err.set_ylabel("|ΔC|")
        self.ax_s_err.set_ylabel("|ΔS|")
        self.ax_c_err.set_xlabel("t")
        self.ax_s_err.set_xlabel("t")
        self.ax_c.set_ylim(-0.05, 1.05)
        self.ax_s.set_ylim(-0.05, 1.05)

        for ax in [self.ax_j, self.ax_j_err, self.ax_c, self.ax_s, self.ax_c_err, self.ax_s_err]:
            ax.grid(True, alpha=0.35)
            ax.legend()

        self.control_title = self.fig.text(0.5, 0.975, "Progress of optimal control", ha="center", va="center")
        self.control_status = self.fig.text(0.01, 0.06, "Esperando selección...", ha="left", va="bottom")
        self.stop_text = self.ax_j.text(0.99, 0.97, "Stopping: PINN=pend. | solve=pend.", transform=self.ax_j.transAxes, ha="right", va="top", bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9})

        self.register_page(
            "control",
            [self.control_panel_ax, *self.control_widget_axes.values(), self.ax_j, self.ax_j_err, self.ax_c, self.ax_s, self.ax_c_err, self.ax_s_err],
            [self.control_title, self.control_status])

    # ====================================================================================================
    # =============================== Pagina del estado ==================================================
    # ====================================================================================================

    """ Crea las graficas del estado y de su entrenamiento """
    def build_state_page(self):
        self.ax_state_var1 = self.fig.add_axes([0.08, 0.58, 0.38, 0.28])
        self.ax_state_var2 = self.fig.add_axes([0.54, 0.58, 0.38, 0.28])
        self.ax_state_err = self.fig.add_axes([0.08, 0.20, 0.38, 0.28])
        self.ax_state_loss = self.fig.add_axes([0.54, 0.20, 0.38, 0.28])

        # Comparacion de las variables de estado y de los errores absolutos.
        self.line_state_var1_ref, = self.ax_state_var1.plot(self.tgrid, np.zeros_like(self.tgrid), label="u clásico", color="#1f77b4", linewidth=1.6)
        self.line_state_var1_pinn, = self.ax_state_var1.plot(self.tgrid, np.zeros_like(self.tgrid), "--", label="u PINN", color="#ff7f0e", linewidth=1.6)
        self.line_state_var2_ref, = self.ax_state_var2.plot(self.tgrid, np.zeros_like(self.tgrid), label="sigma clásico", color="#1f77b4", linewidth=1.6)
        self.line_state_var2_pinn, = self.ax_state_var2.plot(self.tgrid, np.zeros_like(self.tgrid), "--", label="sigma PINN", color="#ff7f0e", linewidth=1.6)
        self.line_state_err_var1, = self.ax_state_err.plot(self.tgrid, np.zeros_like(self.tgrid), label="|error u|", color="#d62728", linewidth=1.6)
        self.line_state_err_var2, = self.ax_state_err.plot(self.tgrid, np.zeros_like(self.tgrid), label="|error sigma|", color="#9467bd", linewidth=1.6)

        # Historial completo y componentes de la perdida.
        self.line_state_loss_all, = self.ax_state_loss.plot([], [], "-", label="loss training", color="#2ca02c", alpha=0.45, linewidth=1.6)
        self.line_state_loss_components = []
        state_component_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f", "#e377c2"]
        for comp_idx, label in enumerate(self.state_loss_labels):
            color = state_component_colors[comp_idx % len(state_component_colors)]
            line_component, = self.ax_state_loss.plot([], [], "--", label=label, color=color, alpha=0.6, linewidth=1.0)
            self.line_state_loss_components.append(line_component)
        self.ax_state_loss.set_yscale("log")

        self.ax_state_var1.set_title("u(t): PINN vs clásico")
        self.ax_state_var2.set_title("sigma(t): PINN vs clásico")
        self.ax_state_err.set_title("Error")
        self.ax_state_loss.set_title("Loss de entrenamiento")
        self.ax_state_err.set_xlabel("t")
        self.ax_state_loss.set_xlabel("epoch")

        for ax in [self.ax_state_var1, self.ax_state_var2, self.ax_state_err, self.ax_state_loss]:
            ax.grid(True, alpha=0.35)
            ax.legend()

        self.state_title = self.fig.text(0.5, 0.91, "PINN Training [State]", ha="center", va="center")
        self.state_status = self.fig.text(0.01, 0.12, "Esperando selección...", ha="left", va="bottom")

        self.register_page("state", [self.ax_state_var1, self.ax_state_var2, self.ax_state_err, self.ax_state_loss], [self.state_title, self.state_status])

    # ====================================================================================================
    # =============================== Pagina del adjunto =================================================
    # ====================================================================================================

    """ Crea las graficas del adjunto y de su entrenamiento """
    def build_adjoint_page(self):
        self.ax_adj_var1 = self.fig.add_axes([0.08, 0.58, 0.38, 0.28])
        self.ax_adj_var2 = self.fig.add_axes([0.54, 0.58, 0.38, 0.28])
        self.ax_adj_err = self.fig.add_axes([0.08, 0.20, 0.38, 0.28])
        self.ax_adj_loss = self.fig.add_axes([0.54, 0.20, 0.38, 0.28])

        # Comparacion de las variables adjuntas y de los errores absolutos.
        self.line_adj_var1_ref, = self.ax_adj_var1.plot(self.tgrid, np.zeros_like(self.tgrid), label="w clásico", color="#1f77b4", linewidth=1.6)
        self.line_adj_var1_pinn, = self.ax_adj_var1.plot(self.tgrid, np.zeros_like(self.tgrid), "--", label="w PINN", color="#ff7f0e", linewidth=1.6)
        self.line_adj_var2_ref, = self.ax_adj_var2.plot(self.tgrid, np.zeros_like(self.tgrid), label="z clásico", color="#1f77b4", linewidth=1.6)
        self.line_adj_var2_pinn, = self.ax_adj_var2.plot(self.tgrid, np.zeros_like(self.tgrid), "--", label="z PINN", color="#ff7f0e", linewidth=1.6)
        self.line_adj_err_var1, = self.ax_adj_err.plot(self.tgrid, np.zeros_like(self.tgrid), label="|error w|", color="#d62728", linewidth=1.6)
        self.line_adj_err_var2, = self.ax_adj_err.plot(self.tgrid, np.zeros_like(self.tgrid), label="|error z|", color="#9467bd", linewidth=1.6)

        # Historial completo y componentes de la perdida.
        self.line_adj_loss_all, = self.ax_adj_loss.plot([], [], "-", label="loss training", color="#2ca02c", alpha=0.45, linewidth=1.6)
        self.line_adj_loss_components = []
        adj_component_colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f", "#e377c2"]
        for comp_idx, label in enumerate(self.adjoint_loss_labels):
            color = adj_component_colors[comp_idx % len(adj_component_colors)]
            line_component, = self.ax_adj_loss.plot([], [], "--", label=label, color=color, alpha=0.6, linewidth=1.0)
            self.line_adj_loss_components.append(line_component)
        self.ax_adj_loss.set_yscale("log")

        self.ax_adj_var1.set_title("w(t): PINN vs clásico")
        self.ax_adj_var2.set_title("z(t): PINN vs clásico")
        self.ax_adj_err.set_title("Error")
        self.ax_adj_loss.set_title("Loss de entrenamiento")
        self.ax_adj_err.set_xlabel("t")
        self.ax_adj_loss.set_xlabel("epoch")

        self.ax_adj_var1.set_xlim(float(np.max(self.tgrid)), float(np.min(self.tgrid)))
        self.ax_adj_var2.set_xlim(float(np.max(self.tgrid)), float(np.min(self.tgrid)))
        self.ax_adj_err.set_xlim(float(np.max(self.tgrid)), float(np.min(self.tgrid)))

        for ax in [self.ax_adj_var1, self.ax_adj_var2, self.ax_adj_err, self.ax_adj_loss]:
            ax.grid(True, alpha=0.35)
            ax.legend()

        self.adjoint_title = self.fig.text(0.5, 0.91, "PINN Training [Adjoint]", ha="center", va="center")
        self.adjoint_status = self.fig.text(0.01, 0.12, "Esperando selección...", ha="left", va="bottom")

        self.register_page("adjoint", [self.ax_adj_var1, self.ax_adj_var2, self.ax_adj_err, self.ax_adj_loss], [self.adjoint_title, self.adjoint_status])

    # ====================================================================================================
    # =============================== Seleccion de iteraciones ===========================================
    # ====================================================================================================

    """ Selecciona una iteracion y actualiza las tres paginas del monitor """
    def select_iteration(self, idx):
        idx = int(idx)
        if idx < 0:
            idx = 0
        if idx >= len(self.iterations):
            idx = len(self.iterations) - 1

        self.selected_index = idx
        iteration = self.iterations[idx]

        j_pinn = np.asarray(self.control_data["J_pinn"]).reshape(-1)
        j_classic = np.asarray(self.control_data["J_classic"]).reshape(-1)
        j_error = np.abs(j_pinn - j_classic)
        x_all = np.arange(1, len(j_pinn) + 1)

        self.line_j_pinn.set_data(x_all, j_pinn)
        self.line_j_classic.set_data(x_all, j_classic)
        self.line_j_err.set_data(x_all, j_error)

        x_selected = idx + 1
        self.line_j_sel_pinn.set_data([x_selected], [float(j_pinn[idx])])
        self.line_j_sel_classic.set_data([x_selected], [float(j_classic[idx])])
        self.line_j_sel_err.set_data([x_selected], [float(j_error[idx])])

        c_pinn = np.asarray(self.control_data["C_pinn"][idx]).reshape(-1)
        s_pinn = np.asarray(self.control_data["S_pinn"][idx]).reshape(-1)
        c_classic = np.asarray(self.control_data["C_classic"][idx]).reshape(-1)
        s_classic = np.asarray(self.control_data["S_classic"][idx]).reshape(-1)

        self.line_c_pinn.set_ydata(c_pinn)
        self.line_s_pinn.set_ydata(s_pinn)
        self.line_c_classic.set_ydata(c_classic)
        self.line_s_classic.set_ydata(s_classic)
        self.line_c_err.set_ydata(np.abs(c_pinn - c_classic))
        self.line_s_err.set_ydata(np.abs(s_pinn - s_classic))

        stop_pinn = self.run_info.get("pinn_stop_iteration", "pend.")
        stop_classic = self.run_info.get("classic_stop_iteration", "pend.")
        self.stop_text.set_text(f"Stopping: PINN={stop_pinn} | solve={stop_classic}")

        intc_value = float(np.asarray(self.control_data["IntC_pinn"]).reshape(-1)[idx])
        self.control_status.set_text(f"Iteración {iteration} | J_pinn={j_pinn[idx]:.6f} | J_clásico={j_classic[idx]:.6f} | "f"|ΔJ|={j_error[idx]:.3e} | IntC={intc_value:.4f}")

        self.update_state_iteration(iteration, idx)
        self.update_adjoint_iteration(iteration, idx)

        for ax in [self.ax_j, self.ax_j_err, self.ax_c, self.ax_s, self.ax_c_err, self.ax_s_err]:
            ax.relim()
            ax.autoscale_view()

        self.refresh()

    # ====================================================================================================
    # =============================== Controles de reproduccion ==========================================
    # ====================================================================================================

    """ Convierte la velocidad del slider en el intervalo utilizado por el timer """
    def playback_interval_ms(self):
        factor = max(0.5, min(3.0, float(self.play_speed)))
        iterations_per_second = 1.0 + (factor - 0.5) * (12.0 - 1.0) / (3.0 - 0.5)
        return int(1000.0 / iterations_per_second)

    """ Avanza una iteracion mientras el boton se encuentra en modo Play """
    def playback_step(self):
        if not self.playing:
            return

        if self.selected_index is None:
            next_index = 0
        else:
            next_index = (self.selected_index + 1) % len(self.iterations)
        self.select_iteration(next_index)

    """ Cambia entre Play y Pausa """
    def toggle_playback(self, event=None):
        if self.playing:
            self.playing = False
            self.play_button.label.set_text("Play")
            if self.play_timer is not None:
                try:
                    self.play_timer.stop()
                except Exception:
                    pass
        else:
            self.playing = True
            self.play_button.label.set_text("Pausa")
            if self.play_timer is None:
                self.play_timer = self.fig.canvas.new_timer(interval=self.playback_interval_ms())
                self.play_timer.add_callback(self.playback_step)
            self.play_timer.interval = self.playback_interval_ms()
            self.play_timer.start()
            self.playback_step()
        self.refresh()

    """ Actualiza la velocidad de reproduccion """
    def change_playback_speed(self, value):
        self.play_speed = float(value)
        if self.play_timer is not None:
            self.play_timer.interval = self.playback_interval_ms()
            if self.playing:
                self.play_timer.stop()
                self.play_timer.start()

    """ Selecciona la iteracion escrita en el panel de controles """
    def submit_iteration(self, text):
        try:
            iteration = int(text.strip())
        except ValueError:
            self.control_status.set_text("Escribe un numero de iteracion valido.")
            self.refresh()
            return

        if iteration not in self.iterations:
            self.control_status.set_text(f"Iteración {iteration}: no existe en esta simulación")
            self.refresh()
            return

        self.select_iteration(self.iterations.index(iteration))

    # ====================================================================================================
    # =============================== Datos de entrenamiento =============================================
    # ====================================================================================================

    """ Carga la informacion del entrenamiento del estado para una iteracion """
    def update_state_iteration(self, iteration, idx):
        source_iteration, full_loss = self.resolve_training_source_iteration(iteration, self.state_loss_by_iteration)
        component_loss = self.state_loss_components_by_iteration.get(source_iteration, np.empty((0, 0), dtype=np.float64))

        if full_loss is not None and np.asarray(full_loss).size > 0:
            full_loss = np.asarray(full_loss)
            epochs_all = np.arange(1, full_loss.size + 1)
            self.line_state_loss_all.set_data(epochs_all, np.maximum(full_loss, 1e-30))
        else:
            self.line_state_loss_all.set_data([], [])
        self.set_component_loss_lines(self.line_state_loss_components, self.state_loss_component_indices, component_loss)

        var1_pred = np.asarray(self.control_data["u_pinn"][idx]).reshape(-1)
        var2_pred = np.asarray(self.control_data["sigma_pinn"][idx]).reshape(-1)
        var1_ref = np.asarray(self.control_data["u_classic"][idx]).reshape(-1)
        var2_ref = np.asarray(self.control_data["sigma_classic"][idx]).reshape(-1)
        err_var1 = np.abs(var1_pred - var1_ref)
        err_var2 = np.abs(var2_pred - var2_ref)

        self.line_state_var1_ref.set_ydata(var1_ref)
        self.line_state_var2_ref.set_ydata(var2_ref)
        self.line_state_var1_pinn.set_ydata(var1_pred)
        self.line_state_var2_pinn.set_ydata(var2_pred)
        self.line_state_err_var1.set_ydata(err_var1)
        self.line_state_err_var2.set_ydata(err_var2)

        self.state_status.set_text(f"State iter {iteration} | aproximación final | " f"max err u={self._safe_nanmax(err_var1):.3e} | " f"max err sigma={self._safe_nanmax(err_var2):.3e}")
        if source_iteration != iteration:
            self.state_status.set_text(self.state_status.get_text() + f" | pérdidas de entrenamiento de iter {source_iteration}")

        for ax in [self.ax_state_var1, self.ax_state_var2, self.ax_state_err, self.ax_state_loss]:
            ax.relim()
            ax.autoscale_view()

    """ Carga la informacion del entrenamiento adjunto para una iteracion """
    def update_adjoint_iteration(self, iteration, idx):
        source_iteration, full_loss = self.resolve_training_source_iteration(iteration, self.adjoint_loss_by_iteration)
        component_loss = self.adjoint_loss_components_by_iteration.get(source_iteration, np.empty((0, 0), dtype=np.float64))

        if full_loss is not None and np.asarray(full_loss).size > 0:
            full_loss = np.asarray(full_loss)
            epochs_all = np.arange(1, full_loss.size + 1)
            self.line_adj_loss_all.set_data(epochs_all, np.maximum(full_loss, 1e-30))
        else:
            self.line_adj_loss_all.set_data([], [])
        self.set_component_loss_lines(self.line_adj_loss_components, self.adjoint_loss_component_indices, component_loss)

        var1_pred = np.asarray(self.control_data["w_pinn"][idx]).reshape(-1)
        var2_pred = np.asarray(self.control_data["z_pinn"][idx]).reshape(-1)
        var1_ref = np.asarray(self.control_data["w_classic"][idx]).reshape(-1)
        var2_ref = np.asarray(self.control_data["z_classic"][idx]).reshape(-1)
        err_var1 = np.abs(var1_pred - var1_ref)
        err_var2 = np.abs(var2_pred - var2_ref)

        self.line_adj_var1_ref.set_ydata(var1_ref)
        self.line_adj_var2_ref.set_ydata(var2_ref)
        self.line_adj_var1_pinn.set_ydata(var1_pred)
        self.line_adj_var2_pinn.set_ydata(var2_pred)
        self.line_adj_err_var1.set_ydata(err_var1)
        self.line_adj_err_var2.set_ydata(err_var2)

        self.adjoint_status.set_text(f"Adjoint iter {iteration} | aproximación final | " f"max err w={self._safe_nanmax(err_var1):.3e} | " f"max err z={self._safe_nanmax(err_var2):.3e}")
        if source_iteration != iteration:
            self.adjoint_status.set_text(self.adjoint_status.get_text() + f" | pérdidas de entrenamiento de iter {source_iteration}")

        for ax in [self.ax_adj_var1, self.ax_adj_var2, self.ax_adj_err, self.ax_adj_loss]:
            ax.relim()
            ax.autoscale_view()

        self.ax_adj_var1.set_xlim(float(np.max(self.tgrid)), float(np.min(self.tgrid)))
        self.ax_adj_var2.set_xlim(float(np.max(self.tgrid)), float(np.min(self.tgrid)))
        self.ax_adj_err.set_xlim(float(np.max(self.tgrid)), float(np.min(self.tgrid)))

    """ Calcula el maximo ignorando valores que todavia no son finitos """
    @staticmethod
    def _safe_nanmax(values):
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.max(finite))

    # ====================================================================================================
    # =============================== Eventos de la ventana ==============================================
    # ====================================================================================================

    """ Responde a los clicks sobre las curvas del funcional """
    def on_pick(self, event):
        if event.artist in [self.line_j_pinn, self.line_j_classic, self.line_j_err]:
            if len(event.ind) == 0:
                return
            idx = int(event.ind[-1])
            self.select_iteration(idx)
            self.show_page("control")
            return

    """ Detiene el timer cuando se cierra la ventana """
    def on_close(self, event):
        self.playing = False
        if self.play_timer is not None:
            try:
                self.play_timer.stop()
            except Exception:
                pass
        self.play_timer = None

    """ Redibuja la ventana despues de actualizar sus datos """
    def refresh(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    """ Mantiene abierta la ventana hasta que el usuario la cierre """
    def run(self):
        plt.ioff()
        plt.show(block=True)


# ====================================================================================================
# =============================== Main ================================================================
# ====================================================================================================

""" Lee los argumentos y abre el monitor """
def main():
    argument_parser = argparse.ArgumentParser(description="Viewer de simulaciones")
    argument_parser.add_argument("simulation", type=str, help="Nombre de carpeta en results o ruta completa")
    arguments = argument_parser.parse_args()

    app = MonitorApp(arguments.simulation)
    app.run()


if __name__ == "__main__":
    main()
