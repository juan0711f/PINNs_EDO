import json
from datetime import datetime
from pathlib import Path

import numpy as np
import params

class saveSimulations:
    def __init__(self, results_folder="results"):
        self.simulation = params.simulation
        self.save_every_n_epochs = params.save_every_n_epochs
        self.t_grid = params.t_grid
        self.max_iter_optimo = params.max_iter_optimo
        self.created_at = datetime.now().isoformat(timespec="seconds")


        # Carpeta de resultados. Si existe la carpeta "params.simulation2, se creará una nueva carpeta cuyo nombre incluirá la fecha y la hora.
        root = Path(results_folder)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root/self.simulation
        if run_dir.exists():
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = root / f"{self.simulation}__{time}"
        run_dir.mkdir(parents=True, exist_ok=False)

        # Carpetas y archivos que se van a guardar.
        self.run_dir = run_dir
        self.run_info_path = self.run_dir / "run_info.json"
        self.control_history_path = self.run_dir / "control_history.npz"

        self.state_training_dir = self.run_dir / "state_training"
        self.adjoint_training_dir = self.run_dir / "adjoint_training"
        self.state_checkpoints_dir = self.run_dir / "state_checkpoints"
        self.adjoint_checkpoints_dir = self.run_dir / "adjoint_checkpoints"

        for folder in (self.state_training_dir, self.adjoint_training_dir, self.state_checkpoints_dir, self.adjoint_checkpoints_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self.saved_epochs = {"state": {}, "adjoint": {}}
        self.iteration_start_time = {}
        self.control_rows = []

        self.write_run_info()
    
    # Crea el archivo de información con los datos iniciales
    def build_network_info(self):
        state_info = {
            "hidden_layers": params.state_hidden_depth,
            "hidden_width": params.state_hidden_width,
            "subintervals": params.state_subintervals,
            "collocation_points_per_subinterval": params.state_n_collocation,
            "epochs": params.state_epochs,
        }

        return {
            "networks": {
                "state": state_info,
                "adjoint": {
                    "hidden_layers": params.adj_hidden_depth,
                    "hidden_width": params.adj_hidden_width,
                    "subintervals": params.adj_subintervals,
                    "collocation_points_per_subinterval": params.adj_n_collocation,
                    "epochs": params.adj_epochs,
                },
            }
        }

    def write_run_info(self):
        info = {
            "simulation": self.simulation,
            "created_at": self.created_at,
            "run_directory": str(self.run_dir),
            "save_every_n_epochs": self.save_every_n_epochs,
            "max_iter_optimo": self.max_iter_optimo,
            "n_time_points": self.t_grid.size,
            "t_grid": self.t_grid.tolist(),
        }
        info.update(self.build_network_info())
        with self.run_info_path.open("w", encoding="utf-8") as file:
            json.dump(info, file, indent=2, ensure_ascii=False)
    
    # Registra la hora de inicio de cada iteración
    def register_control_iteration_start(self, iteration):
        self.iteration_start_time[iteration] = datetime.now().isoformat(timespec="seconds")
    
    # Guardar el punto de control de "params.save_every_n_epochs" en cada sesión de entrenamiento de PINNs
    def save_checkpoint(self, problem_type, control_iteration, epoch, total_epochs, loss_value, var1_pred, var2_pred, var1_ref, var2_ref, J_epoch=np.nan, is_best_epoch=False):

        # El archivo se guarda en una carpeta denominada "Iter_[número de iteraciones]" dentro de la carpeta del "punto de control de época" correspondiente a cada sistema.
        checkpoint_dir = self.state_checkpoints_dir if problem_type == "state" else self.adjoint_checkpoints_dir
        iter_dir = checkpoint_dir / f"iter_{control_iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        saved_epochs = self.saved_epochs[problem_type].setdefault(control_iteration, [])

        if epoch not in saved_epochs:
            saved_epochs.append(epoch)

        out_path = iter_dir / f"epoch_{epoch:06d}.npz"
        if problem_type == "state":
            np.savez_compressed(out_path, t_grid=self.t_grid, epoch=epoch, total_epochs=total_epochs, loss_epoch=loss_value, J_epoch=J_epoch, is_best_epoch=is_best_epoch, u_pinn=var1_pred, sigma_pinn=var2_pred, u_ref=var1_ref, sigma_ref=var2_ref)
        else:
            np.savez_compressed(out_path, t_grid=self.t_grid, epoch=epoch, total_epochs=total_epochs, loss_epoch=loss_value, J_epoch=J_epoch, is_best_epoch=is_best_epoch, w_pinn=var1_pred, z_pinn=var2_pred, w_ref=var1_ref, z_ref=var2_ref)

    def finalize_training(self, problem_type, control_iteration, loss_history):
        out_dir = self.state_training_dir if problem_type == "state" else self.adjoint_training_dir
        out_path = out_dir / f"iter_{control_iteration:03d}.npz"

        saved_epochs = sorted(set(self.saved_epochs[problem_type].get(control_iteration, [])))

        np.savez_compressed(out_path, loss_history=loss_history, epochs_saved=saved_epochs, save_every_n_epochs=self.save_every_n_epochs)

    def register_control_iteration_end(self, iteration, J_pinn, J_classic, J_terms_pinn, J_terms_classic, Cvec_pinn, Svec_pinn, uVec_pinn, sVec_pinn, wVec_pinn, zVec_pinn, Cvec_classic, Svec_classic, uVec_classic, sVec_classic, wVec_classic, zVec_classic, IntC_pinn, IntC_classic):
        info = {
            "iteration": iteration,
            "started_at": self.iteration_start_time.get(iteration, ""),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "J_pinn": J_pinn,
            "J_classic": J_classic,
            "J1_pinn": J_terms_pinn.get("J1", np.nan),
            "J2_pinn": J_terms_pinn.get("J2", np.nan),
            "J3_pinn": J_terms_pinn.get("J3", np.nan),
            "J4_pinn": J_terms_pinn.get("J4", np.nan),
            "J5_pinn": J_terms_pinn.get("J5", np.nan),
            "J6_pinn": J_terms_pinn.get("J6", np.nan),
            "J1_classic": J_terms_classic.get("J1", np.nan),
            "J2_classic": J_terms_classic.get("J2", np.nan),
            "J3_classic": J_terms_classic.get("J3", np.nan),
            "J4_classic": J_terms_classic.get("J4", np.nan),
            "J5_classic": J_terms_classic.get("J5", np.nan),
            "J6_classic": J_terms_classic.get("J6", np.nan),
            "IntC_pinn": IntC_pinn,
            "IntC_classic": IntC_classic,
            "C_pinn": Cvec_pinn,
            "S_pinn": Svec_pinn,
            "u_pinn": uVec_pinn,
            "sigma_pinn": sVec_pinn,
            "w_pinn": wVec_pinn,
            "z_pinn": zVec_pinn,
            "C_classic": Cvec_classic,
            "S_classic": Svec_classic,
            "u_classic": uVec_classic,
            "sigma_classic": sVec_classic,
            "w_classic": wVec_classic,
            "z_classic": zVec_classic,
        }
        self.control_rows.append(info)
        self.write_control_history()
    
    def build_control_history(self):
        iterations = [row["iteration"] for row in self.control_rows]
        started_at = [row["started_at"] for row in self.control_rows]
        finished_at = [row["finished_at"] for row in self.control_rows]

        return {
            "t_grid": self.t_grid,
            "iterations": iterations,
            "started_at": started_at,
            "finished_at": finished_at,
            "J_pinn": [row["J_pinn"] for row in self.control_rows],
            "J_classic": [row["J_classic"] for row in self.control_rows],
            "J1_pinn": [row["J1_pinn"] for row in self.control_rows],
            "J2_pinn": [row["J2_pinn"] for row in self.control_rows],
            "J3_pinn": [row["J3_pinn"] for row in self.control_rows],
            "J4_pinn": [row["J4_pinn"] for row in self.control_rows],
            "J5_pinn": [row["J5_pinn"] for row in self.control_rows],
            "J6_pinn": [row["J6_pinn"] for row in self.control_rows],
            "J1_classic": [row["J1_classic"] for row in self.control_rows],
            "J2_classic": [row["J2_classic"] for row in self.control_rows],
            "J3_classic": [row["J3_classic"] for row in self.control_rows],
            "J4_classic": [row["J4_classic"] for row in self.control_rows],
            "J5_classic": [row["J5_classic"] for row in self.control_rows],
            "J6_classic": [row["J6_classic"] for row in self.control_rows],
            "IntC_pinn": [row["IntC_pinn"] for row in self.control_rows],
            "IntC_classic": [row["IntC_classic"] for row in self.control_rows],
            "C_pinn": np.stack([row["C_pinn"] for row in self.control_rows], axis=0),
            "S_pinn": np.stack([row["S_pinn"] for row in self.control_rows], axis=0),
            "u_pinn": np.stack([row["u_pinn"] for row in self.control_rows], axis=0),
            "sigma_pinn": np.stack([row["sigma_pinn"] for row in self.control_rows], axis=0),
            "w_pinn": np.stack([row["w_pinn"] for row in self.control_rows], axis=0),
            "z_pinn": np.stack([row["z_pinn"] for row in self.control_rows], axis=0),
            "C_classic": np.stack([row["C_classic"] for row in self.control_rows], axis=0),
            "S_classic": np.stack([row["S_classic"] for row in self.control_rows], axis=0),
            "u_classic": np.stack([row["u_classic"] for row in self.control_rows], axis=0),
            "sigma_classic": np.stack([row["sigma_classic"] for row in self.control_rows], axis=0),
            "w_classic": np.stack([row["w_classic"] for row in self.control_rows], axis=0),
            "z_classic": np.stack([row["z_classic"] for row in self.control_rows], axis=0),
        }

    def write_control_history(self):
        if not self.control_rows:
            return

        payload = self.build_control_history()
        np.savez_compressed(self.control_history_path, **payload)


# caffeinate -dimsu perl -MTime::HiRes=time -e 'my($u0,$s0,$cu0,$cs0)=times; my $t0=time; my $rc=system @ARGV; my($u1,$s1,$cu1,$cs1)=times; my $cpu=($cu1-$cu0)+($cs1-$cs0); my $real=time-$t0; printf "CPU_TOTAL=%.3f s (%.0f ms)\nREAL=%.3f s (%.0f ms)\n",$cpu,$cpu*1000,$real,$real*1000; exit($rc==-1?1:($rc&127?128+($rc&127):$rc>>8));' python3 main.py --simulation HCS_n1 --l1 10 --l2 10  --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 4 --state-epochs 10000 --adj-subintervals 4 --adj-epochs 10000
# caffeinate -dimsu perl -MTime::HiRes=time -e 'my($u0,$s0,$cu0,$cs0)=times; my $t0=time; my $rc=system @ARGV; my($u1,$s1,$cu1,$cs1)=times; my $cpu=($cu1-$cu0)+($cs1-$cs0); my $real=time-$t0; printf "CPU_TOTAL=%.3f s (%.0f ms)\nREAL=%.3f s (%.0f ms)\n",$cpu,$cpu*1000,$real,$real*1000; exit($rc==-1?1:($rc&127?128+($rc&127):$rc>>8));' python3 main.py --simulation SCS_1sub --l1 10 --l2 10  --state-pinn-mode SCS_PINNs --adj-pinn-mode SCS_PINNs --state-subintervals 1 --state-epochs 40000 --adj-subintervals 1 --adj-epochs 40000 
# caffeinate -dimsu perl -MTime::HiRes=time -e 'my($u0,$s0,$cu0,$cs0)=times; my $t0=time; my $rc=system @ARGV; my($u1,$s1,$cu1,$cs1)=times; my $cpu=($cu1-$cu0)+($cs1-$cs0); my $real=time-$t0; printf "CPU_TOTAL=%.3f s (%.0f ms)\nREAL=%.3f s (%.0f ms)\n",$cpu,$cpu*1000,$real,$real*1000; exit($rc==-1?1:($rc&127?128+($rc&127):$rc>>8));' python3 main.py --simulation sinsu_HCS_4sub --l1 10 --l2 10  --state-pinn-mode HCS_PINNs --adj-pinn-mode HCS_PINNs --state-subintervals 4 --state-epochs 10000 --adj-subintervals 4 --adj-epochs 10000 --controls sinusoidal