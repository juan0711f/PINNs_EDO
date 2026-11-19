from cli import parse_args
import params

from tool.pinn_tools import set_seeds
from control_algorithm import control_algorithm
from save_simulations import saveSimulations

def main():
    args = parse_args()
    params.apply_cli_args(args)

    set_seeds(98765)

    save_folder = saveSimulations()

    control_algorithm(save_folder=save_folder)

if __name__ == "__main__":
    main()
