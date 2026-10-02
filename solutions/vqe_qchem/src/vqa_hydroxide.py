import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize, OptimizeResult
from ham_sdk_conversion import map_pl_hamiltonian_to_qiskit
import plotting as plot
import pennylane as qml
import pprint as pp
import logging
from argparse import ArgumentParser

_PACKAGE_LOGGER_NAME = __package__ or __name__
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _configure_package_logger() -> logging.Logger:
    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(logging.INFO)

    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        package_logger.addHandler(handler)

    # Keep this logging setup local to this package logger.
    package_logger.propagate = False
    return package_logger


logger = _configure_package_logger()


def classical_diagonalization(hamiltonian: SparsePauliOp) -> float:
    """Classically diagonalize the Hamiltonian to find the ground state energy.

    Args:
        hamiltonian: Hamiltonian operator to diagonalize.

    Returns:
        float: Ground state energy of the Hamiltonian.
    """
    # Convert the SparsePauliOp to a dense matrix
    dense_matrix = hamiltonian.to_matrix()

    # Diagonalize the matrix to find eigenvalues
    eigenvalues, _ = np.linalg.eigh(dense_matrix)

    # The ground state energy is the minimum eigenvalue
    ground_state_energy = np.min(eigenvalues)

    return ground_state_energy

def vqe_loop(input_hamiltonian: SparsePauliOp, optimizer: str, iterations: int, ansatz_layers: int, seed: int) -> OptimizeResult:
    """Run a VQE optimization using only the most basic Qiskit components.

    Args:
        input_hamiltonian: Hamiltonian operator to minimize.
        optimizer: Optimizer to use for parameter optimization.
        iterations: Number of iterations for the optimizer.
        ansatz_layers: Number of ansatz layers (reps).
        seed: Random seed for reproducibility.

    Returns:
        scipy.optimize.OptimizeResult: Optimization result returned by ``minimize``.

    Example:
        >>> from hamiltonian import Hamiltonian
        >>> result = vqe_bare(Hamiltonian.H2_STO6G_REDUX.value)
    """

    # Define the ansatz
    ansatz = n_local(
        num_qubits = input_hamiltonian.num_qubits, # type: ignore
        rotation_blocks = ["rx", "rz"],
        entanglement_blocks = ["cx"],
        reps = ansatz_layers,
        entanglement = "linear",
    )

    # Build the Hamiltonian
    hamiltonian = input_hamiltonian

    # Set up the estimator
    estimator = StatevectorEstimator(seed=seed)

    # Define the cost function for optimization
    def cost_function(
        params, ansatz=ansatz, hamiltonian=hamiltonian, estimator=estimator
    ) -> float:
        pub_estimate = (
            ansatz,
            [hamiltonian],
            [params],
        )
        result = estimator.run(
            pubs=[pub_estimate],
        ).result()
        energy = result[0].data.evs[0]
        num_evaluations = len(plot.RESULTS_DICT["quantum"]["optimization_log"]) + 1
        plot.RESULTS_DICT["quantum"]["optimization_log"][num_evaluations] = {
            "parameters": params.tolist(),
            "estimated_cost": energy,
        }
        return energy

    # Run the optimization
    initial_params = np.random.RandomState(seed=seed).rand(ansatz.num_parameters)  # type: ignore

    optimization_result = minimize(
        cost_function,
        x0=initial_params,
        args=(ansatz, hamiltonian, estimator),
        method=optimizer,
        options={"maxiter": iterations},
        callback=lambda xk: logger.info(f"Current parameters: {xk}, Current energy: {cost_function(xk)}"),
    )

    return optimization_result

def hydroxide_solver(args):
    """Run the VQE optimization for the hydroxide ion (OH-).

    Args:
        args: Command-line arguments containing optimizer settings.

    Returns:
        scipy.optimize.OptimizeResult: Optimization result returned by ``minimize``.
    """
    hydroxide_dataset = qml.data.load("qchem", molname="OH-", bondlength=0.964, basis="STO-3G")

    logger.debug(f"Tapered Hamiltonian: {hydroxide_dataset[0].tapered_hamiltonian}")
    hamiltonian = map_pl_hamiltonian_to_qiskit(hydroxide_dataset[0].tapered_hamiltonian)
    plot.RESULTS_DICT["quantum"]["hamiltonian"] = hamiltonian.to_list()

    classical_energy = classical_diagonalization(hamiltonian)
    plot.RESULTS_DICT["classical"]["value"] = classical_energy

    logger.info(f"Classical diagonalization energy: {classical_energy:.6f} Hartree")

    # Run the VQE optimization
    results = vqe_loop(hamiltonian, optimizer=args.optimizer, iterations=args.iter, ansatz_layers=args.ansatz_layers, seed=args.seed)

    logger.info(f"VQE optimization result: {results}")

    plot.process_result(results, plot.RESULTS_DICT)



if __name__ == "__main__":
    # Parser for command-line arguments
    parser = ArgumentParser(description="Run VQE optimization for the hydroxide ion (OH-).")
    parser.add_argument(
        "--optimizer",
        choices=["COBYLA", "SLSQP", "ADAM"],
        default="COBYLA",
        help="Optimizer to use for parameter optimization",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help="Number of iterations for the optimizer",
    )
    parser.add_argument(
        "--ansatz-layers",
        type=int,
        default=1,
        help="Number of ansatz layers (reps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # save parameters in results dict
    plot.RESULTS_DICT["metadata"]["seed"] = args.seed
    plot.RESULTS_DICT["metadata"]["optimizer"] = args.optimizer
    plot.RESULTS_DICT["metadata"]["max_iterations"] = args.iter
    plot.RESULTS_DICT["metadata"]["qaoa_layers"] = args.ansatz_layers

    hydroxide_solver(args=args)