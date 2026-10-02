"""Knapsack utilities and QUBO mapping helpers.

This module provides helper functions to evaluate knapsack objective and
penalty terms and a mapper that converts a knapsack instance into an Ising
Hamiltonian (QUBO) using Qiskit Optimization. It also exposes a small
script-style entrypoint for local demonstrations.
"""

from dataclasses import asdict, dataclass, is_dataclass
from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.applications import Knapsack
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from scipy.optimize import minimize, OptimizeResult

import logging
from argparse import ArgumentParser


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEED = None  # Global seed for reproducibility, can be set from command-line arguments
OBJECTIVE_FUNC_VALUES = []


@dataclass
class KnapsackInstance:
    """Representation of a knapsack problem instance.

    Attributes:
        values: List of integer values (profits) for each item.
        weights: List of integer weights for each item. Must be same length as
            ``values``.
        capacity: Maximum allowed total weight (knapsack capacity).
    """

    values: list[int]
    weights: list[int]
    capacity: int


def cost_function(x: list[int], knapsack_instance: KnapsackInstance) -> float:
    """Compute the (negative) total value for a binary selection vector.

    The function returns the negative sum of selected item values so that a
    minimizer will prefer selections with larger total value.

    Args:
        x: Binary list where 1 indicates the item is selected and 0 otherwise.
        knapsack_instance: The knapsack instance containing item values and capacity.

    Returns:
        The cost as a float (negative total value).
    """
    cost = 0
    for i in range(len(x)):
        cost -= x[i] * knapsack_instance.values[i]

    return cost


def penalty_function(
    x: list[int], knapsack_instance: KnapsackInstance, penalty_value: float = None
) -> float:
    """Compute a quadratic penalty for capacity violation.

    The penalty is applied when the total weight of the selected items
    exceeds the knapsack capacity. It uses a quadratic term scaled by
    ``penalty_value`` to heavily penalize infeasible selections.

    Args:
        x: Binary list where 1 indicates the item is selected.
        knapsack_instance: The knapsack instance containing item weights and capacity.
        penalty_value: Scaling factor for the penalty (defaults to sum of
            global `VALUES`).

    Returns:
        The penalty term as a float.
    """
    if penalty_value is None:
        penalty_value = sum(knapsack_instance.values)

    penalty = penalty_value * np.square(
        max(
            0,
            sum(x[i] * knapsack_instance.weights[i] for i in range(len(x)))
            - knapsack_instance.capacity,
        )
    )

    return penalty


def objective_function(x: list[int], knapsack_instance: KnapsackInstance) -> float:
    """Compute the combined objective (cost + penalty) for a selection.

    Args:
        x: Binary selection list.
        knapsack_instance: The knapsack instance containing item values and capacity.

    Returns:
        The scalar objective value (cost + penalty).
    """
    return cost_function(x, knapsack_instance) + penalty_function(x, knapsack_instance)


def bruteforce(knapsack_instance: KnapsackInstance) -> tuple[list[int], float]:
    """Brute-force search for the optimal selection vector.

    This helper function exhaustively evaluates all possible binary selection
    vectors and returns the one with the lowest objective function value.

    Args:
        knapsack_instance: The knapsack instance to solve.

    Returns:
        The optimal binary selection vector as a list of integers (0 or 1) and its corresponding objective function value.
    """
    bestValue = float("inf")
    bestArray = []

    for test in itertools.product([0, 1], repeat=len(knapsack_instance.values)):
        test = list(test)
        newValue = objective_function(test, knapsack_instance)
        if newValue < bestValue:
            bestValue = newValue
            bestArray = test.copy()

    return bestArray, bestValue


def map_hamiltonian(
    knapsack_instance: KnapsackInstance, penalty_factor: float
) -> tuple[SparsePauliOp, float]:
    """Convert a `KnapsackInstance` into an Ising Hamiltonian.

    This helper constructs a Qiskit Optimization `Knapsack` application
    model, converts it to a `QuadraticProgram`, maps it to a QUBO and then
    returns the Ising representation (a `SparsePauliOp`) together with the
    associated energy offset.

    Args:
        knapsack_instance: The knapsack instance to convert.
        penalty_factor: The scaling factor for the penalty term.

    Returns:
        A tuple ``(hamiltonian, offset)`` where ``hamiltonian`` is a
        `qiskit.quantum_info.SparsePauliOp` describing the Ising operator and
        ``offset`` is the numeric energy offset (float).
    """
    knapsack = Knapsack(
        values=knapsack_instance.values,
        weights=knapsack_instance.weights,
        max_weight=knapsack_instance.capacity,
    )

    qprog: QuadraticProgram = knapsack.to_quadratic_program()
    qubo = QuadraticProgramToQubo(penalty=penalty_factor).convert(qprog)

    return qubo.to_ising()


def pstate(
    knapsack_instance: KnapsackInstance, hamiltonian: SparsePauliOp
) -> QuantumCircuit:
    """Construct a quantum circuit that prepares a reference state from a
    `KnapsackInstance`.

    Args:
        knapsack_instance: The knapsack instance to create the state for.
        hamiltonian: The Ising Hamiltonian associated with the problem.

    Returns:
        A `QuantumCircuit` that prepares a reference state (e.g., all zeros).
    """
    num_qubits = hamiltonian.num_qubits
    circuit = QuantumCircuit(num_qubits)

    def most_valuable_item():
        """Identify the index of the most valuable item that fits within capacity."""
        best_index = None
        best_value = -float("inf")
        for i in range(len(knapsack_instance.values)):
            if (
                knapsack_instance.weights[i] <= knapsack_instance.capacity
                and knapsack_instance.values[i] > best_value
            ):
                best_value = knapsack_instance.values[i]
                best_index = i
        return best_index

    index = most_valuable_item()
    if index is not None:
        circuit.x(index)  # Flip the qubit corresponding to the most valuable item

    return circuit


def decode_optimization_result(
    optimization_result: OptimizeResult,
    ansatz: QAOAAnsatz,
    hamiltonian: SparsePauliOp,
    estimator: BaseEstimatorV2,
    results: dict,
):
    """Decode the optimization result to extract the best solution and its value.

    This function takes the optimization result from scipy, constructs the corresponding quantum state using the ansatz and optimal parameters, and identifies the solution bitstring with the lowest energy. It then updates the results dictionary with the decoded solution and its value.

    Args:
        optimization_result: The result object returned by scipy.optimize.minimize.
        ansatz: The QAOA ansatz circuit used for the optimization.
        hamiltonian: The cost Hamiltonian as a SparsePauliOp.
        estimator: A Qiskit Estimator instance for evaluating expectation values.
        results: The dictionary to update with the decoded solution and its value.
    """
    optimal_params = optimization_result.x
    optimal_circuit = ansatz.assign_parameters(optimal_params)

    optimal_state = Statevector.from_instruction(optimal_circuit)  # type: ignore
    probabilities = optimal_state.probabilities_dict()
    results["quantum"]["probabilities"] = probabilities


def _x0_parameters(num_params) -> np.ndarray:
    """Generate deterministic initial parameters for the optimizer.

    Args:
        num_params: Number of parameters to generate.

    Returns:
        np.ndarray: Seeded random initial parameter vector.
    """
    params = np.random.RandomState(seed=SEED).random(num_params)
    return params


def _json_default(value):
    """Convert common non-JSON types in experiment results to plain Python objects."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _qiskit_bits_inversion(probabilities, sort_keys: bool = True):
    """Invert Qiskit bitstring keys from q_{n-1}...q_0 to x_0...x_{n-1}.

    If a dict mapping bitstrings->values is provided, the mapping is
    updated in-place so keys become their reversed bitstrings and the same
    values are preserved. By default the resulting mapping is reordered so
    keys are sorted lexicographically. If a list of keys is provided, a new
    list of reversed (and optionally sorted) keys is returned.
    """
    if isinstance(probabilities, dict):
        new = {key[::-1]: value for key, value in probabilities.items()}
        if sort_keys:
            ordered = dict(sorted(new.items(), key=lambda kv: kv[0]))
        else:
            ordered = new
        probabilities.clear()
        probabilities.update(ordered)
        return probabilities

    if isinstance(probabilities, list):
        out = [key[::-1] for key in probabilities]
        return sorted(out) if sort_keys else out

    raise TypeError("_qiskit_bits_inversion expects a dict or list of strings")


def _save_results(results: dict):
    """Save the results dictionary to a JSON file.

    Args:
        results: The dictionary containing optimization results and metadata.
    """
    import json
    import datetime as dt
    from pathlib import Path

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not Path("results").exists():
        Path("results").mkdir()
    result_dirname = f"knapsack_{timestamp}"
    Path(f"results/{result_dirname}").mkdir()
    filename = f"knapsack_results_{timestamp}.json"
    with open(f"results/{result_dirname}/{filename}", "w") as f:
        json.dump(results, f, indent=4, default=_json_default)
    logger.info(f"Results saved to results/{result_dirname}/{filename}")

    def _plot_optimization_curve(
        objective_log: list[dict[str, float]], result_dirname: str, timestamp: str
    ) -> None:
        """Plot the optimization history of the objective function values and saves the figure."""
        plt.figure(figsize=(10, 6))
        plt.plot(
            [log["estimated_cost"] for log in objective_log],
        )
        plt.title(f"Optimization iterations (offset={results['quantum']['offset']:.2f})")
        plt.xlabel("Iteration")
        plt.ylabel("Estimated Cost")
        plt.grid()
        plot_filename = f"results/{result_dirname}/optimization_history_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150)
        logger.info(f"Optimization history plot saved to {plot_filename}")
        plt.close()

    def _plot_top_bitstrings(
        probabilities: dict[str, float],
        result_dirname: str,
        timestamp: str,
        plotname: str = "top_bitstrings",
        k: int = 30,
    ) -> None:
        """Plot the top-k most probable bitstrings and aggregate the rest as 'Other'."""
        top_k_bitstrings = {
            bitstring
            for bitstring, _ in sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:k]
        }

        labels: list[str] = []
        values: list[float] = []
        other_probability = 0.0

        for bitstring, probability in probabilities.items():
            if bitstring in top_k_bitstrings:
                labels.append(bitstring)
                values.append(probability)
            else:
                other_probability += probability

        if other_probability > 0:
            labels.append("Other")
            values.append(other_probability)

        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)
        plt.title(f"Top {k} bitstring probabilities")
        plt.xlabel("Bitstring")
        plt.ylabel("Probability")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()

        plot_filename = Path("results") / result_dirname / f"{plotname}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()

    def _plot_slack_distribution_for_top_info_bits(
        probabilities: dict[str, float],
        aggregated_probabilities: dict[str, float],
        result_dirname: str,
        timestamp: str,
        top_info_count: int = 5,
    ) -> None:
        """
        For the top info-bit assignments, plot the distribution over slack bits.

        Assumes bitstrings are ordered as:
            info bits followed by slack bits.
        """
        if not aggregated_probabilities:
            return

        n_info_bits = len(next(iter(aggregated_probabilities)))

        grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for bitstring, probability in probabilities.items():
            info_bits = bitstring[:n_info_bits]
            slack_bits = bitstring[n_info_bits:]
            grouped[info_bits][slack_bits] += probability

        top_info_bits = [
            info_bits
            for info_bits, _ in sorted(
                aggregated_probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_info_count]
        ]

        for info_bits in top_info_bits:
            slack_distribution = grouped.get(info_bits)

            if not slack_distribution:
                continue

            slack_items = sorted(
                slack_distribution.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            labels = [slack_bits for slack_bits, _ in slack_items]
            values = [prob for _, prob in slack_items]

            plt.figure(figsize=(10, 5))
            plt.bar(labels, values)

            plt.title(
                f"Slack distribution for info bits {info_bits} "
                f"(total probability = {aggregated_probabilities[info_bits]:.4f})"
            )
            plt.xlabel("Slack bits")
            plt.ylabel("Probability")
            plt.xticks(rotation=60, ha="right")
            plt.tight_layout()

            plot_filename = (
                Path("results")
                / result_dirname
                / f"slack_distribution_info_{info_bits}_{timestamp}.png"
            )

            plt.savefig(plot_filename, dpi=150)
            plt.close()

    _plot_optimization_curve(
        objective_log=results["quantum"]["optimization_log"].values(),
        result_dirname=result_dirname,
        timestamp=timestamp,
    )
    _plot_top_bitstrings(
        probabilities=results["quantum"]["probabilities"],
        result_dirname=result_dirname,
        timestamp=timestamp,
    )
    _plot_top_bitstrings(
        probabilities=results["quantum"]["aggregated_probabilities"],
        result_dirname=result_dirname,
        timestamp=timestamp,
        plotname="top_aggregated_bitstrings",
    )


def _aggregate_probabilities(
    probabilities: dict[str, float], bits_labels: list[str]
) -> dict[str, float]:
    """Aggregate probabilities of bitstrings that correspond to the same item selection.

    This function takes a dictionary of bitstring probabilities and a list of bit labels, and aggregates the probabilities of bitstrings that correspond to the same selection of items (ignoring slack variables). The resulting dictionary maps item selection bitstrings to their aggregated probabilities.

    Args:
        probabilities: A dictionary mapping bitstrings (e.g., '0011') to their probabilities.
        bits_labels: A list of bit labels corresponding to the qubits (e.g., ['x0', 'x1', 's0', 's1']).

    Returns:
        A dictionary mapping item selection bitstrings (e.g., '00', '01', '10', '11') to their aggregated probabilities.
    """
    item_bits_indices = [
        i for i, label in enumerate(bits_labels) if label.startswith("x")
    ]
    aggregated = {}
    for bitstring, prob in probabilities.items():
        item_selection = "".join(bitstring[i] for i in item_bits_indices)
        aggregated[item_selection] = aggregated.get(item_selection, 0) + prob
    return aggregated


def process_result(optimization_result: OptimizeResult, results: dict):
    """Process the optimization result and update the results dictionary.

    Args:
        optimization_result: The optimization result from scipy.optimize.minimize.
        results: The dictionary to update with quantum optimization results.
    """
    results["quantum"]["final_cost"] = optimization_result.fun
    results["quantum"]["optimal_parameters"] = optimization_result.x.tolist()
    results["quantum"]["probabilities"] = _qiskit_bits_inversion(
        results["quantum"]["probabilities"]
    )  # type: ignore
    results["quantum"]["aggregated_probabilities"] = _aggregate_probabilities(
        results["quantum"]["probabilities"], results["quantum"]["bits_labels"]
    )
    results["quantum"]["aggregated_solution"] = max(
        results["quantum"]["aggregated_probabilities"],
        key=results["quantum"]["aggregated_probabilities"].get,
    )
    results["quantum"]["unique_solution"] = max(
        results["quantum"]["probabilities"], key=results["quantum"]["probabilities"].get
    )

    _save_results(results)


def knapsack_solver(args):
    """Script entrypoint that demonstrates mapping and prints the Hamiltonian.

    This function creates a sample :class:`KnapsackInstance`, converts it to an
    Ising Hamiltonian using :func:`map_hamiltonian`, and prints the resulting
    operator. It is intended for quick local demonstrations and testing.
    """
    global SEED
    SEED = args.seed

    knapsack_instance = KnapsackInstance(
        values=[8, 10],
        weights=[4, 7],
        capacity=8,
    )

    results = {
        "metadata": {
            "seed": SEED,
            "optimizer": args.optimizer,
            "max_iterations": args.iterations,
            "qaoa_layers": args.qaoa_layers,
            "use_reference_state": args.use_reference_state,
            "penalty_scaling": args.penalty_scaling
            if args.penalty_scaling is not None
            else sum(knapsack_instance.values),
            "knapsack_instance": asdict(knapsack_instance),
        },
        "classical": {
            "solution": None,
            "value": None,
        },
        "quantum": {
            "offset": None,
            "final_cost": None,
            "aggregated_solution": None,
            "unique_solution": None,
            "optimal_parameters": None,
            "bits_labels": None,
            "aggregated_probabilities": None,
            "probabilities": None,
            "hamiltonian": None,
            "optimization_log": {},
        },
    }

    classical_solution, classical_value = bruteforce(knapsack_instance)
    logger.info(f"Classical solution: {classical_solution}, Value: {classical_value}")
    results["classical"]["solution"] = classical_solution
    results["classical"]["value"] = classical_value

    hamiltonian, offset = map_hamiltonian(
        knapsack_instance,
        penalty_factor=args.penalty_scaling,
    )
    results["quantum"]["hamiltonian"] = hamiltonian.to_list()
    results["quantum"]["offset"] = offset
    logger.info(f"Mapped Hamiltonian:\n{hamiltonian}\nOffset: {offset}")
    p_state = None

    if args.use_reference_state:
        p_state = pstate(knapsack_instance, hamiltonian)
    ansatz = QAOAAnsatz(hamiltonian, reps=args.qaoa_layers, initial_state=p_state)

    logger.info(f"Number of qubits: {ansatz.num_qubits}")
    logger.info(f"Initial parameters: {ansatz.parameters}")

    item_bits = [f"x{i}" for i in range(len(knapsack_instance.values))]
    slack_bits = [
        f"s{i}" for i in range(ansatz.num_qubits - len(knapsack_instance.values))
    ]
    all_bits = item_bits + slack_bits
    results["quantum"]["bits_labels"] = all_bits

    estimator = StatevectorEstimator(seed=SEED)

    initial_params = _x0_parameters(ansatz.num_parameters)

    def cost_function_estimator(
        params,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: BaseEstimatorV2,
    ) -> float:
        """Estimate the cost function value for given parameters.

        This helper function takes a parameter vector, constructs the corresponding
        quantum state using the provided ansatz, and estimates the expectation
        value of the cost Hamiltonian using the given estimator.

        Args:
            params: Parameter vector for the ansatz.
            ansatz: The QAOA ansatz circuit.
            hamiltonian: The cost Hamiltonian as a SparsePauliOp.
            estimator: A Qiskit Estimator instance for evaluating expectation values.

        Returns:
            The estimated cost function value as a float.
        """
        pub = (ansatz, [hamiltonian], [params])

        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs

        num_evaluations = len(results["quantum"]["optimization_log"]) + 1
        results["quantum"]["optimization_log"][num_evaluations] = {
            "parameters": params.tolist(),
            "estimated_cost": energy,
        }
        logger.info(
            f"Evaluation {num_evaluations}: Parameters: {params}, Estimated Cost: {energy}"
        )

        return energy

    optimization_result = minimize(
        cost_function_estimator,
        x0=initial_params,
        args=(ansatz, hamiltonian, estimator),
        method=args.optimizer,
        options={"maxiter": args.iterations, "disp": True},
    )
    decode_optimization_result(
        optimization_result, ansatz, hamiltonian, estimator, results
    )

    process_result(optimization_result, results)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Quantum resolution of the Knapsack problem using QAOA."
    )
    parser.add_argument(
        "--use-reference-state",
        action="store_true",
        help="Use a reference prepared state",
    )
    parser.add_argument(
        "--optimizer",
        choices=["COBYLA", "SLSQP", "ADAM"],
        default="COBYLA",
        help="Optimizer to use for parameter optimization",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations for the optimizer",
    )
    parser.add_argument(
        "--qaoa-layers",
        type=int,
        default=1,
        help="Number of QAOA layers (reps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--penalty-scaling",
        type=float,
        help="Scaling factor for the penalty",
    )

    args = parser.parse_args()
    knapsack_solver(args)
