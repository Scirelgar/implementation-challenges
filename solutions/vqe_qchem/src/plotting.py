import json
import datetime as dt
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult
import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DICT = {
        "metadata": {
            "seed": None,
            "optimizer": None,
            "max_iterations": None,
            "qaoa_layers": None,
        },
        "classical": {
            "known_parameters": None,
            "value": None,
        },
        "quantum": {
            "energy": None,
            "optimal_parameters": None,
            "hamiltonian": None,
            "optimization_log": {},
        },
    }

def _plot_optimization_curve(
        objective_log: dict[int, dict[str, float]] | list[dict[str, float]], timestamp: str
    ) -> None:
    """Plot the optimization history of the objective function values and saves the figure."""
    if isinstance(objective_log, dict):
        objective_log = [objective_log[key] for key in sorted(objective_log)]

    plt.figure(figsize=(10, 6))
    plt.plot(
        [log["estimated_cost"] for log in objective_log],
    )
    plt.title(f"Optimization iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Cost")
    plt.grid()
    plot_filename = Path("results") / f"optimization_history_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150)
    logger.info(f"Optimization history plot saved to {plot_filename}")
    plt.close()

def _json_default(value):
    """Convert common non-JSON types in experiment results to plain Python objects."""
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _save_results(results: dict):
    """Save the results dictionary to a JSON file with a timestamp.

    Args:
        results: The results dictionary to save.
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    json_filename = f"vqe_results_{timestamp}.json"
    output_file = output_dir / json_filename

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, default=_json_default)
    
    _plot_optimization_curve(results["quantum"]["optimization_log"], timestamp)

def process_result(optimization_result: OptimizeResult, results: dict):
    """Process the optimization result and update the results dictionary.

    Args:
        optimization_result: The optimization result from scipy.optimize.minimize.
        results: The dictionary to update with quantum optimization results.
    """
    results["quantum"]["final_cost"] = optimization_result.fun
    results["quantum"]["optimal_parameters"] = optimization_result.x.tolist()


    _save_results(results)