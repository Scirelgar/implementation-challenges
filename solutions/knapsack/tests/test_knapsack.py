"""Tests for the knapsack helper module."""

from numbers import Real
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimatorV2

import src.knapsack as knapsack_module
from src.knapsack import (
    KnapsackInstance,
    _x0_parameters,
    bruteforce,
    cost_function,
    cost_function_estimator,
    map_hamiltonian,
    objective_function,
    penalty_function,
    pstate,
)


KNAPSACK_CASES = [
    {
        "name": "two_item_balanced",
        "instance": KnapsackInstance(values=[4, 3], weights=[2, 1], capacity=2),
        "evaluations": [
            {"selection": [0, 0], "cost": 0.0, "penalty": 0.0, "objective": 0.0},
            {"selection": [1, 0], "cost": -4.0, "penalty": 0.0, "objective": -4.0},
            {"selection": [0, 1], "cost": -3.0, "penalty": 0.0, "objective": -3.0},
            {"selection": [1, 1], "cost": -7.0, "penalty": 7.0, "objective": 0.0},
        ],
        "bruteforce": ([1, 0], -4.0),
    },
    {
        "name": "three_item_tight_capacity",
        "instance": KnapsackInstance(values=[6, 2, 5], weights=[3, 2, 4], capacity=5),
        "evaluations": [
            {"selection": [1, 0, 0], "cost": -6.0, "penalty": 0.0, "objective": -6.0},
            {"selection": [1, 1, 0], "cost": -8.0, "penalty": 0.0, "objective": -8.0},
            {"selection": [0, 1, 1], "cost": -7.0, "penalty": 13.0, "objective": 6.0},
            {"selection": [1, 0, 1], "cost": -11.0, "penalty": 52.0, "objective": 41.0},
        ],
        "bruteforce": ([1, 1, 0], -8.0),
    },
    {
        "name": "single_capacity_item_choice",
        "instance": KnapsackInstance(values=[1, 10, 2], weights=[2, 1, 1], capacity=1),
        "evaluations": [
            {"selection": [0, 0, 0], "cost": 0.0, "penalty": 0.0, "objective": 0.0},
            {"selection": [0, 1, 0], "cost": -10.0, "penalty": 0.0, "objective": -10.0},
            {"selection": [0, 0, 1], "cost": -2.0, "penalty": 0.0, "objective": -2.0},
            {"selection": [1, 0, 0], "cost": -1.0, "penalty": 13.0, "objective": 12.0},
        ],
        "bruteforce": ([0, 1, 0], -10.0),
    },
]


def _case_ids(cases):
    return [case["name"] for case in cases]


def _selection_cases(field_name):
    _ = field_name
    return [
        (case, evaluation)
        for case in KNAPSACK_CASES
        for evaluation in case["evaluations"]
    ]


class TestCostFunction:
    """Tests for the negative-value cost helper."""

    @pytest.mark.parametrize(
        ("case", "evaluation"),
        _selection_cases("cost"),
        ids=[
            f"{case['name']}-{evaluation['selection']}"
            for case, evaluation in _selection_cases("cost")
        ],
    )
    def test_returns_negative_selected_value(self, case, evaluation):
        """The cost should be the negated sum of selected item values."""
        assert cost_function(
            evaluation["selection"], case["instance"]
        ) == pytest.approx(evaluation["cost"])


class TestPenaltyFunction:
    """Tests for the quadratic capacity penalty helper."""

    @pytest.mark.parametrize(
        ("case", "evaluation"),
        _selection_cases("penalty"),
        ids=[
            f"{case['name']}-{evaluation['selection']}"
            for case, evaluation in _selection_cases("penalty")
        ],
    )
    def test_penalizes_capacity_overflow_quadratically(self, case, evaluation):
        """The penalty should be zero when feasible and quadratic when infeasible."""
        assert penalty_function(
            evaluation["selection"], case["instance"]
        ) == pytest.approx(evaluation["penalty"])


class TestObjectiveFunction:
    """Tests for the combined objective helper."""

    @pytest.mark.parametrize(
        ("case", "evaluation"),
        _selection_cases("objective"),
        ids=[
            f"{case['name']}-{evaluation['selection']}"
            for case, evaluation in _selection_cases("objective")
        ],
    )
    def test_combines_cost_and_penalty(self, case, evaluation):
        """The objective should equal cost plus penalty for each selection."""
        selection = evaluation["selection"]
        instance = case["instance"]

        expected = cost_function(selection, instance) + penalty_function(
            selection, instance
        )

        assert objective_function(selection, instance) == pytest.approx(expected)
        assert expected == pytest.approx(evaluation["objective"])


class TestBruteforce:
    """Tests for exhaustive knapsack search."""

    @pytest.mark.parametrize("case", KNAPSACK_CASES, ids=_case_ids(KNAPSACK_CASES))
    def test_finds_lowest_objective_selection(self, case):
        """The brute-force helper should return the best binary selection and its score."""
        best_selection, best_value = bruteforce(case["instance"])

        expected_selection, expected_value = case["bruteforce"]
        assert best_selection == expected_selection
        assert best_value == pytest.approx(expected_value)


class TestQiskitHelpers:
    """Tests for the Qiskit-facing helper functions."""

    @pytest.mark.parametrize("case", KNAPSACK_CASES, ids=_case_ids(KNAPSACK_CASES))
    def test_map_hamiltonian_returns_ising_operator_and_offset(self, case):
        """The mapper should produce an Ising operator with the expected size."""
        hamiltonian, offset = map_hamiltonian(case["instance"], penalty_factor=10.0)

        assert isinstance(hamiltonian, SparsePauliOp)
        assert hamiltonian.num_qubits >= len(case["instance"].values)  # type: ignore
        assert isinstance(offset, Real)

    @pytest.mark.parametrize("case", KNAPSACK_CASES, ids=_case_ids(KNAPSACK_CASES))
    def test_reference_state_matches_number_of_items(self, case):
        """The reference state should allocate one qubit per item and remain empty."""
        circuit = pstate(case["instance"])

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == len(case["instance"].values)
        assert circuit.size() == 0

    def test_cost_function_estimator_records_energy_and_pub(self, monkeypatch):
        """The estimator helper should forward the pub structure and track energies."""

        class _FakeJob:
            def __init__(self, energy):
                self._energy = energy

            def result(self):
                return [SimpleNamespace(data=SimpleNamespace(evs=self._energy))]

        class _FakeEstimator(BaseEstimatorV2):
            def __init__(self, energy):
                self.energy = energy
                self.pubs = None

            def run(self, pubs):
                self.pubs = pubs
                return _FakeJob(self.energy)

        monkeypatch.setattr(knapsack_module, "OBJECTIVE_FUNC_VALUES", [])

        ansatz = QuantumCircuit(1)
        hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
        params = [0.25]
        estimator = _FakeEstimator(1.5)

        energy = cost_function_estimator(params, ansatz, hamiltonian, estimator)

        assert energy == pytest.approx(1.5)
        assert knapsack_module.OBJECTIVE_FUNC_VALUES == [1.5]
        assert estimator.pubs == [(ansatz, [hamiltonian], [params])]

    def test_initial_parameter_seed_is_deterministic(self, monkeypatch):
        """The initial-parameter helper should be repeatable for a fixed seed."""
        monkeypatch.setattr(knapsack_module, "SEED", 123)

        params_one = _x0_parameters(5)
        params_two = _x0_parameters(5)

        assert params_one.shape == (5,)
        assert params_two.shape == (5,)
        assert (params_one >= 0).all()
        assert (params_one < 1).all()
        assert params_one == pytest.approx(params_two)
