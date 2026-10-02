import pennylane as qml
from pennylane.ops.op_math import LinearCombination, Prod
from qiskit import QiskitError
from qiskit.quantum_info import SparsePauliOp
import pprint as pp
import logging

logger = logging.getLogger(__name__)

def map_pl_hamiltonian_to_qiskit(hamiltonian: LinearCombination) -> SparsePauliOp:
    """Convert a Pennylane Hamiltonian representation to a Qiskit representation.
    
    The function takes in a Pennylane LinearCOmbination object and construct the corresponding Qiskit SparsePauliOp object.

    the SparsePauliOp class has a method from_sparse_list that can be used to create a SparsePauliOp object from a list of (pauli_string, [indices], coeff) tuples. The pauli_string is a string representation of the Pauli operators (e.g., "XZ"), the indices are the qubits on which the operators act, and coeff is the coefficient of the term in the Hamiltonian.
    
    Args:
        hamiltonian (LinearCombination): A Pennylane Hamiltonian represented as a LinearCombination of operators.
        
    Returns:
        SparsePauliOp: A Qiskit SparsePauliOp object representing the same Hamiltonian.
    """
    
    coeffs, pauli_ops = hamiltonian.terms()
    terms = []
    for pauli_op in pauli_ops:
        if isinstance(pauli_op, Prod):
            observables = ""
            qubits = []
            # Extract the individual operators and their corresponding qubits
            for op in pauli_op.decomposition():
                if isinstance(op, qml.X):
                    observables += "X"
                    qubits.append(op.wires[0])
                elif isinstance(op, qml.Y):
                    observables += "Y"
                    qubits.append(op.wires[0])
                elif isinstance(op, qml.Z):
                    observables += "Z"
                    qubits.append(op.wires[0])
                elif isinstance(op, qml.Identity):
                    observables += "I"
                    qubits.append(op.wires[0])
                else:
                    raise ValueError(f"Unsupported operator type: {type(op)}")
            pauli_op = (observables, qubits)
        else:
            if isinstance(pauli_op, qml.X):
                pauli_op = ("X", [pauli_op.wires[0]])
            elif isinstance(pauli_op, qml.Y):
                pauli_op = ("Y", [pauli_op.wires[0]])
            elif isinstance(pauli_op, qml.Z):
                pauli_op = ("Z", [pauli_op.wires[0]])
            elif isinstance(pauli_op, qml.Identity):
                pauli_op = ("I", [pauli_op.wires[0]])
            else:
                raise ValueError(f"Unsupported operator type: {type(pauli_op)}")
        
        terms.append(pauli_op)
    try:
        logger.debug(f"Terms passed to SparsePauliOp...")
        logger.debug(f"Terms: {pp.pformat(terms)}")
        logger.debug(f"Coefficients: {pp.pformat(coeffs)}")
        logger.debug(f"Number of qubits: {hamiltonian.num_wires+1}")
        sparse_pauli_op = SparsePauliOp.from_sparse_list(
            [
                (observables, qubits, coeff)
                for (observables, qubits), coeff in zip(terms, coeffs)
            ],
            num_qubits=hamiltonian.num_wires+1,
        )
    except QiskitError as e:
        raise ValueError(f"Error creating SparsePauliOp: {e}")
    
    return sparse_pauli_op



if __name__ == "__main__":
    hamiltonian = qml.Hamiltonian(
        [0.5, 0.5, 0.5],
        [
            qml.X(0) @ qml.X(1),
            qml.Y(0) @ qml.Y(1),
            qml.Z(0) @ qml.Z(1),
        ]
    )

    pp.pp(hamiltonian.ops)

    qiskit_hamiltonian = map_pl_hamiltonian_to_qiskit(hamiltonian)

    pp.pp(qiskit_hamiltonian)


