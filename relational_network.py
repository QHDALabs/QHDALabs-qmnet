"""
===============================================================================
QHDALabs-qmnet
Quantum Relational Network with Page-Wootters Formalism
Integrated with the conceptual gravitational string wavefunction model.

Author      : Krzysztof W. Banasiewicz
Organization: QHDALabs
Contact     : https://krzyshtof.com

Description:
This module combines conceptual quantum-network architectures,
relational time dynamics, graph-state generation, scrambling circuits,
and Page-Wootters history-state simulations.

License:
Copyright (c) QHDALabs. All rights reserved.
===============================================================================
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict
from collections import Counter
from scipy.linalg import eig, expm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import CZGate, CCZGate, CCXGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter

# Backend support
try:
    from qiskit_aer import AerSimulator
    _HAS_AER = True
except ImportError:
    from qiskit.providers.basicaer import QasmSimulatorPy as AerSimulator
    _HAS_AER = False


# =============================================================================
# SECTION 1: CONCEPTUAL MODEL
# =============================================================================

def run_theory_conceptual_demo(E_val=1.0, t_val=np.pi / 2):
    """
    Simple conceptual illustration:
    Entanglement + Evolution + Integration (Measurement)
    """
    print("\n--- RUNNING CONCEPTUAL MODEL (Gravitational Strings) ---")

    E = Parameter('E')
    t = Parameter('t')
    qc = QuantumCircuit(2)

    # Step 1: Entanglement (Strings)
    qc.h(0)
    qc.cx(0, 1)

    # Step 2: Evolution (Unitary phase transformation)
    qc.rz(E * t, 0)
    qc.rz(E * t, 1)

    # Step 3: Measurement (Integration)
    qc.measure_all()

    # Simulation
    backend = AerSimulator()
    bound_qc = qc.assign_parameters({E: E_val, t: t_val})

    # Use transpilation for backend consistency
    tqc = transpile(bound_qc, backend)

    result = backend.run(tqc, shots=1024).result()
    counts = result.get_counts()

    print(f"Results for E={E_val}, t={t_val:.2f}: {counts}")
    return counts


# =============================================================================
# SECTION 2: QHDALabs-qmnet MODULES
# Relational Time / Page-Wootters / Quantum Scramblers
# =============================================================================

def graph_state_circuit(
    n_qubits: int,
    edges: List[Tuple[int, int]]
) -> QuantumCircuit:
    """
    Build a graph-state circuit.
    """
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.h(i)

    for (u, v) in edges:
        qc.cz(u, v)

    return qc


def stabilizers_for_graph(
    n_qubits: int,
    edges: List[Tuple[int, int]]
) -> List[SparsePauliOp]:
    """
    Generate stabilizer operators for a graph state.
    """
    neighbors = {i: set() for i in range(n_qubits)}

    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    ops = []

    for v in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[v] = "X"

        for u in neighbors[v]:
            pauli[u] = "Z"

        label = "".join(reversed(pauli))
        ops.append(SparsePauliOp.from_list([(label, 1.0)]))

    return ops


def build_scrambler_U(n_qubits: int, T: int) -> QuantumCircuit:
    """
    Construct a layered scrambling circuit.
    """
    qc = QuantumCircuit(n_qubits)

    for step in range(T):

        for i in range(n_qubits):
            qc.h(i)

        start = step % 2

        for i in range(start, n_qubits - 1, 2):
            qc.cz(i, i + 1)

    return qc


def build_history_state(
    N_clock: int,
    H_s: SparsePauliOp,
    psi0: Statevector
) -> Statevector:
    """
    Construct the Page-Wootters history state.
    """
    T = 2 ** N_clock
    dim_s = 2 ** H_s.num_qubits
    Hs_mat = H_s.to_matrix()

    history = np.zeros(T * dim_s, dtype=complex)

    for t in range(T):
        U_t = expm(-1j * Hs_mat * t)
        psi_t = U_t @ psi0.data

        clock_t = np.zeros(T, dtype=complex)
        clock_t[t] = 1.0

        history += np.kron(clock_t, psi_t)

    return Statevector(history / np.sqrt(T))


def page_wootters_demo(
    N_clock: int = 3,
    N_sys: int = 1
):
    """
    Demonstration of relational time using the Page-Wootters formalism.
    """
    print(f"\n--- PAGE-WOOTTERS DEMO (Relational Time) N={N_clock} ---")

    # System Hamiltonian: Z operator
    H_s = SparsePauliOp.from_list([('Z' * N_sys, 1.0)])

    psi0_s = Statevector.from_label("+" * N_sys)
    psi_phys = build_history_state(N_clock, H_s, psi0_s)

    # Recover effective evolution via clock post-selection
    print("t | <X> expectation")

    for t in range(2 ** N_clock):

        # Extract system state for clock value t
        data = psi_phys.data.reshape(2 ** N_clock, 2 ** N_sys)

        vec_t = data[t, :]
        vec_t /= np.linalg.norm(vec_t)

        exp_x = Statevector(vec_t).expectation_value(
            SparsePauliOp.from_list([('X' * N_sys, 1.0)])
        )

        print(f"{t} | {exp_x.real:+.4f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # 1. Run conceptual presentation model
    counts_conceptual = run_theory_conceptual_demo()

    # 2. Run relational-time analysis
    page_wootters_demo(N_clock=3, N_sys=1)

    # 3. Quantum bridge test (CCZ gate)
    print("\n--- QUANTUM BRIDGE TEST (CCZ) ---")

    if _HAS_AER:
        test_qc = QuantumCircuit(3)

        test_qc.h([0, 1, 2])
        test_qc.ccz(0, 1, 2)
        test_qc.measure_all()

        res = AerSimulator().run(
            transpile(test_qc, AerSimulator()),
            shots=1024
        ).result()

        print("CCZ Bridge (Counts):", res.get_counts())

    print("\nExecution completed. All modules integrated.")

    # Optional visualization
    # plot_histogram(counts_conceptual)
    plt.show()
