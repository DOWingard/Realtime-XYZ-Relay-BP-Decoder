"""Stim circuits for bivariate bicycle memory experiments.

Each syndrome cycle is the depth-8 schedule of Bravyi et al. (arXiv:2308.07915):
one ancilla per check, 6 CX per ancilla, X and Z checks interleaved so every
data qubit is touched by at most one CX per time step.

    t0      RX X-checks;            CX data -> Z-check (direction sZ[0])
    t1..t5  CX X-check -> data (sX[t]);  CX data -> Z-check (sZ[t])
    t6      CX X-check -> data (sX[6]);  M Z-checks
    t7      MX X-checks;            R Z-checks for the next cycle

Qubit layout: left data [0, h), right data [h, 2h), X checks [2h, 3h),
Z checks [3h, 4h), then noiseless reference qubits (Choi experiment only),
with h = l * m.

Experiments:
    "z"    data prepared in |0>, final noiseless Z readout, observables = logical Z.
    "x"    data prepared in |+>, final noiseless X readout, observables = logical X.
    "choi" each logical qubit is prepared noiselessly in a Bell pair with a
           reference qubit; after the noisy cycles one noiseless cycle measures
           every stabilizer, and the 2k observables are the Bell-pair parities
           X_L X_ref and Z_L Z_ref. This tracks X and Z logical failures in one
           shot and gives both detector types in the final layer.

Idle noise is charged to every data/check qubit left untouched during a
time step, except in the first step of the noisy region and in the final
step, where the data qubits are consumed by the noiseless readout.

Detector order: every noisy round lists its Z-check detectors (measured at t6), then its X-check
detectors (t7), each by check index. The Choi experiment's final layer lists X, then Z, unless
readout_in_round_order=True, which lists it like the noisy rounds. The two orders give the same
detector error model up to that relabeling of the final layer's detectors. A window decoder that
reuses one window shape at every position (the uniform boundary) reads each round's syndrome by
position within the round, so it needs every round, the final layer included, in the same order.

relay_bp_compat=True reproduces one quirk of the reference circuits shipped
with IBM's relay_bp: in the last cycle, the idle noise of the right data block
(idle at t6) and of the Z checks (idle at t7) is applied before t5 instead.
Those qubits still take part in t5 and t6, so these faults propagate
differently. With the flag set, the detector error model matches the reference
exactly; the flag changes only final-cycle probabilities, not which faults exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import stim

from rtd.bb_code import BBCode
from rtd.noise import NoiseModel

Experiment = Literal["z", "x", "choi"]
EXPERIMENTS: tuple[str, ...] = ("z", "x", "choi")

# Term (direction) each check uses at time steps t0..t6; None = the check has no CX that step.
X_SCHEDULE: tuple[int | None, ...] = (None, 1, 4, 3, 5, 0, 2)
Z_SCHEDULE: tuple[int | None, ...] = (3, 5, 0, 1, 2, 4, None)

DETECTOR_X, DETECTOR_Z = 0, 1


@dataclass(frozen=True)
class DetectorInfo:
    """Per-detector metadata, indexed like the circuit's detectors.

    check: index of the stabilizer within its type (0 .. l*m - 1)
    round: 1..rounds for noisy cycles; rounds + 1 for the final readout layer
    kind:  DETECTOR_X or DETECTOR_Z
    """

    check: np.ndarray
    round: np.ndarray
    kind: np.ndarray


@dataclass(frozen=True)
class MemoryCircuit:
    circuit: stim.Circuit
    detectors: DetectorInfo
    observable_kind: np.ndarray  # DETECTOR_X / DETECTOR_Z basis of each observable
    rounds: int
    experiment: str


class _Builder:
    """Appends instructions time step by time step and tracks measurement records."""

    def __init__(self, noise: NoiseModel, idle_pool: list[int]):
        self.circuit = stim.Circuit()
        self.noise = noise
        self.idle_pool = idle_pool
        self.num_measurements = 0
        self.touched: set[int] = set()
        self.det_check: list[int] = []
        self.det_round: list[int] = []
        self.det_kind: list[int] = []

    def reset(self, basis: str, qubits: list[int], noisy: bool) -> None:
        self.circuit.append("R" if basis == "Z" else "RX", qubits)
        if noisy and self.noise.p_reset > 0:
            self.circuit.append("X_ERROR" if basis == "Z" else "Z_ERROR", qubits, self.noise.p_reset)
        self.touched.update(qubits)

    def cx(self, pairs: list[tuple[int, int]], noisy: bool) -> None:
        flat = [q for pair in pairs for q in pair]
        self.circuit.append("CX", flat)
        if noisy and self.noise.p_cx > 0:
            self.circuit.append("DEPOLARIZE2", flat, self.noise.p_cx)
        self.touched.update(flat)

    def measure(self, basis: str, qubits: list[int], noisy: bool) -> list[int]:
        args = [self.noise.p_meas] if noisy and self.noise.p_meas > 0 else []
        self.circuit.append("M" if basis == "Z" else "MX", qubits, args)
        self.touched.update(qubits)
        return self._record(len(qubits))

    def measure_pauli_products(self, products: list[list[stim.GateTarget]]) -> list[int]:
        """Noiseless multi-qubit Pauli measurements, one record per product."""
        for product in products:
            targets: list[stim.GateTarget] = []
            for i, t in enumerate(product):
                if i:
                    targets.append(stim.target_combiner())
                targets.append(t)
                self.touched.add(t.value)
            self.circuit.append("MPP", targets)
        return self._record(len(products))

    def _record(self, count: int) -> list[int]:
        start = self.num_measurements
        self.num_measurements += count
        return list(range(start, start + count))

    def detector(self, records: list[int], check: int, round_: int, kind: int) -> None:
        targets = [stim.target_rec(r - self.num_measurements) for r in records]
        self.circuit.append("DETECTOR", targets, [check, round_, kind])
        self.det_check.append(check)
        self.det_round.append(round_)
        self.det_kind.append(kind)

    def observable(self, index: int, records: list[int]) -> None:
        targets = [stim.target_rec(r - self.num_measurements) for r in records]
        self.circuit.append("OBSERVABLE_INCLUDE", targets, index)

    def end_step(self, charge_idle: bool) -> None:
        if charge_idle and self.noise.p_idle > 0:
            idle = [q for q in self.idle_pool if q not in self.touched]
            if idle:
                self.circuit.append("DEPOLARIZE1", idle, self.noise.p_idle)
        self.circuit.append("TICK")
        self.touched = set()


def _pauli_product(basis: str, qubits) -> list[stim.GateTarget]:
    make = stim.target_x if basis == "X" else stim.target_z
    return [make(int(q)) for q in qubits]


def build_memory_circuit(
    code: BBCode,
    rounds: int,
    experiment: Experiment,
    noise: NoiseModel,
    relay_bp_compat: bool = False,
    readout_in_round_order: bool = False,
) -> MemoryCircuit:
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")
    if readout_in_round_order and experiment != "choi":
        raise ValueError(f"readout_in_round_order applies to the choi experiment only (its final layer holds both "
                         f"detector types), got {experiment!r}")

    h, n, k = code.half, code.n, code.k
    data = list(range(n))
    x_checks = list(range(n, n + h))
    z_checks = list(range(n + h, n + 2 * h))
    refs = list(range(2 * n, 2 * n + k))
    b = _Builder(noise, idle_pool=data + x_checks + z_checks)

    x_support = [np.flatnonzero(row) for row in code.hx]
    z_support = [np.flatnonzero(row) for row in code.hz]

    # Noiseless state preparation for the Choi experiment: logical |0>^k, X-stabilizer
    # frame fixed by measurement, then a logical X on logical qubit i controlled by
    # reference qubit i in |+>, giving (|0>|0_L> + |1>|1_L>)/sqrt(2) per pair.
    initial_x_frame: list[int] = []
    if experiment == "choi":
        b.reset("Z", data, noisy=False)
        b.reset("X", refs, noisy=False)
        initial_x_frame = b.measure_pauli_products([_pauli_product("X", s) for s in x_support])
        for i, ref in enumerate(refs):
            b.cx([(ref, int(q)) for q in np.flatnonzero(code.lx[i])], noisy=False)
        b.end_step(charge_idle=False)

    # First noisy step: prepare data (memory experiments) and Z-check ancillas.
    if experiment == "z":
        b.reset("Z", data, noisy=True)
    elif experiment == "x":
        b.reset("X", data, noisy=True)
    b.reset("Z", z_checks, noisy=True)
    b.end_step(charge_idle=False)

    prev_z: list[int] = []
    prev_x: list[int] = []
    for r in range(1, rounds + 1):
        first, last = r == 1, r == rounds

        early_final_idle = last and relay_bp_compat
        for t in range(7):
            if t == 0:
                b.reset("X", x_checks, noisy=True)
            if t == 5 and early_final_idle and noise.p_idle > 0:
                b.circuit.append("DEPOLARIZE1", data[h:] + z_checks, noise.p_idle)
            pairs: list[tuple[int, int]] = []
            if X_SCHEDULE[t] is not None:
                pairs += [(x_checks[i], code.x_check_neighbor(i, X_SCHEDULE[t])) for i in range(h)]
            if Z_SCHEDULE[t] is not None:
                pairs += [(code.z_check_neighbor(i, Z_SCHEDULE[t]), z_checks[i]) for i in range(h)]
            b.cx(pairs, noisy=True)
            if t == 6:
                z_meas = b.measure("Z", z_checks, noisy=True)
                for i in range(h):
                    if not first:
                        b.detector([z_meas[i], prev_z[i]], i, r, DETECTOR_Z)
                    elif experiment in ("z", "choi"):
                        b.detector([z_meas[i]], i, r, DETECTOR_Z)
                prev_z = z_meas
            b.end_step(charge_idle=not (first and t == 0) and not (early_final_idle and t == 6))

        # t7: measure X checks, then either re-prepare Z checks or read out the data.
        x_meas = b.measure("X", x_checks, noisy=True)
        for i in range(h):
            if not first:
                b.detector([x_meas[i], prev_x[i]], i, r, DETECTOR_X)
            elif experiment == "x":
                b.detector([x_meas[i]], i, r, DETECTOR_X)
            elif experiment == "choi":
                b.detector([x_meas[i], initial_x_frame[i]], i, r, DETECTOR_X)
        prev_x = x_meas
        if not last:
            b.reset("Z", z_checks, noisy=True)
            b.end_step(charge_idle=True)

    final_round = rounds + 1
    if experiment in ("z", "x"):
        basis = experiment.upper()
        data_meas = b.measure(basis, data, noisy=False)
        support, prev, kind = (z_support, prev_z, DETECTOR_Z) if basis == "Z" else (x_support, prev_x, DETECTOR_X)
        for i in range(h):
            b.detector([data_meas[q] for q in support[i]] + [prev[i]], i, final_round, kind)
        logicals = code.lz if basis == "Z" else code.lx
        for j in range(k):
            b.observable(j, [data_meas[q] for q in np.flatnonzero(logicals[j])])
        observable_kind = np.full(k, kind, dtype=np.uint8)
    else:
        final_x = b.measure_pauli_products([_pauli_product("X", s) for s in x_support])
        final_z = b.measure_pauli_products([_pauli_product("Z", s) for s in z_support])
        x_layer = [([final_x[i], prev_x[i]], i, DETECTOR_X) for i in range(h)]
        z_layer = [([final_z[i], prev_z[i]], i, DETECTOR_Z) for i in range(h)]
        for records, check, kind in (z_layer + x_layer if readout_in_round_order else x_layer + z_layer):
            b.detector(records, check, final_round, kind)
        bell_z = b.measure_pauli_products(
            [_pauli_product("Z", np.flatnonzero(code.lz[j])) + [stim.target_z(refs[j])] for j in range(k)]
        )
        bell_x = b.measure_pauli_products(
            [_pauli_product("X", np.flatnonzero(code.lx[j])) + [stim.target_x(refs[j])] for j in range(k)]
        )
        for j in range(k):
            b.observable(j, [bell_z[j]])
        for j in range(k):
            b.observable(k + j, [bell_x[j]])
        observable_kind = np.array([DETECTOR_Z] * k + [DETECTOR_X] * k, dtype=np.uint8)
    b.end_step(charge_idle=not relay_bp_compat)

    return MemoryCircuit(
        circuit=b.circuit,
        detectors=DetectorInfo(
            check=np.array(b.det_check, dtype=np.int32),
            round=np.array(b.det_round, dtype=np.int32),
            kind=np.array(b.det_kind, dtype=np.uint8),
        ),
        observable_kind=observable_kind,
        rounds=rounds,
        experiment=experiment,
    )
