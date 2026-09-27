"""Compare two circuits' detector error models independently of detector numbering.

Two circuits can model the same physics while numbering detectors differently
and choosing different logical-operator bases. Each detector is therefore keyed
by the physical measurements it combines, (qubit, n-th measurement of that
qubit), restricted to data and check qubits; noiseless helper measurements
(MPP, reference/helper qubits) are dropped from the key. Error mechanisms are
then matched by their detector-key sets. Observables are compared up to a change
of logical representative: A_mine = M A_ref + N H with M invertible, i.e. both
observable matrices span the same space modulo the rows of H. Logical operators
that differ by a stabilizer differ by exactly such an N H term, and a decoder
output satisfying H e_hat = sigma then fails on one iff it fails on the other.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import stim

DetectorKey = frozenset


def _insert(basis: dict[int, int], row: int) -> bool:
    """Reduce `row` (a GF(2) vector packed into an int) against `basis`; add it if independent."""
    while row:
        top = row.bit_length() - 1
        if top not in basis:
            basis[top] = row
            return True
        row ^= basis[top]
    return False


def _rank_with(basis: dict[int, int], rows: list[int]) -> int:
    extended = dict(basis)
    return len(basis) + sum(_insert(extended, r) for r in rows)


def detector_keys(circuit: stim.Circuit, num_code_qubits: int) -> list[DetectorKey]:
    measurement_key: list[tuple[int, int] | None] = []
    occurrences: dict[int, int] = defaultdict(int)
    keys: list[DetectorKey] = []
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            idx = [len(measurement_key) + t.value for t in inst.targets_copy()]
            keys.append(frozenset(measurement_key[i] for i in idx if measurement_key[i] is not None))
        elif stim.gate_data(inst.name).produces_measurements:
            for group in inst.target_groups():
                q = group[0].qubit_value if len(group) == 1 else None
                if inst.name == "MPP" or q is None or q >= num_code_qubits:
                    measurement_key.append(None)
                else:
                    occurrences[q] += 1
                    measurement_key.append((q, occurrences[q]))
    if len(set(keys)) != len(keys):
        raise ValueError("detector keys are not unique; comparison would be ambiguous")
    return keys


def dem_by_key(
    circuit: stim.Circuit, num_code_qubits: int
) -> dict[frozenset, list[tuple[float, tuple[int, ...]]]]:
    keys = detector_keys(circuit, num_code_qubits)
    dem = circuit.detector_error_model(decompose_errors=False)
    out: dict[frozenset, list[tuple[float, tuple[int, ...]]]] = defaultdict(list)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = frozenset(keys[t.val] for t in inst.targets_copy() if t.is_relative_detector_id())
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        out[dets].append((inst.args_copy()[0], obs))
    return out


@dataclass
class ParityReport:
    errors_mine: int
    errors_reference: int
    only_mine: int = 0
    only_reference: int = 0
    probability_mismatches: int = 0
    max_relative_difference: float = 0.0
    observables_equivalent: bool = False
    examples: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        return (
            self.only_mine == 0
            and self.only_reference == 0
            and self.probability_mismatches == 0
            and self.observables_equivalent
        )


def compare(
    mine: stim.Circuit, reference: stim.Circuit, num_code_qubits: int, rel_tol: float = 1e-9
) -> ParityReport:
    a = dem_by_key(mine, num_code_qubits)
    b = dem_by_key(reference, num_code_qubits)
    report = ParityReport(
        errors_mine=sum(len(v) for v in a.values()),
        errors_reference=sum(len(v) for v in b.values()),
    )
    num_obs_a = mine.num_observables
    num_obs_b = reference.num_observables
    # Rows of H, A_mine and A_ref over the matched error columns, packed as ints.
    det_rows: dict[DetectorKey, int] = defaultdict(int)
    obs_rows_a = [0] * num_obs_a
    obs_rows_b = [0] * num_obs_b
    column = 0

    for key in a.keys() | b.keys():
        ea, eb = sorted(a.get(key, [])), sorted(b.get(key, []))
        if len(ea) != len(eb):
            report.only_mine += max(0, len(ea) - len(eb))
            report.only_reference += max(0, len(eb) - len(ea))
            if len(report.examples) < 5:
                report.examples.append(f"count {len(ea)} vs {len(eb)} for detectors {sorted(key)}")
            continue
        for (pa, oa), (pb, ob) in zip(ea, eb):
            rel = abs(pa - pb) / max(pa, pb)
            report.max_relative_difference = max(report.max_relative_difference, rel)
            if rel > rel_tol:
                report.probability_mismatches += 1
                if len(report.examples) < 5:
                    report.examples.append(f"p {pa} vs {pb} for detectors {sorted(key)}")
            bit = 1 << column
            column += 1
            for det in key:
                det_rows[det] |= bit
            for o in oa:
                obs_rows_a[o] |= bit
            for o in ob:
                obs_rows_b[o] |= bit

    h_basis: dict[int, int] = {}
    for row in det_rows.values():
        _insert(h_basis, row)
    rank_h = len(h_basis)
    k = num_obs_a
    report.observables_equivalent = (
        num_obs_a == num_obs_b
        and _rank_with(h_basis, obs_rows_a) == rank_h + k
        and _rank_with(h_basis, obs_rows_b) == rank_h + k
        and _rank_with(h_basis, obs_rows_a + obs_rows_b) == rank_h + k
    )
    return report
