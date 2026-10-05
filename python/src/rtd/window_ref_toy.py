"""A 3-bit repetition code under phenomenological bit-flip noise, small enough to window by hand.

Data bits q0, q1, q2 start in 0. Checks c0 = q0 + q1 and c1 = q1 + q2 (mod 2) are measured in
rounds 1 .. 6; before each measurement round every data bit flips with probability p_data, and
each check result flips with probability p_meas. Round 7 reads the data bits out without error
and recomputes both checks from them. Detector (r, c) is the XOR of check c's values in rounds r
and r - 1 (round 0 = the known initial value 0); the observable is the readout of q0.

Rows are ordered (1, c0), (1, c1), (2, c0), ..., (7, c1), so row (r, c) = 2 (r - 1) + c. Stim
lists the model's faults sorted by their detectors, which here gives 5 columns per round r:
    5 (r - 1) + 0   q1 flips before round r      rows (r, c0), (r, c1)
    5 (r - 1) + 1   c0 misread in round r        rows (r, c0), (r + 1, c0)
    5 (r - 1) + 2   q0 flips before round r      row (r, c0), and the observable
    5 (r - 1) + 3   q2 flips before round r      row (r, c1)
    5 (r - 1) + 4   c1 misread in round r        rows (r, c1), (r + 1, c1)
"""

from __future__ import annotations

import logging
from pathlib import Path

import stim

logger = logging.getLogger("rtd.window_ref")

TOY_ROUNDS = 6
TOY_P_DATA = 0.01
TOY_P_MEAS = 0.02


def toy_circuit(rounds: int = TOY_ROUNDS, p_data: float = TOY_P_DATA, p_meas: float = TOY_P_MEAS) -> stim.Circuit:
    """Detector coordinates are (check, round, type) with type 1 (Z-type), as rtd-export expects."""
    c = stim.Circuit()
    c.append("R", [0, 1, 2])
    c.append("TICK")
    z = stim.target_z
    for r in range(1, rounds + 1):
        c.append("X_ERROR", [0, 1, 2], p_data)
        c.append("MPP", [z(0), stim.target_combiner(), z(1), z(1), stim.target_combiner(), z(2)], p_meas)
        for check in range(2):
            targets = [stim.target_rec(-2 + check)]
            if r > 1:
                targets.append(stim.target_rec(-4 + check))
            c.append("DETECTOR", targets, [check, r, 1])
        c.append("TICK")
    c.append("M", [0, 1, 2])
    # rec[-3..-1] = q0, q1, q2 readout; rec[-5], rec[-4] = the last c0, c1 results.
    c.append("DETECTOR", [stim.target_rec(-3), stim.target_rec(-2), stim.target_rec(-5)], [0, rounds + 1, 1])
    c.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-1), stim.target_rec(-4)], [1, rounds + 1, 1])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-3)], 0)
    return c


def write_toy_artifact(out: Path, overwrite: bool = False) -> None:
    """circuit.stim plus the artifact rtd-export makes from it, in one directory."""
    from rtd.export import export_artifact

    if out.exists() and any(p.name not in ("expected.json", "DERIVATION.md") for p in out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} already holds artifact files; pass --overwrite to replace them")
    out.mkdir(parents=True, exist_ok=True)
    circuit_path = out / "circuit.stim"
    toy_circuit().to_file(circuit_path)
    export_artifact(circuit_path, out)
    logger.info("toy artifact written", extra={"out": str(out)})
