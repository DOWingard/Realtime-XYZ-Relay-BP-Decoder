"""Circuit-level noise model: an independent fault probability per operation type."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NoiseModel:
    """Probabilities of each fault, applied independently per operation.

    p_cx:    after each CX, one of the 15 non-identity two-qubit Paulis, each with p_cx / 15.
    p_reset: a reset prepares the orthogonal state (X flip after R, Z flip after RX).
    p_meas:  a measurement reports the flipped outcome.
    p_idle:  a qubit untouched during a time step suffers X, Y or Z, each with p_idle / 3.
    """

    p_cx: float
    p_reset: float
    p_meas: float
    p_idle: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} is not a probability")

    @classmethod
    def uniform(cls, p: float) -> NoiseModel:
        """The single-parameter model of Bravyi et al. (arXiv:2308.07915) used for the gross-code results."""
        return cls(p_cx=p, p_reset=p, p_meas=p, p_idle=p)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
