# rtd (Python)

Generates noisy syndrome-extraction circuits for bivariate bicycle (BB) codes and samples
detector/observable shots with [Stim](https://github.com/quantumlib/Stim). These shots are the
input to the C++ decoder.

## Setup

```sh
cd python
uv sync
```

## Sampling shots

```sh
uv run rtd-sample --code gross --experiment choi --p 0.003 \
    --shots 100000 --seed 12345 --out ../data/shots/gross_choi_p0.003_s12345
```

- `--code`: `bb18` [[18,4,3]], `bb72` [[72,12,6]], `gross` [[144,12,12]] or `two_gross` [[288,12,18]].
- `--experiment`:
  - `z` / `x`: memory experiment in one logical basis, with k observables.
  - `choi`: each logical qubit is Bell-paired with a noiseless reference qubit, giving 2k observables and both detector types in every layer.
- `--rounds`: number of noisy syndrome cycles. Defaults to the code distance.
- `--relay-bp-compat`: reproduces the reference circuits distributed with IBM's `relay_bp` (see below).

The output directory holds the circuit, `detectors.npy` and `observables.npy` (uint8, one row per shot),
per-detector metadata (`det_check`, `det_round`, `det_type`) and a `manifest.json`. The manifest
records every parameter, the Stim and numpy versions, and SHA-256 checksums. Stim's seeded output is
reproducible only with the same Stim version on the same CPU architecture.

Logs are JSON lines on stderr.

## Exporting the decoding problem

The decoder's inputs are the detector-by-fault matrix H, the observable matrix A, and the fault
priors p. `rtd-export` produces them from a circuit using IBM's `relay_bp`, so the C++ decoder
gets exactly the matrices `relay_bp` decodes. `relay_bp` is an optional dependency group, pinned to
a specific git commit. Installing it compiles a Rust extension, so it needs `cargo`:

```sh
uv sync --group reference        # a plain `uv sync` removes the group again
uv run rtd-export --circuit ../data/shots/gross_choi_p0.003_s12345/circuit.stim \
    --out ../data/artifacts/gross_choi_p0.003
```

The artifact directory contains:

| File | Contents |
|---|---|
| `H_indptr.npy`, `H_indices.npy` | CSR of H: rows are detectors, columns are faults, `uint32` |
| `A_indptr.npy`, `A_indices.npy` | CSR of A |
| `priors.npy` | fault probabilities, `float64` |
| `col_to_dem.npy` | index of each column among the detector error model's faults |
| `det_*.npy` | detector metadata |
| `syndrome_bias.npy`, `observables_bias.npy` | only present when a fault has p = 1 |
| `manifest.json` | sizes, degree statistics, versions (including the `relay_bp` commit) and checksums; it also records the source circuit's checksum, linking the artifact to the shots sampled from that circuit |

The export uses the undecomposed detector error model. It calls `CheckMatrices.from_dem` with its
default `decomposed_hyperedges=None`. Passing `False` makes `from_dem` return the decomposed
(graphlike) matrices instead, so the export always checks the result against a direct parse of the
error model: one column per fault, in model order.

## Circuit model

- **Syndrome cycle.** The depth-8 cycle of Bravyi et al., *High-threshold and low-overhead
  fault-tolerant quantum memory* (arXiv:2308.07915). One ancilla per check; X and Z checks are
  interleaved so each data qubit takes part in at most one CX per time step.
- **Noise.** `NoiseModel.uniform(p)` uses the same p for every fault type:
  - after each CX, 15-Pauli depolarizing noise;
  - a reset prepares the orthogonal state with probability p;
  - a measurement result is flipped with probability p;
  - a qubit idle during a time step gets 3-Pauli depolarizing noise.
- **Final readout.** Noiseless.

`--relay-bp-compat` moves the last cycle's idle noise on the right data block and the Z checks from
t6/t7 to just before t5, matching the reference circuits. With the flag set, the detector error model
is identical to the reference's: the same error mechanisms with bit-identical probabilities. Without
it, only final-cycle probabilities differ.

## Tests

```sh
uv run pytest
# Also run the export tests (needs the reference group) and the parity checks against
# relay_bp's reference circuits:
RTD_RELAY_TESTDATA=/path/to/relay/tests/testdata/bicycle_bivariate uv run --group reference pytest
```
