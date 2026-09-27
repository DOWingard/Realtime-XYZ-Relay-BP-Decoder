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

## Golden outputs

`rtd-golden` decodes syndromes with `relay_bp` and saves everything the C++ decoder must
reproduce bit for bit. The algorithm flags have no defaults, because `relay_bp`'s Python defaults
differ from the Relay-BP paper. A missing flag is an error that names it, and so is a flag that
belongs to the other decoder.

```sh
uv run --group reference rtd-golden \
    --artifact ../data/artifacts/gross_choi_p0.003 \
    --shots ../data/shots/gross_choi_p0.003_s12345 --first 0 --count 100 \
    --decoder relay --float f32 --alpha none --alpha-scaling 1.0 --gamma0 0.125 \
    --pre-iter 80 --num-sets 600 --set-max-iter 60 --stopping nconv --stop-nconv 5 \
    --gamma-rows 64 --gamma-seed 7 --gamma-interval -0.24 0.66 \
    --posterior-shots 10 --workers 10 --out ../data/golden/gross_choi_p0.003/relay5_f32
```

- `--decoder min_sum` needs `--max-iter`. `--decoder relay` needs `--pre-iter` (iteration limit of
  leg 0), `--num-sets` (legs after leg 0), `--set-max-iter`, `--stopping {pre_iter,nconv,all}`,
  `--stop-nconv`, and the memory-strength table flags `--gamma-rows T --gamma-seed --gamma-interval LO HI`.
- `--alpha` is the check-message scale. It takes a number, `none` (scale 1), or `0` for
  `relay_bp`'s iteration-dependent rule 1 - 2^-(t/`alpha-scaling`). In that rule t = 1, 2, ...
  counts the iterations and restarts at 1 in every relay leg. `--gamma0 none` disables memory. Relay legs then
  ignore their memory strengths and each leg is plain BP.
- Syndromes come from `--shots DIR --first F --count C`, or from `--single-columns K --seed S`.
  The second option decodes the zero syndrome followed by the columns H[:, j] of K distinct columns
  drawn with `numpy.random.default_rng(S).choice(n, K, replace=False)`.
- `--posterior-shots K` saves the posterior log-likelihood ratios of the first K syndromes.

Each shot is decoded by a newly constructed `relay_bp` decoder. The per-leg records come from the
log file `relay_logging.out` that `RelayDecoder` writes into the current directory. The decoder
does not reset its per-leg records between decodes, so each shot needs a fresh decoder. Every
process runs in its own temporary directory, which lets `--workers` decode in parallel without
changing any output. The decoding, success flag, iteration count and posterior match those of one
reused decoder. Before anything is written, the tool checks that H e = syndrome (mod 2) holds
exactly on the converged shots. It also checks that each shot's leg records are consistent with
its iteration count and with the stopping rule.

The weight W(e) is the sum of ln((1 - p_j) / p_j) over the columns j with e_j = 1. It uses
`math.log` per column and adds the terms one at a time in ascending j. That is the same filter and
order as `relay_bp`'s decoding quality, so the float64 result is reproducible. A pairwise or
compensated sum can differ in the last bit.

| File | Contents |
|---|---|
| `detectors.npy` | the syndromes decoded, `uint8` [S, m] |
| `decoding.npy` | the estimate e, `uint8` [S, n] |
| `success.npy` | 1 if H e = syndrome, `uint8` [S] |
| `iterations.npy` | BP iterations, summed over all relay legs, `int64` [S] |
| `weight.npy` | W(e), `float64` [S]; +inf where the decoder did not converge |
| `posterior.npy` | posterior log-likelihood ratios of the first K shots, `float64` [K, n] |
| `columns.npy` | single-column source: the column j behind syndrome row i + 1, `int64` [K] |
| `gammas.npy` | relay: memory strengths `default_rng(seed).uniform(LO, HI, (T, n))`; leg r ≥ 1 uses row r mod T, so row 0 first serves leg T |
| `legs_ptr.npy` | relay: the legs of shot s are entries `legs_ptr[s]` to `legs_ptr[s+1] - 1` of the three leg arrays, `int64` [S + 1] |
| `leg_iterations.npy` | relay: iterations per leg, leg 0 first, `int64` [L] |
| `leg_converged.npy` | relay: 1 if the leg's hard decision satisfied the syndrome, `uint8` [L] |
| `leg_unique_best.npy` | relay: 1 on the returned leg when no other converged leg reached the same weight, `uint8` [L] |
| `manifest.json` | every constructor argument, input checksums (artifact files, shots `detectors.npy`), versions, timing, summary and output checksums |

The manifest records input paths relative to the working directory when they lie inside it, and
otherwise only their final component, so a committed manifest holds no machine-specific paths.

Stopping rules as `relay_bp` implements them:
- `nconv` ends after `--stop-nconv` converged legs, counting leg 0.
- `all` always runs all legs.
- `pre_iter` stops after leg 0 only if leg 0 converged. Otherwise it runs every leg.

The returned result comes from the converged leg with the lowest weight, taking the earliest leg
on ties. If no leg converges, it comes from leg 0.

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
