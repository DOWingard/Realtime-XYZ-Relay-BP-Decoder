# rtd (Python)

Generates noisy syndrome-extraction circuits for bivariate bicycle (BB) codes and samples
detector/observable shots with [Stim](https://github.com/quantumlib/Stim). These shots are the
input to the C++ decoder, which is also callable from Python (`rtd.RelayDecoder`,
`rtd.WindowedDecoder`, and custom decoders for sinter).

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
- `--readout-in-round-order` (`choi` only): lists the final layer's detectors like every noisy round
  (Z checks, then X checks) instead of X checks first. The detector error model is the same up to that
  relabelling. The uniform window boundary needs it for whole-problem (XYZ) decoding: it reuses one
  window shape at every position and reads each round's syndrome by row position within the round.

The output directory holds the circuit, `detectors.npy` and `observables.npy` (uint8, one row per shot),
per-detector metadata (`det_check`, `det_round`, `det_type`) and a `manifest.json`. The manifest
records every parameter, the Stim and numpy versions, and SHA-256 checksums. Stim's seeded output is
reproducible only with the same Stim version on the same CPU architecture.

Logs are JSON lines on stderr.

## Exporting the decoding problem

The decoder's inputs are the detector-by-fault matrix H, the observable matrix A, and the fault
priors p. `rtd-export` produces them from a circuit's undecomposed detector error model with
`rtd.dem` (see "From a detector error model" below). When IBM's `relay_bp` is installed (the
optional `reference` group, pinned to a specific git commit; installing it compiles a Rust
extension, so it needs `cargo`), the export also runs `relay_bp`'s `CheckMatrices.from_dem` on the
same model and stops unless both conversions agree exactly, so the C++ decoder gets the matrices
`relay_bp` decodes. The manifest records whether that cross-check ran.

```sh
uv sync --group reference        # optional; a plain `uv sync` removes the group again
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

The export uses the undecomposed detector error model: one column per fault, in model order. The
cross-check calls `CheckMatrices.from_dem` with its default `decomposed_hyperedges=None`, which
would return decomposed (graphlike) matrices for a decomposed model; the comparison would catch it.

## Decoding in Python: `rtd.RelayDecoder`, `rtd.WindowedDecoder`

The C++ decoders are available in Python through the extension module `rtd._native` (nanobind).
Given the same decoder spec and the same shots they return what `rtd_decode` writes, bit for bit.

### Building the extension

It needs nanobind in the environment (the optional `bindings` group) and is built by CMake, off by
default, into the package directory:

```sh
uv pip install nanobind sinter            # adds nanobind (and sinter for the adapter), removes nothing
cd .. && make python-ext                  # writes python/src/rtd/_native.*.so
```

`uv sync` installs exactly the groups it is given (plus `dev`) and removes the others, so a sync
that should keep `relay_bp` names every group: `uv sync --group reference --group bindings --group
sinter`.

`make python-ext` configures `build/release-py` from the `release` preset with
`-DRTD_ENABLE_PYTHON=ON -DRTD_PYTHON_OUTPUT_DIR=<repo>/python/src/rtd` and builds the `rtd_python`
target.

CMake uses `python/.venv/bin/python` unless `-DPython_EXECUTABLE=...` says otherwise. With the
option on, the libraries are compiled as position-independent code; nothing else changes. The rest
of the package works without the module; the decoder classes import it on first use.

### Whole shot

```python
import numpy as np, stim, rtd

dem = circuit.detector_error_model(decompose_errors=False)
decoder = rtd.RelayDecoder.from_detector_error_model(dem, "relay_bp5", workers=4)
predictions = decoder.decode_batch(detectors)            # [shots, k] uint8
predictions, records = decoder.decode_batch(detectors, records=True, decodings=True)
prediction = decoder.decode(detectors[0], stream=0)      # one shot
```

- A config is a decoder spec in `rtd_decode`'s JSON format, given as a dict, a path, or a preset
  name: `"relay_bp5"` (XYZ-Relay-BP-5, f32, γ₀ = 0.125, 80 + 600 × 60 iterations, 5 solutions,
  γ ∈ [−0.24, 0.66) from seed 1) or `"min_sum"`. Its `window` object is set by the class. Every
  spec `rtd_decode` runs on the CPU is accepted: `"policy": "f32" | "f64"` or
  `"arithmetic": "f32" | "f64" | "int4.2.8" | "int5.2.8" | "int6.2.8"` (fixed point needs a
  shift-form α: constant 1, or adaptive with scaling 1), and in a version-3 spec a
  `"selection"` policy (`null` gives the version-2 decoder).
- `detectors` is uint8 or bool `[shots, m]`, or bit-packed `[shots, ⌈m/8⌉]` with
  `bit_packed_shots=True` (little bit order, stim's `b8`). C-contiguous uint8 input is read in
  place. `bit_packed_predictions=True` packs the output the same way.
- Shot s of a batch draws its relay γ values from stream `stream_offset + s`, as shot s of an
  `rtd_decode` run does, so a batch decoded here with `stream_offset=first` reproduces
  `rtd_decode --first first` on the same shots.
- `workers` threads decode shots in parallel; the result does not depend on how many.
- `records=True` adds per-shot arrays under `rtd_decode`'s names: `success`, `iterations`, `legs`,
  `best_leg`, `weight`, `decode_ns`; `decodings=True` the correction ê `[shots, n]`;
  `record_solutions=N` (constructor) the first N converged legs (`sol_count`, `sol_leg`,
  `sol_iterations`, `sol_weight`, `sol_class`, `sol_hash`, `sol_size`, `returned_class`). The
  return type depends on `records` alone: `decodings=True` (and, for windows, `commits=True`)
  without `records=True` raises `ValueError`.
- Under a selection policy the records hold every decode's confidence, `[shots, 1]` here and
  `[shots, K]` for windows, under `rtd_decode`'s names: `conf_found`, `conf_seen`,
  `conf_distinct`, `conf_classes`, `conf_best_class`, `conf_second_class`, `conf_weight`,
  `conf_gap`, `conf_agreement`, `conf_gap_state` (0 no solution, 1 one class, 2 gap defined),
  `conf_first_legs`, `conf_first_iterations`, `conf_class_sum_class`, `conf_class_sum_top`,
  `conf_agreement_class`, `conf_q_supp`, `conf_q_sum_sq`, `conf_q_total`, `conf_components`,
  `conf_syndrome_ones`, `conf_syndrome_rows`, `conf_decided_class`, `conf_decided_leg`,
  `conf_extra_legs`, `conf_score`, `conf_low`, `conf_stopped_early`. The prediction is that of the
  solution the policy decided.
- Every returned array wraps the buffer the C++ decoder wrote; nothing is copied on the way out.
- `RelayDecoder(problem, config)` takes any `rtd.DecodingProblem`; `from_artifact(dir, config)`
  reads an `rtd-export` artifact.

### Sliding windows and streams

```python
windowed = rtd.WindowedDecoder.from_detector_error_model(
    dem, width=12, commit=8, config="relay_bp5", round_of=1)   # coordinate 1 holds the round
predictions, records = windowed.decode_batch(detectors, records=True, commits=True)
```

A window covers W = `width` rounds and commits the faults of its first C = `commit`; the other
arguments are the spec's window fields (`converge_rounds`, `boundary` `exact`/`uniform`,
`on_failure` `commit_anyway`/`defer`/`flag`, `max_deferrals`, `iteration_cap`). The records add
per-window arrays `[shots, K]` (K window positions, `windowed.num_positions`): `win_iterations`,
`win_legs`, `win_attempts`, `win_converged`, `win_cap_hit`, `win_weight`,
`win_committed_weight`, `win_unexplained`, `win_flagged`, `win_virtual`, `win_decode_ns`, and
`flagged` per shot; `commits=True` adds `commit_ptr`/`commit_faults` (the faults window
q = s·K + k committed are `commit_faults[commit_ptr[q]:commit_ptr[q+1]]`). `windowed.plan` describes
the window shapes and positions. Under a selection policy the `conf_*` arrays are `[shots, K]`,
`conf_low_deferrals` counts the attempts deferred for low confidence, the policy's `on_low`
action (`defer`, `flag`) applies to every window, and a `"history"` adds `hist_<signal>`
`[shots, K, L]` (the signal over the last L windows after window k, L from the spec's `lengths`)
and `hist_state`; the `hist_*` arrays are views of one buffer.

A batch logs, at most for 10 batches per decoder and kind: a warning when more than 1 % of its
shots did not converge, a warning when windows hit the iteration cap, and an error when shots were
flagged, each with the counts and the window policy. `decoder.totals` keeps the running counts
(batches, shots, non-converged, windows at the cap, flagged, low-confidence decodes).

A stream decodes one shot as its rounds arrive:

```python
stream = windowed.stream()
stream.reset(shot)                     # the shot index keys the γ streams, as in a batch
for r, bits in enumerate(rounds):      # bits: the M detectors of round r + 1
    (stream.push_round if r + 1 < len(rounds) else stream.push_final)(bits)
    while stream.window_ready():
        commit = stream.decode_next()  # WindowCommit: window, faults, frame_delta, deferred, record
stream.predicted()                     # [k] uint8, equal to the batch's row for this shot
```

Pushing never decodes; `decode_next` decodes the oldest window whose rounds have all arrived. Out
of order calls raise `RuntimeError` (a round after the readout, `decode_next` with no window
ready); input of the wrong size raises `ValueError`. Every refusal is logged. Under a selection
policy each commit's `record` holds `confidence` (a dict of the values above, or `None`),
`low_confidence` and `low_confidence_deferrals`, and `stream.history()` returns the history after
the last committed window (`{signal: [L], "state": [L], "lengths": [L]}`, `None` without one).

## From a detector error model: `rtd.dem`

`rtd.problem_from_dem(dem, round_of=None, prune_threshold=0.0)` converts a stim detector error
model into a `DecodingProblem`: H (detectors × mechanisms), A (observables × mechanisms), the
priors, and, with `round_of`, each detector's round.

- One column per error mechanism, in the order of the flattened model (repeat blocks unrolled,
  detector shifts applied). A mechanism written with `^` separators flips the symmetric
  difference of its components; a decomposed model is accepted but logged, since its mechanisms
  differ from the undecomposed model's (build with `decompose_errors=False`).
- Mechanisms with p = 0 are dropped, and those with p = 1 are dropped into the biases: the syndrome
  bias b_σ is the XOR of their detector flips, the observable bias b_ℓ of their observable flips;
  the decoder decodes σ ⊕ b_σ and reports ℓ̂ ⊕ b_ℓ. `prune_threshold` t also prunes p ≤ t and
  p ≥ 1 − t (default 0: exactly `relay_bp`'s `prune_decided_errors`).
- `round_of` reads a detector's round from its coordinates: an index into the coordinate tuple
  (this package's circuits keep the round at index 1) or a function of the tuple. Windows need
  rounds 1 … Rt, grouped and of equal size M; `check_time_structure` reports anything else.
- `problem_from_artifact(dir)` reads an artifact back into the same form.

The conversion reproduces the committed artifacts exactly (bb18 R = 9, and gross R = 12 when the
data directory holds it), priors bit for bit.

## Monte Carlo with sinter: `rtd.sinter_decoders()`

```python
import sinter, rtd
tasks = [sinter.Task(circuit=circuit,
                     detector_error_model=circuit.detector_error_model(decompose_errors=False),
                     decoder="rtd_relay")]
stats = sinter.collect(num_workers=6, tasks=tasks, max_errors=100,
                       custom_decoders=rtd.sinter_decoders("relay_bp5", windows=((12, 8),)))
```

`sinter_decoders(config, windows=..., round_of=1, ...)` returns `sinter.Decoder`s named
`rtd_relay` (whole shot) and `rtd_window_<W>_<C>` for each window. Give every task its
undecomposed model: without one, sinter builds a decomposed model, whose mechanisms differ. A
compiled decoder numbers the shots it decodes 0, 1, 2, … across batches and uses stream s for
shot s, so on the same shots it predicts exactly what `rtd_decode` predicts. Each sinter worker
compiles its own decoder and so reuses the same γ draws on different shots; the draws do not
depend on the syndromes, so estimated rates are unbiased.

Any spec the decoder classes accept works here (fixed point and selection policies included).
For a long collection, `sinter_decoders(..., require_native_sha256=rtd.sinter_adapter.native_module_sha256())`
pins the extension module: every worker checks the sha256 of the module it loaded and refuses to
decode with another build. A compiled decoder with `keep_records = True` keeps every batch's
records in `batches` as `(first stream, records)`, to compare the decodes sinter's path made with
another decoder's shot by shot.

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

## XZ decoding: splitting the problem by detector type

For a CSS code, X-type detectors see only the Z part of a fault and Z-type detectors only its X
part, and X-basis observables are flipped only by Z parts (and the other way round). The XYZ
problem (H, A, priors) therefore splits into two independent problems:

- half `x`: the X-type detectors (`det_type` 0) and the X-basis observables (`observable_type` 0);
- half `z`: the Z-type detectors (`det_type` 1) and the Z-basis observables (`observable_type` 1).

Each fault is restricted to the half's rows. A fault with no row in the half is dropped. Faults
whose restricted columns are identical are one event for that half and become one column that
occurs when an odd number of them occur: its prior is the fold
`p = p * (1.0 - q) + q * (1.0 - p)` over the members in ascending XYZ column index. The column's
representative is its smallest XYZ index. Columns are ordered by representative, and rows keep
their XYZ order, so each half is still grouped by round. The split does not assume the pairing
of detector and observable types; it checks it. Every fault that flips a paired observable must
have a row in the half, and all members of a merged column must flip the same paired
observables. Either violation is an error.

```sh
uv run rtd-xz-split --artifact ../data/artifacts/gross_choi_p0.003 \
    --shots ../data/shots/gross_choi_p0.003             # writes ../data/artifacts/gross_choi_p0.003_xz/{x,z}
uv run rtd-xz-split-shots --shots ../data/shots/gross_choi_p0.003 \
    --split ../data/artifacts/gross_choi_p0.003_xz --first 0 --count 500 --out /tmp/shots_xz
```

`--shots` supplies `observable_type.npy`. Without it the basis of each observable is inferred:
an observable belongs to the type whose rows every fault flipping it has. Each half is an
ordinary artifact that `rtd_decode` and `rtd-golden` load. It also contains
`col_members_ptr.npy` / `col_members.npy` (uint32 CSR: the XYZ columns merged into each column),
`det_global.npy` and `obs_global.npy` (the XYZ row and observable of each row and observable), and
an `"xz"` block in the manifest (half, parent path and manifest checksum, merged and dropped
column counts). Its `source_circuit` checksum is the parent's. The shot directories written per
half record the same circuit checksum, which is how `rtd_decode` matches them to the half. A shot
decoded in XZ mode fails when either half predicts a wrong observable.

On the gross code at p = 3·10⁻³ over 12 rounds, each half has 936 rows, 8,784 columns and 30,672
edges; the XYZ problem has 1,872 rows, 71,280 columns and 419,904 edges. In each half, 8,784 XYZ
columns are dropped and the other 62,496 merge into the 8,784 columns, every one of which merges
at least two faults.

## Campaigns: chunked, resumable decoding runs

`rtd-campaign` samples shots in chunks from recorded seeds and decodes each chunk with a pinned
copy of `rtd_decode`, once per decoder configuration ("arm"). It keeps the per-shot results and
deletes the raw shots, because the seed regenerates them bit for bit. All arms decode the same
shots, so they can be compared shot by shot.

```sh
uv run rtd-campaign init campaign.json      # -> ../data/campaigns/<name>/
uv run rtd-campaign run ../data/campaigns/<name> [--max-wall S] [--arms a,b] [--chunks N]
uv run rtd-campaign status ../data/campaigns/<name>
uv run rtd-campaign summarize ../data/campaigns/<name> [--bootstrap B] [--seed S]
```

The config requires every field; use null where a field allows it:

```json
{
  "name": "a3_xz_p0.003",
  "circuit": {"code": "gross", "experiment": "choi", "p": 0.003, "rounds": 12, "relay_bp_compat": true},
  "mode": "xz",
  "seed": 1,
  "chunk_shots": 60000,
  "arms": {"relay5": {"spec": "configs/xz_relay5_f32.json", "workers": 6, "cpus": "0-5",
                      "record_solutions": 0, "save_commits": false,
                      "save_solution_supports": false, "extra_args": []}},
  "stop": {"max_shots": 2000000, "max_wall_seconds": null, "min_failures": 100,
           "failure_arm": "relay5", "paired": null},
  "keep_raw": false
}
```

- `circuit`: optionally `"readout_in_round_order": true` (`choi` only; see `--readout-in-round-order`
  above); the artifact name then ends in `_rorder`. `init` refuses an arm whose spec asks for the
  uniform window boundary when the rounds of a decoded artifact list their detectors, (type, check),
  in different orders.
- `mode`: `xyz` decodes the whole problem; `xz` decodes both halves of the XZ split; `z_only` and
  `x_only` decode one half.
- `arms`: `spec` is an `rtd_decode` spec file, copied byte for byte into the campaign's
  `specs/<arm>/` together with any files it names by relative path (`"path"` or `"directory"`
  values), so arms whose specs use the same relative name keep their own files.
  `workers` and `cpus` are passed to `rtd_decode`.
  `record_solutions`, `save_commits` and `save_solution_supports` become the corresponding
  `rtd_decode` flags when set. `extra_args` are appended verbatim. Flags the driver sets itself
  (`--artifact`, `--shots`, `--config`, `--out`, `--first`, `--count`, `--workers`, `--cpus`)
  are refused. `init` checks that the pinned binary's `--help` lists every flag an arm needs,
  and refuses a spec whose `window.mode` is not `whole_shot` when that list lacks
  `--save-commits`: a binary without windowed decoding would ignore the window and decode
  whole shots.
- `stop`: after every chunk the run stops when any configured criterion holds. The criteria are
  `max_shots` sampled; `max_wall_seconds` of campaign compute time (sampling plus decoding,
  summed over all runs); `min_failures` failures of `failure_arm`; and `paired`, which stops once
  the 95% interval on V = P_fail(test) / P_fail(reference) is within ±`rel_halfwidth` of V. At least
  one of `max_shots` and `max_wall_seconds` must be set. The last chunk is shortened to meet
  `max_shots` exactly. A run first decodes chunks sampled earlier that a selected arm still
  lacks; `max_wall_seconds`, `min_failures` and `paired` are also checked after each of those,
  and one that becomes true stops the run. A criterion that already held when the run started
  does not stop this catch-up, so an arm added to a finished campaign still gets every chunk.
- `stop.paired` is one `{"reference", "test", "rel_halfwidth"}` object, a non-empty list of such
  objects, or null. A list holds only when every listed pair is within its own `rel_halfwidth`.
  With `min_failures` on the reference arm, a campaign with several test arms then runs until the
  reference has that many failures or every comparison is resolved, whichever comes first. Each
  listed pair must name two different arms, no pair may appear twice (also not reversed), and each
  `rel_halfwidth` must be a finite number > 0. Each pair is judged like a single pair: on the
  chunks both arms completed, with n₁₁ shots where both fail, n₁₀ where only the reference fails
  and n₀₁ where only the test arm fails, V = (n₁₁ + n₀₁) / (n₁₁ + n₁₀), the delta-method interval
  V·exp(±1.96·σ) with σ² = 1/(n₁₁ + n₀₁) + 1/(n₁₁ + n₁₀) − 2n₁₁ / ((n₁₁ + n₀₁)(n₁₁ + n₁₀)) (exact
  rules for degenerate counts), and relative half width max(hi/V − 1, 1 − lo/V). A pair stays unmet
  until both arms have failed at least once. There is no larger minimum count, so arms that mostly
  fail on the same shots can meet ±25% early (n₁₁ = 10, n₁₀ = 0, n₀₁ = 1 already gives −17%/+21%,
  after 10 reference failures). In `status` and in
  the result of `run`, the criterion is still called `paired`. For a list, its record has one entry
  per pair under `pairs`, plus `unmet` and `pairs_met`. The listed pairs are also summarized.
- Optional fields: `rtd_decode` (the binary to pin; default `build/release/src/rtd_decode`),
  `pairs` (extra `[{"reference", "test"}]` pairs to summarize), `description`,
  `gamma_seed_per_chunk` (default `true`; see below) and, per arm, `decode_timeout_seconds`
  (wall seconds one `rtd_decode` invocation, i.e. one half of one chunk, may take; `null` or
  absent means no limit). Relative paths are resolved against the repository root.

`init` writes `campaign.json` (the config, byte for byte), `circuit.stim`, `bin/rtd_decode`,
`specs/<arm>/` and `init.json` (paths and SHA-256 of the binary, the specs and every file they
reference, the circuit and the artifacts). Every run checks these checksums before it decodes.
It never rebuilds the binary. Artifacts are cached under `data/artifacts/`, named
`<code>_<experiment>_p<p>_r<rounds>[_compat]`, with the XZ halves in `<name>_xz/{x,z}/`. An
existing artifact exported from the same circuit (same circuit SHA-256) is reused under whatever
name it has. Otherwise the circuit is exported with `rtd-export` (which needs the reference
group) and then split. Running `init` again with the same config does nothing. A config that
only adds arms, or changes `stop`, `keep_raw`, `pairs` or an arm's `decode_timeout_seconds`,
extends the campaign. Any other change is refused.

Chunk c samples with stim seed `(seed · 1000003 + c) mod 2^63` in batches of 100,000 shots.
`chunks/<c>/chunk.json` records the seed, the shot count, the SHA-256 of the sampled detector
and observable bytes, and the SHA-256 of each half's arrays. A chunk that must be decoded again
(for an arm added later, or after an interruption) is regenerated and checked against those
checksums. Raw shots still on disk from an earlier attempt (a retry, or a run that stopped before
deleting them) are decoded again only if their arrays match the recorded checksums; otherwise
they are regenerated. Each arm's output goes to
`chunks/<c>/<arm>/` (or `<arm>/x/` and `<arm>/z/`), `rtd_decode`'s log to
`chunks/<c>/logs/<arm>.<half>.log`, and `done.json` marks the arm complete for the chunk. Raw
shots in `chunks/<c>/shots/` are deleted once the chunk's arms are decoded, unless `keep_raw` is
set.

Unattended operation:
- Every run resumes where the last one stopped. The unit of work is one (chunk, arm) pair.
- A failed chunk is logged with its context (the command, exit code and the tail of the
  decoder's log) and retried once. A second failure stops the run with exit code 2.
- An `rtd_decode` invocation that runs longer than its arm's `decode_timeout_seconds` is logged
  with its context, its process group is terminated (SIGTERM, SIGKILL after 10 s), and the chunk
  counts as failed: retried once, then the run stops.
- A chunk that fails after a stop was requested is not retried (exit code 2); the next run
  redoes it.
- The first SIGINT, SIGTERM or SIGHUP lets the current chunk finish and then stops (exit 0). A
  second one terminates `rtd_decode` (or stops between sampling and decoding), removes the arm's
  partial output and exits with code 130; the next run redoes that chunk. Under `nohup`, SIGHUP
  stays ignored, so closing the terminal does not stop the run.
- If the driver itself is killed (SIGKILL, the OOM killer), the kernel sends `rtd_decode`
  SIGTERM. Each decoder's pid is kept in `chunks/<c>/logs/<arm>.<half>.pid` while it runs, and a
  new run first terminates any decoder an earlier run left behind. It also deletes raw shots of
  chunks whose arms are all done (unless `keep_raw`), partially written shot directories and
  per-chunk spec copies that an interrupted run left behind.
- JSON markers (`chunk.json`, `done.json`, `init.json`, `campaign.json`, `summary.json`) are
  written to a temporary file, synced and renamed into place, and raw shots are written to a
  temporary directory that is renamed when complete. A full disk or a crash therefore never
  leaves a truncated marker. A marker that is unreadable anyway counts as absent: it is renamed
  to `<name>.corrupt-<time>` and its chunk or arm is done again.
- Logs are JSON lines on stderr and in `campaign.log`. Only one `run` may be active per campaign
  at a time. A chunk is not started without enough free disk for its raw shots, and `init`
  refuses chunks whose raw detectors would exceed 2 GB.

`summarize` writes `summary.json`. For each arm, and for each half in split modes, it reports the
shots, the failures, P_block with its Wilson 95% interval, the block rate per cycle
1 − (1 − P_block)^{1/R} with the mapped interval, and (for the combined outcome) the rate per
logical qubit per cycle. It also reports the converged fraction, the mean iterations with a
bootstrap 95% interval, nearest-rank quantiles (p50 to p99.9) with bootstrap intervals, legs,
and decode time. For XZ the combined iterations are given as the sum and the maximum over the
halves. For each configured pair it reports the paired counts n₁₁, n₁₀ and n₀₁, the ratio V with
its delta-method interval on ln V, and a paired bootstrap.

Relay γ draws. `rtd_decode` draws a shot's γ values from a stream keyed on (the spec's seed, the
shot's row in its shots file). Chunk files restart at row 0, so with one seed row i of every
chunk would reuse the same draws. With `gamma_seed_per_chunk` (the default), an arm whose spec
has a `"uniform"` `gamma_source` decodes chunk c with a copy of its pinned spec in which only
`gamma_source.seed` is replaced, by

    seed_c = splitmix64(spec_seed XOR splitmix64(c + 1))      (64-bit)

where `splitmix64(x)`: x += 0x9E3779B97F4A7C15; x = (x ^ (x >> 30)) · 0xBF58476D1CE4E5B9;
x = (x ^ (x >> 27)) · 0x94D049BB133111EB; return x ^ (x >> 31), all modulo 2^64. Both steps
are bijections, so different chunks get different seeds. seed_c depends only on the spec seed
and c: every arm with the same spec seed and both halves of a split shot see the same draws for
the same shot, which keeps paired comparisons and replays aligned. The copy is written beside
the pinned spec (`specs/<arm>/.chunk-<c>.<name>`, so relative references resolve as before) and
removed after the arm's decode; `done.json` records `gamma` (`source`, `per_chunk`, `spec_seed`,
`seed`, `rule`) and `config_sha256`, and `rtd_decode`'s `run.json` holds the spec it used. A
version-2 spec (with a `window` object) is handled the same way. Explicit γ tables are used as
pinned. `init.json` records the switch; it cannot change once a campaign exists, and a campaign
initialized before the switch existed keeps the spec seeds. With the spec seed on every chunk,
failures of shots that share a row are weakly positively correlated, so the intervals, which
treat shots as independent, are slightly too narrow. `summary.json` states which case applies
under `caveats`.

## Records of a campaign

```python
from rtd import records
rec = records.load("../data/campaigns/a3_xz_p0.003", "relay5", names=("logical_failure", "iterations"))
rec.shots, rec.chunks            # shot count, completed chunks in order
rec.chunk, rec.shot_in_chunk     # per shot: its chunk and its row in that chunk's shots
rec.half("x")["iterations"]      # per-half arrays, concatenated over chunks
rec["logical_failure"]           # combined outcome: any half failed
rec["success"]                   # every half converged
rec["iterations_sum"], rec["iterations_max"]
```

In every mode `rec[name]` holds the combined outcome (`logical_failure`, `flagged`, `success`)
and the per-shot sums and maxima over the halves (`iterations_sum`, `iterations_max`,
`legs_sum`, `legs_max`, `decode_ns_sum`, `decode_ns_max`, `weight_sum`); analysis meant for every
mode uses these names. With one half (`xyz`, `z_only`, `x_only`) the sums and maxima equal that
half's values, and `rec[name]` also holds every array of that half under its own name. With two
halves the other arrays are only in `rec.half(h)`.

`rec.gamma_seeds` maps each chunk to the γ seed the arm decoded it with (`None` for explicit
tables or chunks recorded by an older driver); `rec.gamma_seed` gives it per shot. A shot's
draws are keyed on (`gamma_seed`, `shot_in_chunk`), which is what a replay needs.

Every `.npy` array of `rtd_decode`'s output is concatenated along the shot axis in chunk order.
This covers per-window [S, K] and per-solution [S, K, N] arrays as well. CSR pairs
(`commit_ptr`/`commit_faults`, `solsup_ptr`/`solsup_idx`) are concatenated with their offsets
shifted. Only chunks with a readable `chunk.json` and `done.json` are included. `names` limits what is read, and `chunks`
selects particular chunks.

## Sliding-window reference decoder: `rtd.window_ref`

A deliberately plain implementation (scipy sparse matrices, one window at a time) of
sliding-window decoding. It is the independent reference the C++ windowed decoder is tested
against, and it can run any of four inner decoders.

**The rule.** Rounds are r = 1 … Rt (Rt = R + 1; the last is the noiseless readout), with M
detectors per round, rows sorted by round. s(j) is the earliest round fault column j touches;
every column touches only rounds s(j) and s(j) + 1 (checked at load). Window k starts at
t_k = 1 + kC and spans W rounds. It decodes the columns with s(j) ∈ [t_k, t_k + W) against the
residual syndrome of those rounds, and commits those with s(j) ∈ [t_k, t_k + C). Columns with
s(j) = t_k + W − 1 lose their rows in round t_k + W. Those that then have identical supports merge
into one column with prior p = p₁(1 − p₂) + p₂(1 − p₁) (folded in ascending column index),
represented by the smallest index. The first window reaching Rt is final: it covers [t_k, Rt]
and commits everything left. A commit flips the column's rows in the residual (the carry into the
next window) and XORs its observables into the logical frame. Local columns are ordered by
representative and local rows in global order, so W ≥ Rt gives the whole-shot problem unchanged.

- `--boundary exact` builds every window from the true matrix. `uniform` uses the bulk window
  (the one at k = 1) everywhere, zero-pads rounds past Rt, and maps committed columns to global
  ones by a shift in time. A column with no counterpart is *virtual*: it is counted, and it is
  left out of the committed-fault lists. The bulk window with every deferral width must end by
  round Rt − 2, or the plan is rejected.
- `--on-failure commit_anyway` commits the inner decoder's output even if it did not converge.
  `flag` does the same and marks the shot. `defer` retries the same start with width W + aC,
  a = 1 … `--max-deferrals`, and flags when those run out. A deferral attempt that reaches Rt is
  final. The shot is then complete, and the remaining positions record an empty window
  (attempts 0).
- `--converge C'` sets the rounds on which H e = s must hold. relay_bp, PyMatching and BP+LSD check
  every row, so they need C′ = W and no `--iteration-cap`.

Inner decoders (`--inner`): `relay` (relay_bp, a freshly constructed decoder per window decode,
memory strengths from an explicit table per window shape: `default_rng([seed, shape]).uniform(LO,
HI, (T, n_shape))`, relay leg r ≥ 1 uses row r mod T), `pymatching` (graph-like windows only),
`bplsd` (ldpc's BP+LSD: min-sum, 30 iterations, LSD order 0, LSD run on every decode) and `brute`
(exact minimum weight by enumerating the solution space; tiny windows only, reports ties). In
Python, any object with `decode(H_local, priors_local, syndrome_local, context) -> InnerResult`
(support, iterations, legs, converged, weight) can be used.

```sh
python -m rtd.window_ref dump-plan --artifact ../test/fixtures/bb18_choi_r9/artifact \
    --width 4 --commit 2 --converge 4 --boundary exact --on-failure defer --max-deferrals 2 --out /tmp/plan
python -m rtd.window_ref compare-plans /tmp/plan /tmp/plan_from_cpp
python -m rtd.window_ref decode --artifact A --shots S --count 100 --width 4 --commit 2 --converge 4 \
    --boundary exact --on-failure flag --max-deferrals 0 --inner bplsd --save-commits --out /tmp/run
python -m rtd.window_ref golden --artifact A --shots S --count 100 --width 4 --commit 2 --converge 4 \
    --boundary uniform --on-failure flag --max-deferrals 0 --float f32 --alpha none --alpha-scaling 1.0 \
    --gamma0 0.125 --pre-iter 80 --set-max-iter 60 --num-sets 50 --stopping nconv --stop-nconv 3 \
    --gamma-rows 3 --gamma-seed 17 --gamma-interval -0.24 0.66 --workers 2 --out DIR
python -m rtd.window_ref toy-artifact --out ../test/fixtures/toy_rep3
```

`dump-plan` writes `plan.json` (the spec, Rt, M, the number of window positions, every shape's
rows, columns, edges and merged columns, and every placement's window, attempt, shape, first
round, rounds, committed rounds and finality) together with, per shape, the local H in CSR
(`uint32`), priors (`float64`), commit flags, convergence rows and observable masks (`uint64`).
Per placement it writes the global column behind each local column; for the exact boundary it
also writes the merged members. `compare-plans` reports every difference, with priors compared
bit for bit. `decode` writes `rtd_decode`'s per-shot arrays (`success`, `iterations`, `legs`,
`weight`, `predicted_observables`, `logical_failure`, `flagged`) and the per-window `win_*`
arrays [shots × positions], and it checks each shot before writing. The checks are: the
unexplained counts add up to the residual; with the exact boundary no column is committed twice,
H c + residual = σ and A c = frame; and if every window converged, the residual is zero.

`golden` does the same with relay_bp and writes a directory the C++ windowed decoder must
reproduce: `manifest.json` (artifact path and manifest checksum, shots source, window object,
decoder and γ-table settings, relay_bp commit, summary, sha256 of every file), `spec.json` (the
equivalent `rtd_decode` version-2 spec with `gamma_source` `explicit_shapes`),
`gammas/shape_<i>.npy`, the decoded `detectors.npy` / `observables.npy`, and the `win_*`,
`commit_ptr`/`commit_faults`, `predicted_observables`, `logical_failure` and `flagged` arrays.
The manifest also records the source circuit's checksum and `rounds`, so the directory is
accepted as `rtd_decode --shots`. Weights are `math.log((1 − p)/p)` per local column, summed in
ascending local index.

Committed fixtures:
- `test/fixtures/toy_rep3/`: a 3-bit repetition code, 6 noisy rounds plus readout
  (phenomenological, p_data = 0.01, p_meas = 0.02). It holds the circuit, its artifact,
  `expected.json` (three cases for windows (3, 1) and (4, 2): per-window carry, residual,
  solution, commits and frame) and `DERIVATION.md`, which works every value out by hand.
- `test/fixtures/bb18_choi_r9/`: bb18 choi, R = 9, p = 0.01, no compat flag, 100 shots
  (`rtd-sample … --rounds 9 --shots 100 --seed 909`), with `shots/` and `artifact/`.
- Windowed goldens `window_<W>_<C>_<boundary>_<policy>/` in `test/fixtures/bb18_choi/` and
  `test/fixtures/bb18_choi_r9/`. All use relay_bp f32 with γ₀ = 0.125, 80 iterations in leg 0,
  60 in each of up to 50 relay legs, stop after 3 solutions, α = 1, and γ tables of T = 3 rows
  from seed 17 on [−0.24, 0.66] (T is kept small because the tables dominate the fixture's
  size). `defer` uses max_deferrals 2 with the exact boundary and 1 with the uniform one: at R = 9
  a wider uniform bulk window would reach the readout. The R = 3 goldens decode the same 100 syndromes as the whole-shot
  goldens next to them. Their observables come from re-sampling those shots with
  `rtd-sample --code bb18 --experiment choi --p 0.01 --rounds 3 --shots 200 --seed 2026`, which
  reproduces the recorded `detectors.npy` checksum exactly.

## Fixed-point emulator, test vectors and window netlists

`rtd.fixed_ref` is an integer-only (numpy) model of Relay-BP in the format intN.S.M of IBM's FPGA
decoder: log-likelihoods scaled by S and saturated to N magnitude bits plus a sign, memory
strengths stored as β = round((1 − γ)M), γ·x computed as x − β⊗x with a truncating shift-and-add
product, min-sum scaling α = 1 − 2^{−k} as x − (x >> k). It is the golden model the C++
`"arithmetic": "int4.2.8"` decoder must match bit for bit (commits, iterations per window,
convergence, frame). `FixedRelay` decodes one problem; `FixedInnerSpec` plugs it into
`rtd.window_ref` as the inner decoder of a sliding-window run.

```sh
python -m rtd.fixed_golden golden --artifact A --shots S --count 100 --format int4.2.8 \
    --alpha adaptive --gamma0 0.125 --pre-iter 80 --set-max-iter 60 --num-sets 20 \
    --stopping after_n_converged --stop-count 5 --gamma-rows 16 --gamma-seed 23 \
    --gamma-interval -0.24 0.66 --width 8 --commit 4 --converge 8 --boundary exact \
    --on-failure commit_anyway --workers 3 --out DIR
python -m rtd.fixed_golden vectors DIR
python -m rtd.fixed_netlist window --artifact A --width 12 --commit 8 --converge 8 \
    --boundary uniform --format int4.2.8 --out NETLIST_DIR
python -m rtd.fixed_netlist lift --artifact A
```

`golden` writes the arrays of a window-reference golden (`win_*`, `commit_ptr`/`commit_faults`,
`predicted_observables`, `logical_failure`, `flagged`, per-shot totals), the γ tables
`gammas/shape_<i>.npy`, the equivalent `rtd_decode` spec `spec.json` and a manifest with the
sha256 of every file. `vectors` decodes the golden again, checks that it reproduces it, and adds
the hardware-facing test vectors: `vectors.npz` holds the syndrome stream and, per window shape,
H~ (CSR), the quantised priors λ_int (int8), commit and convergence masks, the observable mask
of every column and the quantised memory strengths β_int of every relay leg; per placement the
global columns; per shot and window the iterations, legs, attempts, convergence, cap hits, the
window's first-round detectors after the carry of earlier commits, the frame after the window,
the local solution and the local committed columns; per shot the committed faults and the predicted observables. `vectors.json`
describes every array and carries the npz checksum; `load_vectors` reads and checks them.

`fixed_netlist window` writes one window's decoding graph for a hardware description
(`netlist.json` + `netlist.npz`): check and fault nodes with their rounds, H~ in CSR and CSC,
quantised priors, commit and convergence masks, the carry map (the rows of the next window's
first round each committed column flips) and the frame matrix. For a bivariate-bicycle code it
also records the lift: the window graph is invariant under Z_L × Z_M′ acting on the check index
c = M′a + b, so every node has an orbit and a shift and the graph is a base graph with shifts.
`read_netlist` checks the checksum, CSR against CSC, the carry map against H~ and the base graph
against H~. `fixed_netlist lift` verifies the lift of a whole artifact and counts check and
fault orbits per round (the gross code: a 26 × 990 base graph with 5,832 base edges, 2 check
orbits and 82 fault orbits per bulk round).

Committed fixtures: `test/fixtures/bb18_choi_r9/fixed_*`, five goldens with test vectors
(int4 exact, int4 uniform with flag and cap 150, int6 exact with deferral, int5 uniform with
deferral and cap 200, int4 one window as wide as the shot), 100 shots each, γ₀ = 0.125,
30 + 12 × 20 iterations, γ tables of 3 rows from seed 17.

## Tests

```sh
uv run pytest
# Also run the export tests (needs the reference group) and the parity checks against
# relay_bp's reference circuits:
RTD_RELAY_TESTDATA=/path/to/relay/tests/testdata/bicycle_bivariate uv run --group reference pytest
```

The campaign tests need the reference group, because `init` exports artifacts. Most of them run
the driver against a stand-in decoder script. `test_campaign_paired_stop.py` judges the paired
stopping criterion (one pair, a list, null) on hand-built campaign directories and compares the
single-pair and null forms with a recorded golden, `tests/data/campaign_paired_stop_golden.json`;
only its end-to-end runs need the reference group. That makes retries, interruptions, resumption and
the stopping criteria deterministic and fast. `test_real_decoder_small_campaign` decodes small
bb18 campaigns in each mode with `build/release/src/rtd_decode` (two workers on CPUs 10 and 11).
It is skipped when that binary has not been built. The gross-code split test runs only when
`data/artifacts/gross_choi_p0.003` and `data/shots/gross_choi_p0.003` exist.

The decoder tests (`test_bindings.py`, `test_sinter_adapter.py`) are skipped until the extension
is built. They reproduce the whole-shot `relay_f32` golden, all fifteen window goldens and the five
fixed-point goldens bit for bit, check that workers, bit-packing, shot-by-shot decoding, streams
and the sinter path (records included) all give the batch's results, that a selection policy's
confidence and history reach the batch records and the stream alike, that flagged shots and cap
hits are logged, that a decoder pinned to another module build refuses to run, and that bad input
is refused. `test_dem.py` checks the conversion
against the committed artifacts.

The window-reference tests (`test_window_ref.py`, `test_window_toy.py`) run the plan properties
(Lee et al.'s detector rule gives the same column and commit sets as the s(j) rule; the commit
sets partition the columns; merged columns are never committed), the policies, the dump round
trip and the hand-derived toy without relay_bp. With the reference group they also check the
whole-shot identity: with W ≥ Rt, the windowed decode equals the `relay_f32` and `relay_f64`
goldens bit for bit. They also re-decode the first shots of every windowed golden.
