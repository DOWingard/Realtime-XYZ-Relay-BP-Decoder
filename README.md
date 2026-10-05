# rtd — an XYZ-Relay-BP decoder for qLDPC codes

`rtd` is a C++23 implementation of Relay-BP (Müller et al., *Improved belief propagation is
sufficient for real-time decoding of quantum memory*, arXiv:2506.01779), built to decode the
circuit-level noise of the [[144,12,12]] bivariate bicycle ("gross") code, with a Python pipeline
that generates the decoding problems and the reference outputs it is tested against.

- **Bit-exact** with the reference implementation `relay_bp` on identical inputs: the same
  correction ê, convergence flag, iteration count, solution weight, per-leg record and final
  marginals, bit for bit, for plain min-sum, memory BP and Relay-BP in 32- and 64-bit floats.
- **About 10× faster per iteration** on one core, and faster still with a thread team on one
  decoding problem (numbers below).
- A batch driver with a fully specified configuration file, per-shot timing, logical error rates
  with confidence intervals and a provenance record for every run.
- Sliding-window decoding, a selection policy with confidence signals, and an integer
  (fixed-point) arithmetic matching IBM's FPGA format, each reproduced against a Python reference
  and benchmarked (see [Benchmarking](#benchmarking)).

## The problem being solved

A syndrome-extraction circuit repeated R times produces m detectors (parity checks of measurement
outcomes that are 0 in the absence of faults) and has n possible elementary faults. Their relation
is a binary matrix H ∈ F₂^{m×n}: H_ij = 1 when fault j flips detector i. Fault j occurs
independently with prior probability p_j. A shot yields a syndrome σ ∈ F₂^m, and the decoder must
return a correction ê ∈ F₂^n with Hê = σ that is likely, meaning a low weight
W(ê) = Σ_{j: ê_j = 1} λ_j with λ_j = ln((1 − p_j)/p_j). A second matrix A ∈ F₂^{k×n} maps faults
to flips of the logical observables; decoding succeeds when Aê equals the true flips.

For the gross code decoded in both X and Z bases at once (XYZ decoding) at physical error rate
p = 3·10⁻³ over 12 rounds: m = 1872, n = 71,280, and H has 419,904 nonzeros.

**Min-sum belief propagation** works on the Tanner graph of H, which has one node per row (check),
one per column (variable) and one edge per nonzero. Each edge carries a message from variable to
check, μ, and one back, η. One iteration updates every check, then every variable:

- check i: η_{i→j} = (−1)^{σ_i ⊕ (number of negative μ into i, excluding j)} · α · min_{j'≠j} |μ_{j'→i}|
- variable j: M_j = Λ_j + Σ_i η_{i→j}, and μ_{j→i} = Λ_j + Σ_{i'≠i} η_{i'→j}, computed as a prefix
  sum plus a suffix sum over the checks in ascending row order (never as M_j − η_{i→j}, which
  rounds differently)
- hard decision: ê_j = [M_j ≤ 0]; stop when Hê = σ.

**Memory BP** replaces the prior evidence λ_j by Λ_j = (1 − γ_j)λ_j + γ_j M_j from the previous
iteration. **Relay-BP** runs a sequence of such legs: leg 0 with a scalar γ₀, then each relay leg
with fresh per-variable strengths γ_j drawn from an interval that includes negative values, every
edge message reset to λ but the marginals M carried over. Every leg that satisfies Hê = σ offers a
solution; the lowest-weight one is kept, and the relay stops after S solutions (XYZ-Relay-BP-5:
S = 5, γ₀ = 0.125, γ ∈ [−0.24, 0.66), 80 iterations in leg 0 and 60 in each of up to 600 relay
legs).

## Requirements

- GCC 14 or newer (tested with GCC 16) or Clang with libstdc++; the kernels use
  `std::experimental::simd`, `std::expected`, `std::mdspan` and `std::stacktrace`.
- CMake ≥ 3.28, Ninja, OpenSSL (libcrypto, for SHA-256 verification of inputs).
- Fetched at configure time with pinned checksums: nlohmann/json 3.12, GoogleTest 1.17,
  Google Benchmark 1.9.4.
- The Python pipeline (`python/`, see its README) produces the inputs.

## Building and testing

```sh
cmake --preset release && cmake --build --preset release   # -O3 -march=native
ctest --preset release                                     # unit, property and golden tests
```

| Preset | Purpose |
|---|---|
| `debug` | Debug build, tests |
| `release` | Optimised build, tests, benchmarks |
| `profile` | Optimised with symbols and frame pointers, for `perf` |
| `asan` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `tsan` | ThreadSanitizer |
| `tidy` | Clang build with clang-tidy on every first-party target (configuration and the reason for each disabled check in `.clang-tidy`) |

The `Makefile` wraps these (`make test`, `make asan`, `make tsan`, `make tidy`, `make fmt`).

Every translation unit that runs a decoder kernel is compiled with `-ffp-contract=off
-fno-fast-math`: fusing a·b + c into one FMA or reassociating a sum would change the rounding and
break bit-exactness. Vectorisation is used only where it cannot change a result: within a row
(minimum and parity are exact in any order) or across independent columns.

### Tests

- `test/unit`: every component on hand-built inputs, including each validation failure; the kernels
  and the full decoder against a deliberately naive reference decoder on random graphs; the harness.
- `test/golden`: the committed fixture `test/fixtures/bb18_choi` (the [[18,4,3]] code, 100 shots)
  with nine reference outputs, checked bit for bit under every edge layout, column order and
  executor; and fifteen sliding-window goldens (`test/fixtures/bb18_choi*/window_*`, written by the
  Python windowing reference with `relay_bp` inside) reproduced array by array, by the stream
  decoder directly and by `rtd_decode`, whose sliding mode must also reproduce every whole-shot
  golden with one window (W ≥ R + 1).
- Fixed point: the policy against the paper's worked values, the backend against a naive integer
  decoder in every format, layout, order and executor, and five integer-emulator goldens
  (`test/fixtures/bb18_choi_r9/fixed_*`, written by `python -m rtd.fixed_golden`) reproduced by
  the stream decoder and by `rtd_decode`. Larger emulator goldens run with
  `RTD_FIXED_GOLDEN_DIR=data/golden/fixed build/release/test/rtd_tests --gtest_filter='FixedGoldenExternal.*'`.
- Gross-code goldens are too large to commit. Generate them with the Python `rtd-golden` tool, then
  `make gross-golden ARTIFACT=data/artifacts/gross_choi_p0.003 GOLDEN=data/golden/gross_choi_p0.003`.

## Decoding shots: `rtd_decode`

```sh
build/release/src/rtd_decode \
    --artifact data/artifacts/gross_choi_p0.003 \
    --shots data/shots/gross_choi_p0.003 \
    --config configs/xyz_relay5_f32.json \
    --out data/runs/relay5 --workers 6 --cpus 0-5 --warmup 2
```

`--first/--count` select a range of shots, `--workers` runs that many decoders concurrently,
`--cpus` pins their threads, `--save-decodings` also stores every ê (with sliding windows: the
correction the windows committed), `--no-verify` skips input checksums. `--record-solutions N`
(0–20, 0 = off) records the first N solutions of every decode (of every window, with sliding
windows), `--save-solution-supports` (needs N > 0) also stores their supports, and
`--save-commits` stores the faults each window commits (sliding windows only; with a whole-shot
spec it is rejected, since the returned ê is already in `decodings` or among the saved supports). Logs are JSON lines on
stderr (timestamp, level, context, message, run id, and error and stack trace on failures). Exit
codes: 0 success, 2 usage or configuration, 3 unreadable or inconsistent input, 4 output, 5 decode
failure, 1 internal error.

The decoder is described by a JSON spec in which **every field is required**, so a run's
configuration is always written down in full (`configs/` has examples):

```json
{
  "version": 2,
  "policy": "f32",
  "backend": "cpu",
  "layout": "row_major",
  "column_order": "wavefront",
  "block_rows": 64,
  "executor": {"type": "serial"},
  "alpha": {"rule": "constant", "value": 1.0},
  "gamma0": 0.125,
  "pre_iter": 80,
  "set_max_iter": 60,
  "num_sets": 600,
  "stopping": {"rule": "after_n_converged", "count": 5},
  "gamma_source": {"type": "uniform", "seed": 1, "low": -0.24, "high": 0.66},
  "window": {"mode": "whole_shot"}
}
```

- `version`: `2`, or `3` with a required `selection` object (see
  [Selection policy and confidence](#selection-policy-and-confidence-spec-version-3); `"selection":
  null` decodes exactly as version 2). A version-1 spec (no `version`, no `window`) is rejected with
  a message; adding `"version": 2` and `"window": {"mode": "whole_shot"}` gives the identical decoder.
- `window`: `{"mode": "whole_shot"}` decodes each shot as one problem. `{"mode": "sliding",
  "width": W, "commit": C, "converge_rounds": C′, "boundary": "exact" | "uniform", "on_failure":
  "commit_anyway" | "defer" | "flag", "max_deferrals": D, "iteration_cap": I | null}` describes a
  sliding-window schedule (1 ≤ C < W, C ≤ C′ ≤ W, D = 0 unless `defer`, I ≥ 1), with every field
  required and no other field allowed. See [Sliding-window decoding](#sliding-window-decoding).
- `policy`: `f32` | `f64`. `backend`: `cpu` (`cuda` is reserved; see below).
- `arithmetic`: the number format, in place of `policy` (a spec names exactly one of the two):
  `f32`, `f64`, or a fixed-point format `int4.2.8` | `int5.2.8` | `int6.2.8`. Fixed point runs on
  the cpu backend only, with `alpha` 1, 1 − 2^{−k} or adaptive with scaling 1. See
  [Fixed-point arithmetic](#fixed-point-arithmetic-intnsm).
- `layout`: `row_major` | `column_blocked`; `column_order`: `wavefront` | `degree_classes` |
  `natural`; `block_rows`: rows per wavefront block. These change speed only, never results.
- `executor`: `{"type": "serial"}` or `{"type": "team", "threads": T}` (T threads cooperate on each
  decode; with sliding windows each window shape's decoder has its own team).
- `alpha`: `{"rule": "constant", "value": a}` or `{"rule": "adaptive", "scaling": s}` for
  α(t) = 1 − 2^{−(t+1)/s}, t counting iterations from 0 within each leg.
- `gamma0`: memory strength of leg 0, or `null` for no memory term (plain min-sum).
- `pre_iter`, `set_max_iter`, `num_sets`: iterations of leg 0, iterations of each relay leg, and
  the number of relay legs.
- `stopping`: `after_leg0` (stop if leg 0 converged, otherwise run every relay leg),
  `after_n_converged` with `count` (leg 0 included), or `all_legs`.
- `gamma_source`: `{"type": "none"}`, `{"type": "uniform", "seed", "low", "high"}`,
  `{"type": "explicit", "path": "table.npy"}` (a [T, n] table; relay leg r uses row r mod T; with
  sliding windows only when the window plan has a single shape, i.e. W ≥ R + 1), or
  `{"type": "explicit_shapes", "directory": "dir"}` (sliding only: `dir/shape_<i>.npy`, a
  [T, n_i] table for every window shape i, as `rtd_window_plan` numbers them). Relative paths are
  resolved against the spec's directory. The uniform generator hashes (seed, stream, leg) with
  SplitMix64 into a xoshiro256+ state, where the stream is the shot's index in its shots file, so
  results do not depend on how shots are split across workers.

The output directory receives `success`, `iterations`, `legs`, `best_leg`, `weight`, `decode_ns`,
`predicted_observables`, `logical_failure` (and optionally `decodings`) as `.npy` arrays, one
entry per shot, and `run.json` last: the spec, input checksums, git revision, compiler and flags,
CPU, thread placement, and a summary with the block error rate P_block (the fraction of shots in
which any logical observable is predicted wrong) and its Wilson 95% interval, two per-cycle logical
error rates derived from it over R rounds,

  ler_per_cycle            p_L = 1 − (1 − P_block)^{1/R}
  ler_per_qubit_per_cycle  p_L = [1 − (2(1 − P_block)^{1/k} − 1)^{1/R}] / 2

and nearest-rank quantiles (p50, p90, p99, p99.9, max) of iterations, legs and decode time. The
first is the whole-memory rate per cycle that Bravyi et al. (2024) and Müller et al. (2025) plot;
the second further assumes the k logical qubits fail independently and is about k times smaller.
`run.json` also records the `window` object and the recording options.

**Recorded solutions.** Every relay leg that satisfies Hê = σ offers a solution. With
`--record-solutions N` the first N of them per decode (in leg order, leg 0 included) are
recorded, in arrays whose first axis is the shot, second the window (K = 1 for a whole-shot
decode) and third the slot:

| array | dtype, shape | content |
|---|---|---|
| `sol_count` | u32 [S, K] | solutions found, which may exceed N (they are counted, not stored) |
| `sol_leg` | u32 [S, K, N] | leg of the solution |
| `sol_iterations` | u32 [S, K, N] | iterations of all legs up to and including that leg |
| `sol_weight` | f64 [S, K, N] | W(ê); +∞ in empty slots |
| `sol_class` | u64 [S, K, N] | logical class A·ê as a k-bit mask, bit o = observable o |
| `sol_hash` | u64 [S, K, N] | h = 0, then h ← splitmix64(h ⊕ j) over the support in ascending order |
| `sol_size` | u32 [S, K, N] | \|ê\| |
| `returned_class` | u64 [S, K] | class of the returned ê, also when no leg converged |
| `solsup_ptr`, `solsup_idx` | u64 [S·K·N + 1], u32 | with `--save-solution-supports`: slot q = (shot·K + window)·N + s holds the columns `solsup_idx[solsup_ptr[q]:solsup_ptr[q+1]]`, ascending; empty slots are empty |

Slots at or past `sol_count` are empty (weight +∞, every other field 0). Classes do not include
the artifact's `observables_bias` (no exported gross-code artifact has one). k must be at most 64.
The summary gains a `solutions` block (slots, distribution of `sol_count`, decodes with more
solutions than slots). Without the flag nothing changes, bit for bit. Legs that land on the same
ê share a hash, so distinct solutions can be counted, and per-class quantities (for example
Σ e^{−W(ê)} over the distinct solutions of each logical class) follow from these arrays alone.

### Sliding-window decoding

A memory experiment measures its checks for R noisy rounds and then reads out the data qubits,
which counts as a noiseless round R + 1 = Rt; each round has M detectors (rows of H). A fault j
touches detectors of at most two consecutive rounds, and s(j) is the earlier one. Instead of
decoding all Rt·M rows at once, a sliding-window decoder decodes window k = 0, 1, … over the W
rounds starting at t_k = 1 + k·C, and commits only the faults with t_k ≤ s(j) < t_k + C:

- **Window problem.** Rows = rounds [t_k, t_k + W); columns = faults with t_k ≤ s(j) < t_k + W.
  Faults of the window's last round lose their rows in the next round, which has not been
  measured; those that become indistinguishable are merged into one column with prior
  p = p₁(1 − p₂) + p₂(1 − p₁) (the probability that an odd number of them occurred). The first
  window that reaches the readout is final: it covers every remaining round and commits every
  remaining fault.
- **Carry.** The window syndrome is σ restricted to the window's rows with the syndrome of every
  fault committed so far removed (σ ⊕ H·c), which changes only the first round of the next window.
  Committed faults update the correction c and the logical frame ℓ̂ = A·c; ℓ̂ at the end is the
  prediction.
- **Convergence rows.** The inner decoder stops when Hê = σ holds on the first C′ rounds of the
  window (all rows of a final window).
- **Boundary.** `exact` cuts every window from the true matrix, so the first window, the bulk and
  the last window have different shapes. `uniform` uses the shape of window 1 for every position,
  with rows past the readout read as zero; a committed bulk column that has no counterpart at a
  position (before round 1 or past the readout) is *virtual*: it changes the residual syndrome and
  the frame but is not recorded as a fault.
- **Non-convergence.** `commit_anyway` commits leg 0's final ê; `flag` does the same and flags the
  shot; `defer` commits nothing and decodes the same start round again with width W + C (up to D
  times), and flags the shot if every attempt fails. An `iteration_cap` bounds the iterations of
  one window decode, summed over its legs.

`rtd_decode` builds the window plan (every distinct window shape, its Tanner graph and priors,
and the schedule of window positions and deferral attempts) once from the artifact before it
loads the shots, logs `window plan built` (shapes with rows, columns, edges, merged and committed
columns, the number of positions K, the schedule length, virtual commits, build time, memory), and
records the same in `run.json` under `plan`. A plan that cannot be built (rows not grouped in
rounds of equal size, a fault spanning more than two rounds, no bulk window for `uniform`, k > 64)
is rejected with exit code 2 before anything is decoded or written. Every worker owns one relay
decoder per window shape. γ draws: with `uniform`, window k, deferral attempt a of shot s uses the
stream s + k·2³² + a·2⁵⁶ of a generator of the shape's width, so a plan with one window (W ≥ Rt)
reproduces a whole-shot run bit for bit; `explicit_shapes` gives each shape its table.

Per-shot arrays keep their names with windowed meanings: `success` = every window converged,
`iterations` and `legs` = sums over the windows and their attempts, `best_leg` = −1, `weight` =
Σ λ over the committed faults, `decode_ns` = the whole shot (all rounds pushed, all windows
decoded), `predicted_observables` = the frame, `decodings` = the committed correction c (dense
[S, n]). Added, with K window positions per shot:

| array | dtype, shape | content |
|---|---|---|
| `win_iterations`, `win_legs` | u32 [S, K] | summed over the position's attempts |
| `win_attempts` | u8 [S, K] | attempts made (a + 1 for the committed attempt a); 0 for a position that an earlier final window already decided (defer only), whose record is otherwise empty |
| `win_converged`, `win_cap_hit` | u8 [S, K] | of the committed attempt |
| `win_weight` | f64 [S, K] | W(ê) of the committed attempt's solution, +∞ if it did not converge |
| `win_committed_weight` | f64 [S, K] | Σ λ over the committed columns of that solution |
| `win_unexplained` | u32 [S, K] | detectors of the committed rounds that the commits leave unexplained |
| `win_flagged` | u8 [S, K] | the window failed and its policy flags that; with a version-3 low-confidence action (`on_low` `defer` or `flag`) also a failed window under `commit_anyway`, and a converged low-confidence window that `flag` commits or `defer` commits with no attempt left (`win_converged` tells the two cases apart) |
| `win_virtual` | u32 [S, K] | virtual commits (uniform boundary) |
| `win_decode_ns` | u64 [S, K] | wall time of the position's decode calls, attempts included |
| `flagged` | u8 [S] | some window was flagged (either case of `win_flagged`) |
| `commit_ptr`, `commit_faults` | u64 [S·K + 1], u32 | with `--save-commits`: cell q = shot·K + k committed the faults `commit_faults[commit_ptr[q]:commit_ptr[q+1]]`, global column indices, ascending |

Recorded solutions are per window (the solution arrays' K axis): a solution's class is the XOR,
over its support, of the committed columns' observable masks (columns the window does not commit
count 0), so it is the logical class the window would assign to the rounds it commits, and
`returned_class` is the window's change of the frame. Saved supports are window-local column
indices of the committed attempt's shape (the plan dump of `rtd_window_plan` maps them to global
columns). The summary gains `windows`: positions, windows decoded and skipped, distributions of
iterations, legs and decode time per window and of the largest window per shot, and Wilson
intervals on the converged, cap-hit, deferred and flagged fractions, with the unexplained and
virtual totals.

Window events are logged as JSON lines with the shot, window, attempts and the policy's action:
`window did not converge; policy applied` (warn), `window did not converge; decoded again with a
wider window` (warn), `deferral limit reached` (error), `window hit the iteration cap` (warn). The
first 20 of each kind are logged individually; the rest are counted, the counts appear in the
progress lines, and one `(run total)` line per kind reports the total when decoding finishes. A
round of the wrong size is an error of that shot's decode.

With the `team` executor each window shape's decoder owns a team of T threads, pinned to the same
CPUs. Idle team threads spin for 2¹⁴ pause instructions after their last job (about 0.3 ms on the
development machine) and then sleep, so the teams of shapes not in use cost no CPU time; only the
first 0.3 ms after a switch between shapes overlaps with the next team's work.

### Selection policy and confidence (spec version 3)

A decode that finds several solutions (converged legs) must return one of them, may stop before the
relay stopping rule, and can say how much it trusts its answer. A version-3 spec describes this in
a `selection` object, applied to every decode (every window, with sliding windows):

```json
"selection": {
  "rule": "lowest_weight",
  "stop": {"rule": "fixed"},
  "capacity": 20,
  "confidence": {"signal": "gap", "threshold": 2.0, "single_class": "high",
                 "extra_legs": 0, "on_low": "none"},
  "history": {"lengths": [1, 2, 3, 5], "signals": ["gap", "weight", "commit_weight"]}
}
```

Every field is required (`confidence` and `history` may be `null`). The rules see the first
`capacity` (1–20) solutions in leg order. A solution is identified by its support hash; its class
is A·ê.

- `rule` picks the returned solution. `lowest_weight` (the default decoder's choice, and the only
  rule that never replaces the controller's answer): the solution of least W(ê), the earliest on
  ties. `class_sum`: the class L with the largest Z̃_L = Σ exp(−(W(ê) − W*)) over its distinct
  solutions (W* the least weight; earliest class on ties), represented by its lightest solution.
  `largest_agreement`: the class with the most distinct solutions (ties: the lighter, then the
  earlier class).
- `stop` ends a decode early, checked after each solution. `fixed` never does (the relay rule
  alone decides). `agree`/`agree_distinct` with `count` m: when m solutions (converged legs, or
  distinct solutions) share the class of the lightest one. `gap` with `threshold` t: when the gap
  Δ = W₂ − W* is defined and Δ ≥ t, W₂ being the least weight outside the lightest solution's class.
  `gap_extend` with `count` n₀ and `threshold` t: after n₀ solutions, unless another class lies
  within t of the lightest (then it continues until the gap reaches t). Any rule other than
  `fixed`, `extra_legs` > 0 and an `on_low` action need `stopping` `after_n_converged` with
  count ≤ `capacity`; with `extra_legs` > 0 also count + `extra_legs` ≤ `capacity`.
- `confidence` rates each decode by one `signal`: `gap` (Δ; low when Δ < t; a decode whose solutions
  all lie in one class has no gap and counts as low or high as `single_class` says), `agreement`
  (distinct solutions of the lightest one's class / distinct solutions; low when < t), `weight`
  (W*), `first_legs` and `first_iterations` (legs and iterations to the first solution) and `q_supp`
  (Q⁽²⁾ = √(Σ_c L_c²)/Λ over the connected components c of the lightest solution's support, two
  columns connected when they share a detector, L_c = Σ λ over the component, Λ = Σ λ over all
  columns); these four are low when they exceed t. A decode with no solution is always low. A low
  decode runs up to `extra_legs` further legs once the stopping rule is met, ending as soon as a
  new solution makes it no longer low. Every leg converges at most once, so with count +
  `extra_legs` ≤ `capacity` the rules see every solution of the extension, and the confidence
  describes the solution the decoder returns (a decoder built without that check runs no more
  extra legs than the rules have free slots). A cap (`iteration_cap`) that ends an extension does
  not count as a cap hit: the stopping rule was already met. With sliding windows, `on_low` acts
  on a low window: `defer` decodes it again
  with a wider window (it uses the deferral attempts, so it needs `on_failure` `defer` and
  `max_deferrals` ≥ 1; a low window with no attempt left is committed and flagged), `flag` commits
  it and flags it, `none` only records it. With `defer` or `flag`, a window that did not converge
  (no solution, the lowest confidence) is flagged when committed, also under `commit_anyway`.
- `history` (sliding windows only) reports after every window a value computed from that window
  and the L − 1 before it, for every listed L (1–64) and signal, the form of a real-time
  post-selection rule; the decoder never aborts. The windows' decode confidences combine as
  independent problems: `gap` the least defined Δ, `agreement` the product, `weight`, `first_legs`,
  `first_iterations` sums, `q_supp` √(Σ Σ_c L_c²)/Σ Λ, `density` fired detectors / detectors, each
  undefined when some window in the range had no solution; `commit_weight` is Σ λ over the union
  of the faults the windows committed and `commit_q_supp` its Q⁽²⁾ on the global problem.

Values are computed in a fixed order of IEEE operations (sums in slot or column order from 0.0,
λ = log((1 − p)/p) per column), so another implementation can reproduce them bit for bit. The
decoder allocates nothing per decode for any of this. Added outputs, with K window positions per
shot (K = 1 for a whole-shot decode) and L history lengths:

| array | dtype, shape | content |
|---|---|---|
| `conf_found`, `conf_seen`, `conf_distinct`, `conf_classes` | u32 [S, K] | solutions found, seen by the rules, distinct, distinct classes |
| `conf_best_class`, `conf_second_class` | u64 [S, K] | class of the lightest solution; class holding W₂ |
| `conf_weight`, `conf_gap`, `conf_agreement` | f64 [S, K] | W* (+∞ with no solution), Δ (NaN unless defined), agreement |
| `conf_gap_state` | u8 [S, K] | 0 no solution, 1 one class only (Δ undefined), 2 Δ defined |
| `conf_first_legs`, `conf_first_iterations` | u32 [S, K] | legs and iterations to the first solution |
| `conf_class_sum_class`, `conf_class_sum_top`, `conf_agreement_class` | u64, f64, u64 [S, K] | the class-sum choice and its Z̃; the largest-agreement choice |
| `conf_q_supp`, `conf_q_sum_sq`, `conf_q_total`, `conf_components` | f64, f64, f64, u32 [S, K] | Q⁽²⁾, Σ_c L_c², Λ, components |
| `conf_syndrome_ones`, `conf_syndrome_rows` | u32 [S, K] | fired detectors and detectors of the decoded syndrome |
| `conf_decided_class`, `conf_decided_leg` | u64, u32 [S, K] | the rule's choice and the leg of its representative |
| `conf_extra_legs` | u32 [S, K] | extra legs run for low confidence |
| `conf_score`, `conf_low`, `conf_stopped_early` | f64, u8, u8 [S, K] | the signal oriented so that larger is worse (+∞ with no solution, NaN without a value, e.g. the gap of a one-class decode, whose category is `conf_gap_state`); low; ended by `stop` before the relay rule was met |
| `conf_low_deferrals` | u32 [S, K] | deferrals made for low confidence (sliding windows) |
| `hist_<signal>`, `hist_state` | f64, u8 [S, K, L] | history values; the combined gap state |

A position that ran no decode keeps empty values (counts 0, weight +∞, NaN). The summary gains a
`confidence` block (gap states, low-confidence decodes, early stops, extra legs, low-confidence
deferrals, decisions that differ from the lowest-weight one). Without a selection policy nothing
changes, bit for bit.

### Fixed-point arithmetic (intN.S.M)

`"arithmetic": "int4.2.8"` (or `int5.2.8`, `int6.2.8`) runs the decoder in the integer format of
IBM's FPGA Relay-BP decoder (Maurer et al., arXiv:2510.21600), bit for bit as the Python emulator
`rtd.fixed_ref` computes it. BP in the log-likelihood domain is unchanged by a common scale, so
every log-likelihood (priors λ, messages, marginals) is multiplied by S and rounded to an integer
with N magnitude bits and a sign, range ±(2^N − 1). A memory strength γ is stored as
β_int = round((1 − γ)·M) ∈ [0, 2M] and γ·x is computed as x − β⊗x with the truncating
shift-and-add product β⊗x = sign(x) Σ_{bits b of |x|} ⌊β_int 2^b / M⌋. One variable update is
Λ = sat(β⊗λ + M̃ − β⊗M̃) with M̃ = sat(M), σ = Λ + Σ μ exactly, ν_k = sat(σ − μ_k), ê = [σ ≤ 0];
a check update scales the two smallest magnitudes by α = 1 − 2^{−k} as x − (x >> k) (adaptive:
k = t + 1, t restarting at every leg). A solution's weight is Σ λ_int / S. Rounding is half away
from zero; details the paper leaves open (saturation limits, accumulator width, the α schedule,
+∞ priors) are listed with their reasons in `include/rtd/core/fixed_arith.hpp`. Messages are
stored as bytes, and the row-major check pass runs on 16-byte vectors. The float policies are
unchanged by this. Whole-shot and sliding decodes, every executor and every layout give the same
results.

A fixed-point format combines with a version-3 `selection` policy. The rules then compare the
decoder's own solution weights, Σ λ_int / S: W*, Δ, the class sums Z̃ and the `weight` signal are
in those quantised units, so a `gap` threshold means the same distance in log-likelihood up to the
rounding and saturation of each λ (|λ_int| ≤ 2^N − 1). Q⁽²⁾, Λ and `commit_weight` are computed from the unquantised
λ = log((1 − p)/p) of the problem, as for the float formats.

## Replaying recorded solutions: `rtd_select_replay`

```sh
build/release/src/rtd_select_replay --artifact data/artifacts/gross_choi_p0.003_xz/z \
    --run data/runs/rec20 --slots 5 --stop gap --threshold 2.0 --out data/runs/rec20_s5_gap
```

Reads a run made with `--record-solutions N --save-solution-supports` (whole-shot or sliding; the
window plan is rebuilt from the run's spec) and replays every decode through the selection rules
as if the run had kept only the first S ≤ N solutions and stopped by the given rule (`--count`
and `--threshold` as in the spec). It writes `rep_<value>.npy` [S, K]: every confidence value
above over the first S solutions (`seen`, `state`, `weight`, `best_class`, `gap`, `second_class`,
`distinct`, `agreement`, `first_legs`, `first_iterations`, `class_sum_class`, `class_sum_top`,
`agreement_class`, `q_supp`, `q_sum_sq`, `q_total`, `components`) and the stopped run's outcome:
`stop_solutions`, `stop_decision` (class of the lightest solution among them, or the returned class
when there is none), `stop_iterations` and `stop_legs` (those of the last solution used, or the
recorded decode's totals when it found fewer than S), and `stop_by_rule`. A replay of more
solutions than the recording's stopping count is refused, since later legs never ran.

## Window plans: `rtd_window_plan`

```sh
build/release/src/rtd_window_plan --artifact data/artifacts/gross_choi_p0.003 \
    --width 12 --commit 8 --converge 12 --boundary exact --on-failure flag --max-deferrals 0 \
    --out data/plans/gross_12_8
```

Builds the window plan of an artifact for one window specification, as `rtd_decode` does, and
writes it out: `plan.json` (the specification, Rt, M, K, every shape's rows, columns, edges and
merged columns, and every placement's window, attempt, shape, first round, rounds, committed
rounds and finality) and, per shape i, `shape_<i>_H_indptr.npy` / `shape_<i>_H_indices.npy`
(local H as CSR, u32), `shape_<i>_priors.npy` (f64), `shape_<i>_commit.npy` and
`shape_<i>_converge.npy` (u8 per local column and row), `shape_<i>_class.npy` (u64 commit class
per local column); per placement (k, a), `placement_<k>_<a>_columns.npy` (the global column of
each local column; 0xFFFFFFFF for a virtual one) and, for the exact boundary, the members of merged
columns as CSR (`placement_<k>_<a>_members_ptr.npy`, `_members.npy`). Local columns are ordered by
their smallest global member and local rows by global row, so with W ≥ Rt the single window is the
whole problem in its own order.

## Benchmarks: `rtd_bench`

```sh
build/release/src/bench/rtd_bench --artifact data/artifacts/gross_choi_p0.003 \
    --shots data/shots/gross_choi_p0.003
```

`check_pass/*` and `variable_pass/*` time one pass of each kernel over the whole graph for every
layout and column order; `iteration/*` reports seconds per min-sum iteration for the serial
executor and pinned teams of 2, 3 and 6 threads; `relay5_decode/*` times complete
XYZ-Relay-BP-5 decodes. Any Google Benchmark flag (`--benchmark_filter=...`) may follow.

`build/release/src/bench/rtd_bench_fixed --artifact A --shots S` compares the number formats on
the calling thread (pin it with `taskset`): `iteration/<format>` is plain min-sum (100 iterations,
α = 1), `relay5/<format>` is Relay-BP-5 with α = 1 − 2^{−t}; each reports seconds per iteration
and per edge update, for `f32`, `int4.2.8` and `int6.2.8`.

## Decoding in memory: `rtd_api` and the Python module

`rtd_api` (`include/rtd/api`, built with the harness) runs the same decoders as `rtd_decode` on
arrays in memory instead of files:

- `api::parse_spec(json, base_dir)` reads a decoder spec (version 2, or version 3 with any
  selection policy; any number format, f32, f64 or a fixed-point intN.S.M; cpu backend).
- `api::Problem::create(ProblemArrays, graph_options)` validates and copies H and A (CSR), the
  priors and, optionally, each detector's round and the syndrome and observable biases of pruned
  p = 1 faults (the problem decodes σ ⊕ b_σ and reports ℓ̂ ⊕ b_ℓ).
- `api::WholeShotDecoder` and `api::WindowedDecoder` decode batches of unpacked or bit-packed shots
  with a fixed number of worker threads and return every array `rtd_decode` writes (predictions,
  per-shot and per-window records, corrections, committed faults, recorded solutions and, under a
  selection policy, the `conf_*` and `hist_*` arrays in `BatchResult::confidence`). Shot s uses
  γ stream `stream_offset + s`, so the outputs equal `rtd_decode`'s on the same shots, byte for
  byte; the tests compare them array by array (plain, selection-policy and fixed-point specs,
  whole shot and windows) and against the window and fixed-point goldens.
- `api::WindowedDecoder::open_stream()` returns a `Stream` fed round by round (`push_round`,
  `push_final`, `window_ready`, `decode_next`), with its own inner decoders; under a selection
  policy each commit's record carries its confidence, the low-confidence action applies, and
  `history()` gives the per-window signal over the last L windows.

The batch driver of `rtd_api` is its own (the harness's reads artifacts and shots from files), so
a change to what `rtd_decode`'s workers record must be made in `src/api` as well; the tests that
compare the two array by array catch any difference.

Errors are `std::expected<…, api::ApiError>` values with a code (`invalid_spec`,
`invalid_problem`, `plan_rejected`, `gamma_source`, `invalid_input`, `stream_state`, …) and a
message.

With `-DRTD_ENABLE_PYTHON=ON` (off by default) CMake also builds the nanobind extension
`rtd._native` over `rtd_api`; `-DRTD_PYTHON_OUTPUT_DIR=<repo>/python/src/rtd` places it in the
Python package, where `rtd.RelayDecoder`, `rtd.WindowedDecoder` and `rtd.sinter_decoders()` use it
(see `python/README.md`). The option compiles the static libraries as position-independent code
and changes nothing else.

## Design

```
include/rtd/core   decoder: graph, priors, arithmetic policies, kernels, backends, relay controller
include/rtd/window sliding windows: time structure, window spec, window plan (shapes and
                   schedule), stream decoder
include/rtd/io     .npy, manifests, SHA-256, artifact / shots / golden loaders
src/harness        rtd_decode: spec, command line, batch runner and its whole-shot and
                   sliding-window workers, solution recording, result writer, JSON logger,
                   system facts
src/api            rtd_api: problems, decoders and streams on arrays in memory
src/python         the nanobind extension rtd._native (RTD_ENABLE_PYTHON)
src/tools          rtd_window_plan
src/bench          rtd_bench
src/cuda           device backend interface (RTD_ENABLE_CUDA)
test/              unit, property and golden tests; the bb18 fixture
python/            circuit generation, sampling, export, reference outputs (whole shot, windows,
                   fixed point), campaign driver and statistics, Python decoders
```

- **Layers.** `rtd_core` depends on the standard library only and never allocates, logs or throws
  inside a decode. `rtd_window` (standard library and `rtd_core` only) holds the window logic once:
  `WindowPlan` is immutable and shared by all workers; `StreamDecoder` is fed round by round
  (`push_round`, `push_final`) and decodes the oldest ready window on `decode_next`, with any inner
  decoder that satisfies the `SyndromeDecoder` concept, so a batch run and a real-time stream
  execute the same code. `rtd_io` loads and validates inputs; `rtd_harness` owns threads, timing,
  configuration and output. Errors cross every boundary as `std::expected` values carrying enough
  context to log.
- **Backend boundary: one leg.** `RelayDecoder` runs the leg schedule, tracks the best solution and
  applies the stopping rule; a backend (any type satisfying the `LegBackend` concept) owns all
  mutable state and runs one leg. Plain min-sum and memory BP are the zero-relay-leg
  configurations of the same code path.
- **Arithmetic policy.** The kernels are written once against a policy type that fixes the number
  format and the operations (`FloatArith<float>`, `FloatArith<double>`, `FixedArith<N, S, M>`), so
  the fixed-point formats run the identical schedule, tie rule and summation order.
- **One message array.** Rows partition the edges and so do columns, so the check pass and the
  variable pass can each update the same array in place.
- **Row-major layout.** Each row's messages are contiguous and padded to 16 slots with the largest
  float, which is neutral for both the minimum and the sign parity, so the check pass runs on whole
  aligned vectors with no index loads and no tail. The first iteration of a leg reads the priors
  through the slot's column instead of first writing them into every slot.
- **Wavefront column order.** Columns are grouped by the block of rows containing their last row,
  then by degree, then by their row list. After checking a block, the decoder immediately updates
  the columns that block completes, while their messages are still in L1/L2. The order of
  operations inside every node is unchanged, so this is bit-identical to the flooding schedule.
- **Degree-specialised variable kernels.** Each run of equal-degree columns dispatches to a kernel
  with the degree known at compile time (up to 16), fully unrolled.
- **Fused convergence test.** The variable pass appends each column with M_j ≤ 0 to a support list
  and flips its rows in a bitset, so testing Hê = σ costs O(|ê|·d) per iteration instead of a pass
  over the edges; W(ê) is summed over the sorted support in ascending column order.
- **Thread team.** For a single decode, a persistent team splits each iteration's row blocks by
  work, with a spinning barrier (tens of nanoseconds) between the passes; columns that straddle two
  threads' rows are updated after the first barrier. Results are identical for every thread count.
- **Column-blocked layout** (`column_blocked`): the alternative in which each degree run's k-th
  messages are contiguous, making the variable pass SIMD across columns; it is slower overall on the
  development machine because its check pass must gather, and is kept, tested, for other hardware.
- **CUDA.** `CudaBackend` fixes the device backend's interface; built with `-DRTD_ENABLE_CUDA=ON`
  it compiles as plain C++ and its factory reports that the backend is not implemented yet.

## Benchmarking

Everything below was measured with this repository's tools. Experiments run as campaigns
(`rtd-campaign`, see `python/README.md`): shots are sampled from the circuit in fixed-size chunks,
decoded by `rtd_decode` with a different γ stream per chunk, and appended with atomic markers, so a
campaign can be stopped and resumed without losing a chunk. Several decoder configurations ("arms")
decode the same shots, which makes every comparison between arms paired. Intervals are 95%: Wilson
for a rate, a log-normal interval on the ratio of two paired failure counts (V below, from the
discordant shots) and Katz for a ratio of unpaired rates. A tail quantile or a rate is called
"judged" only when at least 100 events lie beyond it; fewer is marked provisional. Circuits are
generated with `relay_bp_compat=True` (the final round's idle noise placed as in the reference
implementation), the noise model of Müller et al. at physical error rate p.

Speed was measured on an AMD Ryzen 5 3600 (6 cores in two 3-core complexes, each with its own 16 MB L3),
performance governor, boost on, GCC 16.2 with `-O3 -march=native -ffp-contract=off`, threads
pinned. The decoding problem is the gross code's XYZ matrix at p = 3·10⁻³ over 12 rounds
(m = 1872, n = 71,280, 419,904 edges).

### Correctness

- Bit for bit against `relay_bp` on every golden: the bb18 fixture (nine configurations, each
  under eight layout / column-order / executor combinations) and 741 gross-code shots (plain
  min-sum, memory BP, adaptive α, Relay-BP-5 in f32 and f64, single-column syndromes), serial and
  with thread teams. This holds for builds with both GCC 16 and Clang 22.
- The full suite (436 tests at the last full run) runs clean under AddressSanitizer + UndefinedBehaviorSanitizer and
  under ThreadSanitizer, and clang-tidy reports nothing on the sources, tests and benchmarks.

### Agreement with published results

The new numbers have no ground truth of their own, so each stage was first checked against
published results on the same circuits. "Pass" means the published value lies inside our 95%
interval after allowing the reading error (a factor of about 1.25) for values taken off a plot;
published rates are per cycle unless stated.

| quantity | ours | published |
|---|---|---|
| XYZ-Relay-BP-5, p = 4·10⁻³ | 1.08·10⁻⁴ [0.89, 1.30]·10⁻⁴ | 1.5·10⁻⁴ (Müller), 1.2·10⁻⁴ (Maurer) |
| XYZ-Relay-BP-5, p = 5·10⁻³ | 2.24·10⁻³ [1.85, 2.70]·10⁻³ | 2.2·10⁻³, 2.6·10⁻³ | 
| XZ-Relay-BP-5, p = 2, 3, 4, 5·10⁻³ | 3.0·10⁻⁶, 5.8·10⁻⁵, 6.9·10⁻⁴, 5.7·10⁻³ | 3·10⁻⁶, 5.0·10⁻⁵, 7·10⁻⁴, 6·10⁻³ (Müller) |
| XZ rate against S = 1, 2, 3, 5, 7, 9 at p = 3·10⁻³ | 1.0, 0.74, 0.64, 0.58, 0.50, 0.47 (·10⁻⁴) | 1.0, 0.7, 0.6, 0.47, 0.43, 0.39 (·10⁻⁴) |
| mean iterations of the X decoder, p = 1, 2, 3, 4, 5·10⁻³ | 36.6, 64.4, 146.6, 633, 3250 | 37, 65, 145, 620, 3800 |
| mean iterations, XYZ, p = 3·10⁻³ | 336.1 [334.9, 337.5] | 330.8 | 
| Z-type failure per 12-round shot, p = 3·10⁻³ | 3.58·10⁻⁴ [3.41, 3.76]·10⁻⁴ | 3.87·10⁻⁴ [3, 5]·10⁻⁴ (Beverland) |
| sliding-window matching vs global matching, rotated surface code d = 7, 9, 11, R = 20, 100, 200 | all nine global points | Skoric et al. |
| sliding (3, 1), BP+LSD, gross code, T = 6, 9, 12, 15, 18, 21 | commits identical to Lee et al.'s on 100,000, 12,000, 38,000, 6,400, 5,600 and 4,800 shots; P/T 1.70·10⁻³ at T = 6 | 1.61·10⁻³ |
| whole-shot BP+LSD, T = 12, p = 3·10⁻³ | 0.957% [0.938, 0.976] per shot | 0.95% | 

- The 1.6% excess of the XYZ mean iteration count is a property of the γ draws, not of the
  decoder: `relay_bp`'s own draws on the same shots give 334.7 [330.4, 339.6] and ours 333.9, a
  paired difference of +0.85 [−2.5, 4.1].
- XYZ at p = 3·10⁻³ has few failures (17 in 180,000 shots; 100 in the 1.58·10⁶-shot `sinter` run),
  so its agreement is the weakest of the table; p = 1.5·10⁻³ and 2·10⁻³ are out of reach by direct
  sampling (about 3·10⁷ and 2·10⁸ XYZ shots for 100 failures).
- **Not reproduced.** The Z-type cutoff curves of Maurya et al. (S = 1, 1 + 300 legs, p = 10⁻³):
  our failure rate at an iteration cutoff is 1.5–2.2× above their Fig. 2 at every judged cutoff
  from 80 to 1,000 iterations, while the fraction of decodes still running after 170 and 250
  iterations is below their Fig. 27 (7.2·10⁻⁵ against 2.8·10⁻⁴ and 3.2·10⁻⁵ against 5·10⁻⁵) and
  matches it at 500 (9·10⁻⁶ against 8·10⁻⁶). The two published figures are not consistent with each
  other, a variant of the setup ("half x") made no difference (V = 0.97 at cutoff 80), and the
  discrepancy is untraced.
- **Exact maximum likelihood on a small code.** On the [[18,4,4]] bivariate-bicycle code (2¹⁸ error
  patterns, 16 logical classes, unequal priors) the measured ML failure rate 0.2992 [0.2964, 0.3020]
  equals the exact 0.2981 at p ∈ [0.05, 0.12]; Relay-BP's P(fail | Δ) matches the exact posterior
  in every Δ bin (χ² p = 0.67 for S = 5, 0.32 for S = 20); and for S = 20 the class-sum decision is
  the ML class in 99.88% of the shots in which the search found every class whose exact posterior is
  at least 0.2 of the ML class's.
- **Internal identities** (all pass): one window wider than the shot reproduces the whole-shot
  decoder on 168,000 shots (V = 1); recorded solutions recomputed from their saved supports agree;
  the replay at S = 5 of an S = 9 run equals a direct S = 5 run; `rtd_decode`'s sliding mode equals
  the Python window reference array by array on every golden.

### Accuracy against the paper

100,000 shots, XYZ-Relay-BP-5 in f32 (γ₀ = 0.125, 80 iterations in leg 0 and 60 in each relay
leg, up to 600 relay legs, stop after 5 solutions, γ uniform on [−0.24, 0.66)):

| | this decoder | Müller et al. 2025 |
|---|---|---|
| block failures | 8 of 100,000, P_block = 8.0·10⁻⁵ (95%: 4.1·10⁻⁵ – 1.6·10⁻⁴) | |
| logical error rate per cycle, 1 − (1 − P_block)^{1/12} | 6.7·10⁻⁶ (95%: 3.4·10⁻⁶ – 1.3·10⁻⁵) | (7 ± 1)·10⁻⁶ |
| mean BP iterations per decode | 335.6 ± 1.2 | 330.8 ± 0.5 |

All eight failures converged to a correction in the wrong logical class. One shot did not
converge in 601 legs. The iteration count is heavy-tailed (median 281, p99 1163, maximum 36,080);
the top 0.1% of shots alone add about 7 to the mean, so the quoted standard error understates its
uncertainty. The mean sits 1.5% above the paper's.

### Speed

Medians of five repetitions (`rtd_bench`), wavefront column order unless stated:

| | serial | team 2 | team 3 | team 6 |
|---|---|---|---|---|
| min-sum iteration | 392 µs | 219 µs | 155 µs | 113 µs |
| min-sum iteration, degree order | 445 µs | 240 µs | 168 µs | 149 µs |
| XYZ-Relay-BP-5 decode (256 shots, mean 371 iterations) | 167 ms | 96.9 ms | 66.1 ms | 52.3 ms |

- One pass over the edges in the row-major layout: check pass 107 µs (3.9·10⁹ edges/s), variable
  pass 297 µs (1.4·10⁹ edges/s). The fused iteration (392 µs) is faster than the two passes run
  back to back because the variable pass reads each block's fresh messages from cache.
- Column-blocked layout: check pass 390 µs, variable pass 56 µs, 446 µs together. Its fast
  variable pass does not make up for the gathering check pass on this machine.
- A relay iteration (450 µs serial) costs more than a plain one because it also mixes the memory
  term into every bias.
- A team of 6 spans both core complexes, so its barriers cross between the two L3 caches. That is
  why it scales less well than a team of 3.
- `relay_bp` took 3.4–4.3 ms per iteration on the same machine and problem, about 10× slower
  than one core here.
- Throughput: six independent serial decoders on cores 0–5 finished 26.8 decodes per second in
  the 100,000-shot run, with a mean of 224 ms per decode (p50 181 ms, p99 840 ms) as they shared
  the caches.

In real-time terms, the fastest single decoder (a team of 6) needs about 4.4 ms of wall time per
syndrome round. The paper's real-time budget is about 600 iterations per 12-round window with
1 µs rounds, which assumes 20 ns FPGA iterations. One iteration here takes about 5,700 times
that long.

### Whole-shot decoding against the number of rounds

XYZ-Relay-BP-5, p = 3·10⁻³, one run of R rounds per shot (so one decode of R + 1 rounds of
detectors). XZ decoding handles the X and Z sectors as two separate problems of 936 × 8,784 each
and a shot fails when either does; rates are per cycle, 1 − (1 − P_block)^{1/R}. "Not converged" is
the fraction of shots in which no leg satisfied Hê = σ.

| mode, R | shots | failures | rate per cycle | not converged |
|---|---|---|---|---|
| XZ, 12 | 3,000,000 | 2140 | 5.95·10⁻⁵ [5.70, 6.20]·10⁻⁵ | 9.4·10⁻⁵ |
| XZ, 24 | 510,000 | 925 | 7.56·10⁻⁵ | 4.3·10⁻⁴ |
| XZ, 48 | 90,000 | 420 | 9.74·10⁻⁵ | 1.7·10⁻³ |
| XYZ, 12 | 180,000 | 17 | 7.9·10⁻⁶ [4.9, 12.6]·10⁻⁶ | 0 of 180,000 |
| XYZ, 24 | 16,500 | 6 | 1.5·10⁻⁵ (too few failures to judge) | |

The non-convergence rate of XZ grows 4.6× (Katz 95%: 3.9–5.5) from R = 12 to 24 and 17.7× to 48,
while the rate per cycle grows by only 1.27× and 1.64×: a longer shot loses most to shots that never
find any solution within the leg budget, and the number of iterations per shot grows faster than
the number of rounds (the mean per round is 1.05× larger at R = 24 and 1.40× at R = 48 than at
R = 12). A second, independent XYZ run through `sinter` on 1,583,792 shots (100 failures) gives
5.3·10⁻⁶ per cycle [4.3, 6.4]·10⁻⁶ against the paper's 7·10⁻⁶ [6, 8]·10⁻⁶ (the intervals overlap;
the point estimate is about 25% low). That run also checks the `sinter` adapter against `rtd_decode`
on 300 shots: every output is identical.

### Sliding-window decoding

Windows of W rounds that commit C rounds each (see [Sliding-window decoding](#sliding-window-decoding)).
V is the paired ratio failures(windowed) / failures(reference) on the same shots and the same
random γ draws, with a 95% interval from the discordant counts (n₁₀ shots only the windowed decoder
gets wrong, n₀₁ only the reference). V = 1 means no loss from windowing. XZ decoding, p = 3·10⁻³,
exact window boundaries, `commit_anyway` unless stated.

**Reference and bit-exactness.** A window wider than the shot (W = 13, R = 12) reproduces the
whole-shot decoder bit for bit on all 168,000 shots (V = 1 exactly). The Python window reference
combined with BP+LSD commits exactly the correction of Lee, English and Bartlett
(arXiv:2510.05795) on 182,800 shots. At R = 12 the reference is whole-shot decoding (110 failures
in 168,000 shots, 5.46·10⁻⁵ per cycle); at R = 48 it is the (24, 12) window, because whole-shot
decoding itself degrades with R (V of whole-shot over (24, 12) is 1.09 [0.89, 1.33]).

| (W, C) | V at R = 12 | V at R = 48 |
|---|---|---|
| (3, 1) | 4.34 [3.60, 5.22] | 3.45 [2.82, 4.22] |
| (4, 1) | 1.36 [1.13, 1.65] | 1.10 [0.89, 1.36] |
| (6, 1) | 0.95 [0.80, 1.14] | 0.81 [0.66, 1.00] |
| (6, 4) | 1.38 [1.14, 1.67] | 1.40 [1.14, 1.72] |
| (8, 4) | 1.10 [0.91, 1.33] | 0.85 [0.69, 1.05] |
| (10, 4) | 0.93 [0.77, 1.11] | 0.73 [0.58, 0.92] |
| (12, 8) | 1.11 [0.93, 1.33] | 0.77 [0.62, 0.96] |
| (16, 12) | | 0.84 [0.70, 1.01] |

- **Buffer.** The loss vanishes once the buffer B = W − C reaches about 4 rounds (the smallest
  buffer from which every V interval reaches 1: at R = 12, 4 for C = 4 and C = 8 and 5 for C = 1;
  at R = 48, 4, 2, 2 and 3 for C = 4, 8, 12 and 1).
  Where V is below 1 at R = 48 the windowed decoder beats the reference, because it avoids
  whole-shot non-convergence.
- **Rate per cycle is roughly independent of R.** (12, 8) gives 6.05·10⁻⁵ per cycle at R = 12 (122
  failures in 168,000 shots), 7.09·10⁻⁵ at R = 24 (119 in 70,000) and 6.4·10⁻⁵ at R = 48 (77 in
  25,000); the whole-shot rate grows over the same range (above).
- **Cost.** At R = 12 (12, 8) runs 2.6× the iterations per shot of the whole-shot decoder (769 against
  293 as recorded), because its windows overlap; at R = 48 it runs 0.94× those of the (24, 12)
  reference (3,420 against 3,654). The largest window of a shot costs about 300 iterations at
  R = 12 and 460 at R = 48.
- **Boundary.** The `uniform` boundary (one window shape for every position, needs R ≥ 21) gives
  83 failures against 77 for `exact` at R = 48, (12, 8), 25,000 shots; no difference is resolved.
- **Convergence rows C′.** Requiring only the committed rounds to be satisfied (C′ = C = 8) is bad:
  V = 2.09 [1.74, 2.52] at R = 12 and 1.57 [1.27, 1.94] at R = 48. C′ = 10 matches C′ = 12
  (V = 1.01 [0.84, 1.21] and 0.81 [0.66, 1.00]) at 11% (R = 12) to 13% (R = 48) more iterations
  than C′ = 8.
- **Non-convergence.** `defer` with up to 2 retries at width W + C gives V = 0.97 [0.81, 1.17] at
  R = 12 and 0.61 [0.48, 0.77] at R = 48 (61 failures against 100 for the reference; 9 of 25,000
  shots are left flagged). `flag` (commit leg 0's answer and mark the shot) keeps 55 of 24,991 shots
  failing at R = 48.
- **Iteration caps are costly.** A hard cap per window multiplies V at R = 12 by 80 (100
  iterations), 28 (333), 9.1 (1,000) and 4.6 (2,000); at R = 48 by 57, 22, 7.4 and 3.9. The mean
  count only falls from 769 to 547 at a cap of 1,000, since the cost is in the tail.

XYZ confirmation of the best windows (R = 12 and 48, p = 3, 4, 5·10⁻³) is still running; the XZ
choices carried over are (12, 8) with C′ = 10, `defer` with 2 retries, exact boundaries at R = 12 and
uniform at R = 48.

### Real-time operation of windows

Measured iteration counts of (12, 8) windows at R = 48 (bulk windows, larger of the two halves;
the leg budget caps one decode at 36,080 iterations), run through a queue model in which each
window arrives every C = 8 rounds of τ_rd and takes iterations × τ_it. The iteration time is not
measured here: it is IBM's 24 ns per iteration on an FPGA, so these are statements about what the
measured iteration counts imply for such hardware.

| p | median | p99 | p99.9 | mean | shortest sustainable τ_rd (mean load < 1): no cap / cap 500 |
|---|---|---|---|---|---|
| 2·10⁻³ | 87 | 1,056 | 5,040 | 163 | 0.49 µs / 0.40 µs |
| 3·10⁻³ | 202 | 5,634 | 36,032 | 526 | 1.58 µs / 0.76 µs |
| 4·10⁻³ | 555 | 36,076 | 36,080 | 2,528 | 7.6 µs / 1.17 µs |

- With 1 µs rounds, p = 2·10⁻³ and no cap, the queue is stable (load 0.49) but a stream of windows
  has a delay of 555 rounds at p99 and 980 at p99.9 (maximum 1,738) from the tail of a few
  iteration-heavy windows; a cap of 500 iterations per window brings these to 16, 20 and 33
  rounds. The mean iteration count alone does not bound delay.
- The largest p at which a 1 µs round time is sustainable is about 2.6·10⁻³ without a cap and
  3.6·10⁻³ with a cap of 500 (log-log interpolation).
- The cap is not free: at p = 3·10⁻³, R = 48, a cap of 500 iterations per window gives a failure
  rate 14.6× [12.1, 17.7] that of the uncapped (24, 12) reference (V, from the cap rows above).
- On the development CPU the same window decodes at 68 µs per iteration with six workers (indicative
  only: other jobs were running), against the 24 ns assumed.

### Which solution to return, and how much to trust it

Whole-shot XZ decoding at R = 12, p = 3·10⁻³, recording the first S = 20 solutions of every decode
(1,000,000 shots) and replaying the selection rules over the recording.

- **More solutions help, slowly.** Failure rate per cycle of the lowest-weight rule: 1.00·10⁻⁴ at
  S = 1 (66 iterations per shot), 5.4·10⁻⁵ at S = 5 (294), 4.2·10⁻⁵ at S = 20 (1,085).
- **Class-sum beats lowest-weight only slightly.** Choosing the logical class with the largest
  Σ exp(−W(ê)) over its solutions gives V = 0.988 [0.978, 0.997] against the lowest-weight
  solution at S = 5 and 0.954 [0.931, 0.978] at S = 20. Choosing the class with most solutions is
  worse (V = 1.14 at S = 5).
- **Where failures come from** (S = 5): of the failed shots, 12.7% have no converged leg, 50.6% found
  the true logical class among the solutions but returned another, and 36.7% never found it.
  A forced-decode experiment that would give a clean ML-style certificate fails its own control
  (agreement 0.852 against a 0.95 floor: forcing perturbs the decoder), so no such certificate is
  claimed.
- **A confidence signal that works for a subset.** The weight gap Δ = W₂ − W* between the lightest
  solution and the lightest one of any other logical class, used to discard the lowest-confidence
  shots (500,021 report shots, S = 5): discarding 0.2% of shots cuts the failure rate per shot from
  6.4·10⁻⁴ to 2.1·10⁻⁴ (319 to 107 failures, a factor of 3); discarding more gains nothing, since
  the gap does not flag the remaining ~100 failures. The signal choice and the threshold were fixed on a
  separate half of the shots.
- **Switching decoders inside a window** (more relay legs for windows with a low gap; XZ, R = 48,
  (12, 8), 26,000 shots, vs the same schedule without it). Fraction of windows switched
  0.1% / 0.3% / 1.1% / 4.4% / 10% / 100% gives V = 1.00 / 0.97 / 0.95 / 0.90 [0.84, 0.97] /
  0.87 [0.80, 0.94] / 0.85 [0.78, 0.93] at 2 / 7 / 28 / 159 / 455 / 4,247 extra iterations per
  shot. Deferring a low-confidence window to a wider one gave no V resolved below 1. Whether the
  switching keeps up with the syndrome stream depends on the time per iteration: at 24 ns per
  iteration the weak-window load is already 97% of the window period, and only the first two rows
  fall under the backlog bound.
- **Comparison with BP+LSD and with Lee et al.'s signals** (1,000,000 identical shots, XZ,
  R = 12, p = 3·10⁻³). BP+LSD whole-shot fails on about 1.2% of shots against 0.07% for
  Relay-BP-5. Discarding 1% of shots by BP+LSD's own cluster signal Q⁽²⁾ leaves 5,166 failures;
  applying the same signal to Relay-BP-5's solutions leaves 219 (ratio 0.042 [0.037, 0.048]).
  Reproducing Lee et al.'s own curves (Fig. 4(b), p = 3·10⁻³, BP+LSD, T = 12), the failure rate of
  the kept shots matches theirs for the cluster signal (1.17·10⁻⁴ against 1.15·10⁻⁴ at 10%
  discarded), for the correction weight (1.79·10⁻⁴ against 1.5·10⁻⁴ at 30%) and for the detector
  density (2.10·10⁻³ against 2.10·10⁻³ at 70%), with the same ranking: density is the weakest, the
  cluster signal the best.
- **No signal wins everywhere.** Comparing signals by the failure rate of the kept shots against the
  correction weight at 1% / 5% / 20% discarded: the gap has ratio 0.36–0.50 [95% interval] at 1%,
  0.45–0.66 at 5% and is not defined at 20%, while a search-effort signal is better at 5% and 20% (0.25–0.40 and 0.11–0.27); a combination of signals beats all
  of them but is not computable online. Switching rates of 0.3–10% lie inside the crossover.

### Fixed-point arithmetic

Paired against float32 on the same shots and γ draws, Relay-BP-5 with α = 1 − 2^{−t}, whole-shot
R = 12; ratio of block rates per cycle (95% interval). Published values are read off the log plot
of Maurer et al. (arXiv:2510.21600), so they carry a factor of about 1.25.

| mode, format | p | f32 failures / int failures | ratio int / f32 | published |
|---|---|---|---|---|
| XZ, int4.2.8 | 0.002 | 71 / 106 (1,956,900 shots) | 1.49 [1.20, 1.87] | 1.4 |
| XZ, int4.2.8 | 0.003 | 104 / 137 (151,800) | 1.32 [1.09, 1.59] | 1.1 |
| XZ, int4.2.8 | 0.004 | 111 / 115 (13,900) | 1.04 [0.88, 1.22] | 1.15 |
| XYZ, int6.2.8 | 0.004 | 95 / 148 (92,400) | 1.56 [1.26, 1.93] | 1.3 |

All measured ratios fall inside the published bands. The XYZ int6 point at p = 3·10⁻³ is still
running. int4 also needs fewer iterations (mean 201 against 244 at p = 3·10⁻³, XZ). With windows
((12, 8), uniform boundary, R = 24, 12,300 shots, cap 1,000 iterations per window) int4 against
float32 is V = 0.90 [0.77, 1.04]; the cap, not the number format, dominates the loss there (11.5×
the whole-shot float32 rate, 15 failures in the whole-shot arm).

**Hardware model.** IBM's synthesised window decoder for one sector of the gross code (the X half
of 12 + 1 cycles) uses 2,106,738 LUTs, 51.6% of an XCVU19P, with 5-bit messages and a 12 ns clock at
2 clocks per iteration (24 ns). A resource and timing model calibrated on that point (lift
structure measured from our exported graphs: the window graph of 864 rows, 8,424 columns and
29,232 edges is a 72-fold lift of a base graph of 12 rows, 117 columns and 406 edges, which gives
`f` = the number of lift copies processed in parallel) says, for 1 µs rounds on one device: an int4
X or XZ decoder meets the round time only fully parallel (f = 72) at a 12 ns clock, and for
f = 18, 24, 36 or 72 at a 4 ns clock; fully parallel it takes 0.49 (X) or 0.98 (XZ) of the device.
An XYZ int6 decoder takes 9.1 devices fully parallel and meets 1 µs on none at either clock. These
are modelled, not synthesised: per-edge costs, width scaling and the 4 ns clock are assumptions
(flagged as such in the model output); folding over time, which could divide the logic by up to W
more, is not modelled.

### Status

Finished and quoted above: the anchors, whole-shot decoding against R, the whole XZ window study
(buffer, boundary, C′, policies, caps, tails and the queue model), the soft-output analyses on
recorded solutions, XZ fixed point at p = 2, 3, 4·10⁻³ and the windowed int4 point. Running or not
yet analysed: the XYZ confirmation of the window study at R = 12 and R = 48, the XYZ int6 point at
p = 3·10⁻³, and and quiet single-worker timing cells for the round-count study. Not built: the CUDA
backend (the interface and a stub compile; the development machine has no NVIDIA GPU) and any FPGA implementation; the hardware numbers above are from the model.
