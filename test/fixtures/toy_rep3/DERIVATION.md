# Toy fixture: windowed decoding of a 3-bit repetition code, worked by hand

Every value in `expected.json` is derived here from the definitions alone. The fixture's tests
then run the windowed decoder with an exact minimum-weight inner decoder and require the same
values.

## 1. The problem

Data bits q0, q1, q2 start in 0. Two checks, c0 = q0 ⊕ q1 and c1 = q1 ⊕ q2, are measured in
rounds 1 to 6. Before each of these rounds every data bit flips with probability p_d = 0.01, and
each check result is misread with probability p_m = 0.02. Round 7 reads the data bits without
error and recomputes both checks from them. So Rt = 7 rounds, M = 2 detectors per round, m = 14.

Detector (r, c) is the XOR of check c's values in rounds r and r − 1 (round 0 is the known 0).
Row index: (r, c) ↦ 2(r − 1) + c. The observable is the readout of q0.

A flip of data bit q before round r changes every later value of the checks containing q, so it
fires only the detectors of round r for those checks. A misread of check c in round r changes
only that one value, so it fires (r, c) and (r + 1, c). Columns (from `rtd-export`; stim sorts the
faults by their detectors), for r = 1 … 6 and b = 5(r − 1):

| column | fault | rows (detectors) | observable |
|---|---|---|---|
| b + 0 | q1 flips before round r | (r, c0), (r, c1) | – |
| b + 1 | c0 misread in round r | (r, c0), (r + 1, c0) | – |
| b + 2 | q0 flips before round r | (r, c0) | flips |
| b + 3 | q2 flips before round r | (r, c1) | – |
| b + 4 | c1 misread in round r | (r, c1), (r + 1, c1) | – |

n = 30. Every column has s(j) = r (its earliest round) and touches only rounds r and r + 1. No
column has s = 7. Notation below: q1(r) = column b+0, m0(r) = b+1, q0(r) = b+2, q2(r) = b+3,
m1(r) = b+4.

Weights λ = ln((1 − p)/p):
- data flip: λ_d = ln(0.99/0.01) = ln 99 ≈ 4.5951
- misread: λ_m = ln(0.98/0.02) = ln 49 ≈ 3.8918
- merged column (below): p_x = 0.02·0.99 + 0.01·0.98 = 0.0296, λ_x = ln(0.9704/0.0296) ≈ 3.4899

Order used repeatedly: λ_x < λ_m < λ_d < λ_m + λ_x < 2λ_m < λ_d + λ_x < 2λ_x + λ_m.
(3.49 < 3.89 < 4.60 < 7.38 < 7.78 < 8.09 < 10.87)

## 2. Window layouts

A window starting at round t with width W (non-final) holds rows of rounds [t, t + W), the columns
with s ∈ [t, t + W), and commits those with s ∈ [t, t + C). Columns with s = L = t + W − 1 lose
their rows in round L + 1. In the last round L: m0(L) keeps only (L, c0), the same support as
q0(L), so they merge into one column x0(L) (representative = the smaller index, m0(L) = 5(L−1)+1;
prior folded in ascending index: 0.02·(1 − 0.01) + 0.01·(1 − 0.02) = 0.0296). Likewise m1(L) keeps
only (L, c1), the support of q2(L): merged x1(L), representative q2(L) = 5(L−1)+3, prior
0.01·(1 − 0.02) + 0.02·(1 − 0.01) = 0.0296. q1(L) keeps both rows of round L and merges with
nothing. The final window (first with t + W − 1 ≥ 7) covers rounds [t, 7], keeps every column
with s ≥ t unchanged and commits them all.

(W, C) = (3, 1): windows at t = 1, 2, 3, 4 (last rounds 3, 4, 5, 6) and the final one at t = 5
(5 + 2 ≥ 7) covering rounds 5–7. Commit sets: s = 1, s = 2, s = 3, s = 4, then s ≥ 5.

(W, C) = (4, 2): windows at t = 1 (rounds 1–4, commit s ∈ {1, 2}), t = 3 (rounds 3–6, commit
s ∈ {3, 4}), and the final one at t = 5 (5 + 3 ≥ 7) covering rounds 5–7, commit s ≥ 5.

Carry: before window k, the residual of its first round t_k is σ(t_k) ⊕ u_k, where
u_k = (H c)(t_k) is what the corrections already committed flip in round t_k (only columns with
s = t_k − 1 can contribute). `expected.json` lists both the residual
(`residual_first_round_before_decode`) and u_k (`carry`).

## 3. Case A: (3, 1), faults m0(2) = 6, q1(4) = 15, q0(6) = 27

σ: column 6 fires (2, c0), (3, c0); column 15 fires (4, c0), (4, c1); column 27 fires (6, c0).
σ = {(2,c0), (3,c0), (4,c0), (4,c1), (6,c0)} = rows 2, 4, 6, 7, 10. True observable: 1
(column 27 flips q0).

**Window 0** (t = 1, rounds 1–3, merged x0(3) = {11, 12}, x1(3) = {13, 14}). Residual of round 1:
(0, 0), carry (0, 0). Window syndrome {(2,c0), (3,c0)}. (2,c0) is touched by q1(2), m0(2), q0(2),
m0(1). Using m0(2) explains both detectors at λ_m ≈ 3.89. The next cheapest, q0(2) + x0(3), costs
λ_d + λ_x ≈ 8.09; anything using q1(2), m0(1) or more columns costs more still. Unique solution
{6}. Commit set s = 1: nothing. Frame 0.

**Window 1** (t = 2, rounds 2–4, merged x0(4) = {16, 17}, x1(4) = {18, 19}). Residual of round 2:
(1, 0), carry (0, 0). Syndrome {(2,c0), (3,c0), (4,c0), (4,c1)}. (4,c1) needs q1(4) (λ_d, also
explains (4,c0)), x1(4) (λ_x) or m1(3) plus a column cancelling (3,c1) (≥ λ_m + λ_x). With m0(2)
for the first two detectors: m0(2) + q1(4) = λ_m + λ_d ≈ 8.49 against m0(2) + x0(4) + x1(4) =
λ_m + 2λ_x ≈ 10.87. Without m0(2), (2,c0) needs q0(2) (λ_d) and (3,c0) then needs another
column (≥ λ_m), giving at least λ_d + λ_m + λ_x ≈ 11.98. Unique solution {6, 15}. Commit s = 2:
column 6. It flips (2,c0) and (3,c0) out of the residual. Frame 0.

**Window 2** (t = 3, rounds 3–5, merged x0(5) = {21, 22}, x1(5) = {23, 24}). σ(3) = (1, 0); the
committed column 6 flips (3, c0), so u = (1, 0) and the residual of round 3 is (0, 0): column 6
straddled the commit edge between rounds 2 and 3, and the carry cancels its second detector.
Syndrome {(4,c0), (4,c1)}: q1(4) at λ_d ≈ 4.60; q0(4) + q2(4) costs 2λ_d, m0(3) + q0(3) costs
λ_m + λ_d, m0(4) + m1(4) + x0(5) + x1(5) costs 2λ_m + 2λ_x. Unique solution {15}. Commit s = 3:
nothing. Frame 0.

**Window 3** (t = 4, rounds 4–6, merged x0(6) = {26, 27}, x1(6) = {28, 29}). Residual of round 4:
(1, 1), carry (0, 0). Syndrome {(4,c0), (4,c1), (6,c0)}. Round 4 as before: q1(4) at λ_d. (6,c0):
x0(6) at λ_x ≈ 3.49 (the window cannot tell m0(6) from q0(6)); q1(6) + x1(6) costs λ_d + λ_x;
m0(5) + q0(5) costs λ_m + λ_d. Unique solution {15, 26} (26 = representative of x0(6)), weight
λ_d + λ_x ≈ 8.09. Commit s = 4: column 15. Residual left: {(6, c0)}. Frame 0.

**Window 4** (final, t = 5, rounds 5–7, no merging). Residual of round 5: (0, 0), carry (0, 0).
Syndrome {(6,c0)}. m0(6) would also fire (7,c0), and nothing else touches round 7's c0 row, so it
cannot be used. q0(6) costs λ_d; q1(6) + q2(6) costs 2λ_d; m0(5) + q0(5) costs λ_m + λ_d. Unique
solution {27}. Commit all: column 27 flips the observable. Frame 1.

Final frame 1 = true observable: no logical failure.

## 4. Case B: (4, 2), faults m1(2) = 9, q0(4) = 17, m1(6) = 29

σ: 9 fires (2,c1), (3,c1); 17 fires (4,c0); 29 fires (6,c1), (7,c1).
σ = rows 3, 5, 6, 11, 13. True observable: 1 (column 17).

**Window 0** (t = 1, rounds 1–4, merged x0(4) = {16, 17}, x1(4) = {18, 19}; commit s ∈ {1, 2}).
Residual of round 1: (0, 0), carry (0, 0). Syndrome {(2,c1), (3,c1), (4,c0)}. (2,c1) and (3,c1):
m1(2) at λ_m; q2(2) + q2(3) costs 2λ_d. (4,c0): x0(4) at λ_x (q0(4) is inside it and cannot be
used separately); q1(4) + x1(4) costs λ_d + λ_x; m0(3) + q0(3) costs λ_m + λ_d. Unique solution
{9, 16}, weight λ_m + λ_x ≈ 7.38. Commit s ∈ {1, 2}: column 9 (s = 2, the last committed round;
its second detector (3,c1) lies in the next window). Frame 0.

**Window 1** (t = 3, rounds 3–6, merged x0(6) = {26, 27}, x1(6) = {28, 29}; commit s ∈ {3, 4}).
σ(3) = (0, 1); u = (0, 1) from column 9; residual of round 3 (0, 0). Syndrome {(4,c0), (6,c1)}.
(4,c0): q0(4) at λ_d (x0(4) no longer exists: s = 4 is not the last round here); m0(3) + q0(3)
or m0(4) + q0(5) cost λ_m + λ_d; q1(4) + q2(4) costs 2λ_d. (6,c1): x1(6) at λ_x; q1(6) + x0(6)
costs λ_d + λ_x; m1(5) + q2(5) costs λ_m + λ_d. Unique solution {17, 28}, weight λ_d + λ_x ≈
8.09. Commit s ∈ {3, 4}: column 17, which flips the observable. Frame 1.

**Window 2** (final, t = 5, rounds 5–7). Residual of round 5: (0, 0), carry (0, 0). Syndrome
{(6,c1), (7,c1)}. Only m1(6) touches (7,c1), so it is in every solution, and it explains both
detectors; adding anything else adds positive weight. Unique solution {29}. Commit: column 29.
Frame 1.

Final frame 1 = true observable.

## 5. Case C: (3, 1), faults m1(1) = 4, m1(2) = 9 (a carry that is not cancelled by σ)

σ: 4 fires (1,c1), (2,c1); 9 fires (2,c1), (3,c1); (2,c1) cancels. σ = rows 1, 5. True
observable 0.

**Window 0** (t = 1, rounds 1–3, merged x0(3), x1(3) = {13, 14}). Residual of round 1: (0, 1),
carry (0, 0). Syndrome {(1,c1), (3,c1)}. m1(1) + m1(2) costs 2λ_m ≈ 7.78; q2(1) + x1(3) costs
λ_d + λ_x ≈ 8.09; q2(1) + q2(2) + m1(2) or m1(1) + q2(2) + x1(3) cost more. Unique solution
{4, 9}. Commit s = 1: column 4, flipping (1,c1) and (2,c1) in the residual. Frame 0.

**Window 1** (t = 2, rounds 2–4). σ(2) = (0, 0), but u = (0, 1) from column 4, so the residual of
round 2 is (0, 1): the carry alone creates this detector. Syndrome {(2,c1), (3,c1)}: m1(2) at
λ_m; q2(2) + q2(3) costs 2λ_d; q2(2) + m1(3) + x1(4) costs λ_d + λ_m + λ_x. Unique solution {9}.
Commit s = 2: column 9. Frame 0. Residual now empty.

**Windows 2, 3, 4.** Window 2's round 3: σ(3) = (0, 1), u = (0, 1) from column 9, residual
(0, 0). Every later residual is empty, and the empty correction is the unique minimum (all
λ > 0). Nothing is committed. Frame 0 throughout.

Final frame 0 = true observable.

## 6. Checked by the code

`python/tests/test_window_toy.py` rebuilds the three cases with `rtd.window_ref` and the
brute-force inner decoder and asserts, window by window, the residual, the carry, the solution,
the committed columns, the frame, and that the minimum-weight solution is unique. The derivation
was written before the code was run; on the first run they agreed on every value, so no
correction was needed on either side. The same test also runs the cases with PyMatching as the
inner decoder (every column of this code touches at most two detectors, so matching is exact
here) and requires the same results.
