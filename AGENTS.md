# AGENTS.md

zo0k is a reference implementation of the zero-knowledge IOPP of the paper
*Zero-Knowledge IOPPs for Constrained Interleaved Codes*: a polynomial
commitment for multilinear evaluations over GF(2^128) and GF(2^256), with
BLAKE3 or Keccak-256 Merkle trees and Fiat–Shamir. It is cryptographic code:
a change that keeps honest proofs verifying can still break soundness or
zero knowledge. The simulator exists to catch exactly that.

## Rules

1. **No comments.** Do not write `//`, `///`, `//!` or `/* */` comments, doc
   attributes, or `#` comments in the TOML files. Names and structure carry the
   meaning. `lint` rejects every comment.
2. **No unit tests.** `#[test]`, `#[cfg(test)]`, doctests, and a `tests/`
   directory are banned, and `lint` rejects them. Correctness is established by
   the deterministic simulator `sim` (`src/bin/sim/`), and only by it.
3. **Every change goes through the simulator.**
   - New behaviour → a new check in `src/bin/sim/`. A per-seed check goes in the
     module that matches the area. A deterministic check goes in
     `one_off_checks()` in `main.rs`.
   - A bug found by any means → first reproduce it as a failing seed or check.
     Append the seed to `sim/seeds.txt` as `0x<seed> <check> (<cause>)`, fix the
     bug, and keep the seed forever. A bug must never be able to come back.
   - Prove that a new check works: break the code it guards on purpose, confirm
     `sim` fails, then restore the code.
4. **Zero dependencies.** `[dependencies]` stays empty: std only. If a
   primitive is missing, write it here, verify it against an independent
   reference (published vectors, a differential run against the reference
   implementation in a throwaway harness), and record known answers in
   `src/bin/sim/vectors.rs`.
5. **TigerStyle.**
   - Assert preconditions, postconditions and invariants, at least two per
     non-trivial function.
   - Every loop has an explicit bound.
   - Avoid recursion; where it exists, it must have a fixed depth bound.
   - Functions are at most 70 lines, with simple and explicit control flow.
   - Sizes are explicit.
   - Do not allocate in hot loops.
   - Names put units and qualifiers last: `len_bytes`, `threads_max`.
   - Handle every error. Adversarial input (proofs, parameters, JSON, arguments)
     gives `Err` and never panics.
6. **Trunk only.** Work on `main`. When `./check.sh` passes, commit and push to
   `main`. There are no pull requests and no hosted CI: the agent thread is the
   review, and `./check.sh` on the agent's own machine is the CI.

## Commands

| Purpose | Command |
|---|---|
| Everything, before every push | `./check.sh` |
| Fast loop: lint, fmt, clippy, carry-less sim | `./check.sh quick` |
| Simulator gate | `cargo build --release --bins && target/release/sim` |
| Reproduce one seed | `cargo run --release --bin sim -- --seed 0x…` |
| Hunt for bugs with fresh seeds | `target/release/sim --hunt --seconds 600` |
| Remove comments automatically | `cargo run --release --bin lint -- --fix` |

`sim` starts the `zo0k` binary for its CLI checks, so build it with
`--bins`. The gate takes about 30 s. `./check.sh` also runs:

- the simulator again over the portable field arithmetic, which uses no
  carry-less multiplication;
- the MSRV check, `cargo +1.88`.

## Invariants that need care

- **Proofs are byte-reproducible.** For a fixed `Rng::from_seed`, proof
  bytes are a function of the statement and the seed only, whatever the thread
  count. Do not change the proof format or the transcript order unless you mean
  to. If you do, mirror the change in `src/bin/sim/composed.rs` (an independent
  composed prover). The sim reports when the two are out of step.
- **Known-answer vectors are the only guard against a wrong but
  self-consistent hash.** Never weaken or regenerate `src/bin/sim/vectors.rs`
  from the code under test.
- **The sim uses pub internals.** These are claim, code_switch, mask, witness,
  sumcheck, base_case, oracle, merkle, transcript, parallel, blake3, keccak and
  json. If you rename one or narrow its visibility, update the sim.
- **Paper outputs must match.** `paper/scaling-tables.tex` and
  `paper/frontiers.csv` must equal the output of `zo0k paper`, which the sim
  checks.

## Map

| Paper / concern | Module |
|---|---|
| Layout, encodings, communication | `parameters`, `costs` |
| Mask polynomial | `mask` |
| Steps 1–3: mask sum, batching, sumcheck | `sumcheck` |
| Step 4: code switch | `code_switch` |
| Step 5: base case | `base_case` |
| Claims, succinct linear forms | `claim` |
| Reed–Solomon (additive FFT, Cantor basis) | `reed_solomon`, `witness` |
| Merkle commitments | `merkle`, `oracle` |
| Round-by-round soundness | `soundness` |
| Fiat–Shamir, proof of work | `transcript` |
| Query counts and proof-of-work tuning | `tuning` |
| Fields | `field` |
| Hashes | `hash`, `blake3`, `keccak` |
| Parallelism, JSON, entropy | `parallel`, `json`, `entropy` |
| Protocol composition | `lib.rs` |
| CLI | `src/bin/zo0k/` |
| Simulator | `src/bin/sim/` |
| Lint | `src/bin/lint/` |
