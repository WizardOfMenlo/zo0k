# zo0k

A reference implementation of the zero-knowledge IOPP of the paper
*Zero-Knowledge IOPPs for Constrained Interleaved Codes*, compiled into a
non-interactive argument for evaluations of a committed multilinear polynomial.

The crate implements the construction of the paper's section "Full
construction of the zero-knowledge IOPP" with Reed–Solomon codes, Merkle trees
without salts, and the Fiat–Shamir transformation. It is generic over the
field and the hash function, and provides the fields `GF(2^256)` and
`GF(2^128)` and the hash functions BLAKE3 and Keccak-256. The paper's
evaluation uses `GF(2^256)` and BLAKE3, which are the defaults. The crate also
contains the soundness and size analysis behind that evaluation, and
reproduces its tables and figure.

This is research software. It has not been audited, its field arithmetic is
not constant time, and the encoding of its proofs can change between versions.

## Quick start

```sh
cargo test --release
cargo run --release -- bench --num-variables 20        # standard parameters
cargo run --release -- bench --paper polylog-20        # a row of the paper's tables
cargo run --release -- bench --num-variables 20 --field gf2_128
cargo run --release -- analyze --paper polylog-20      # layout, errors, sizes
cargo run --release -- search --num-variables 20       # find a schedule
cargo run --release --example evaluation               # a short program on the library
```

`bench` commits to a table of `2^n` field elements, proves an evaluation,
verifies the proof, and reports median times and sizes. With `--out DIR` it
saves the statement and the proof, which `zo0k verify DIR` checks. Every
command takes `--field gf2_256|gf2_128` and `--hash blake3|keccak256`, and
`--threads N` limits the worker threads. Run `zo0k help` for all options.

Building needs Rust 1.88 or later and a 64-bit target. The field arithmetic
uses carry-less multiplication instructions (PCLMULQDQ on x86-64, PMULL on
AArch64) when the build enables them, and a slow portable implementation
otherwise. `.cargo/config.toml` enables them. That file applies to builds in
this repository only: a crate that depends on this one has to enable them
itself, with `RUSTFLAGS="-C target-feature=+pclmulqdq"` on x86-64 and
`RUSTFLAGS="-C target-feature=+aes"` on AArch64, or with
`RUSTFLAGS="-C target-cpu=native"`.

## Library

```rust
use zo0k::{Blake3, Field, Gf2_256, Parameters, Protocol, Rng};

fn main() -> Result<(), zo0k::Error> {
    // Tables of 2^10 entries, 100 bits of security.
    let parameters = Parameters::standard(10, 100, Gf2_256::SIZE)?;
    let protocol = Protocol::<Gf2_256, Blake3>::new(parameters)?;
    let mut rng = Rng::from_entropy();
    let table = rng.fields(1 << 10);
    let point = rng.fields(10);

    let (commitment, witness) = protocol.commit(table, &mut rng)?;
    let (value, proof) = protocol.prove(&commitment, witness, &point, &mut rng)?;
    protocol.verify(&commitment, &point, value, &proof)
}
```

A crate that uses the library alone can depend on it with
`default-features = false`, which leaves out the command-line tool and its
dependencies.

A table lists the values of a multilinear polynomial on `{0,1}^n` in
lexicographic order, and `prove` returns the value of that polynomial at the
point together with the proof. A commitment can be opened once: `prove`
consumes the witness, because the protocol is zero knowledge on fresh inputs,
that is, when the randomness of the commitment has not been used by another
proof.

### Fields and hash functions

`Protocol<F, H>` works for every field `F: RsField` and hash function
`H: Hasher`.

* A `Field` supplies arithmetic, an encoding and uniform sampling. An
  `RsField` also has Reed–Solomon codes: evaluation points and an encoder.
  Every `BinaryField`, a field that contains `GF(2^128)`, is an `RsField`
  through the additive FFT of `reed_solomon`. A field of another kind
  implements `RsField` with its own encoder.
* A `Hasher` is an incremental hash function with 32-byte digests and one
  independent function per `Domain` (Merkle leaves, Merkle nodes, transcript,
  challenges, proof of work). Merkle trees, the Fiat–Shamir transformation and
  the proof of work are built from it.

The soundness error depends on the size of the field, so parameters are
checked against the field of the protocol.

### Parameters

```json
{
  "num_variables": 20,
  "stages": [
    { "folding": 3, "domain_size": 448512, "queries": 87 },
    { "folding": 3, "domain_size": 133120, "queries": 52 },
    { "folding": 3, "domain_size": 50688, "queries": 34 },
    { "folding": 3, "domain_size": 19968, "queries": 26 }
  ],
  "security_bits": 100,
  "pow_bits": 24,
  "johnson_slack": 0.0002
}
```

Stage `i` has an oracle of `2^folding` rows, each a codeword of a Reed–Solomon
code of length `domain_size`, and is queried `queries` times. Every stage but
the last is a sumcheck followed by a code switch, and the last is a sumcheck
followed by the base case. One stage gives the paper's sublinear protocol.

`pow_bits` is the number of bits of the proof of work before every round of
queries. Three further proofs of work are optional and absent above: a stage
can set `sumcheck_pow_bits`, for a proof of work before each folding challenge
of its sumcheck, and `ood_pow_bits`, for one before the out-of-domain sample of
its code switch; `base_case_pow_bits`, next to `pow_bits`, is for one before
the combination challenge of the base case. A small field needs them; see
[`GF(2^128)`](#gf2128).

The soundness analysis divides the error of a round by `2^bits` for the proof
of work that precedes it. Parameters are accepted only if the sum of these
quotients over the field of the protocol is at most `2^-security_bits`.
`analyze --optimize-queries` (or `tuning::optimize_queries`) chooses the query
counts of a schedule, `tuning::assign_pow_bits` its optional proofs of work,
and `Parameters::standard` a whole schedule.

## Performance

Median times on an Apple M4 Max with 16 threads, for rows of the paper's
tables. Proving includes 24 bits of proof of work per stage, which takes most
of the proving time on small tables.

| Row | Table | Commit | Prove | Verify | Argument |
|---|---|---|---|---|---|
| `sublinear-20` | 32 MiB | 0.02 s | 0.1 s | 1.2 ms | 544 KiB |
| `polylog-20` | 32 MiB | 0.03 s | 0.3 s | 1.0 ms | 125 KiB |
| `polylog-24` | 512 MiB | 0.45 s | 0.55 s | 1.0 ms | 165 KiB |
| `polylog-26` | 2 GiB | 2.2 s | 1.4 s | 1.1 ms | 182 KiB |

With Keccak-256, on `polylog-20` and `polylog-24`, committing and verifying
take about a third longer and proving about twice as long.

### `GF(2^128)`

On the same schedule, the smaller field is about twice as fast and its proofs
are a quarter to a half smaller. It needs other schedules, however: the errors
that do not depend on the queries are inversely proportional to the size of
the field. The largest is the mutual correlated agreement error of the first
sumcheck round, which the analysis bounds in the list-decoding regime. Over
`GF(2^128)` the rows of the paper's tables therefore have between 41 and 56
bits of security instead of 100:

```sh
cargo run --release -- analyze --paper polylog-24 --field gf2_128
```

Two changes bring the smaller field to the target. A larger Johnson slack
lowers those errors, at the cost of more queries, and proofs of work before
the sumcheck challenges, the out-of-domain samples and the combination
challenge of the base case cover the rest. `zo0k search --field gf2_128` gives
each of those rounds the fewest bits that the target needs, at most
`--max-pow-bits` (24 by default, as before the queries), with the smallest
slack for which that limit suffices. `Parameters::standard` does the same.
The results for 100 bits, with 24 bits before the queries:

| Table | Limit | Slack | Commit | Prove | Verify | Argument |
|---|---|---|---|---|---|---|
| `2^20` entries, 16 MiB | 24 | 0.015 | 0.02 s | 0.7 s | 0.6 ms | 93 KiB |
| `2^20` entries, 16 MiB | 18 | 0.03 | 0.02 s | 0.2 s | 0.6 ms | 97 KiB |
| `2^24` entries, 256 MiB | 24 | 0.025 | 0.27 s | 0.8 s | 0.7 ms | 129 KiB |
| `2^26` entries, 1 GiB | 24 | 0.03 | 1.1 s | 2.2 s | 0.8 ms | 151 KiB |

Compared with the rows for the same tables over `GF(2^256)`, the arguments are
between a sixth and a quarter smaller, and committing is 1.6 to 2 times
faster. Proving takes longer with the limit of 24 bits, because up to four
rounds per stage then have a proof of work of that size. A smaller limit
trades argument size for that time: at `2^20` entries, a limit of 18 bits
costs 4% of the argument size and takes a third of the time.

As for the rows of the paper's tables, the bound on the soundness error of the
interactive protocol is about `2^-76` for these schedules, and the proofs of
work account for the remaining 24 bits.

## Structure

| Paper | Module |
|---|---|
| Layout and encodings; communication | `parameters`, `costs` |
| The mask polynomial | `mask` |
| Steps 1–3 of a stage: mask sum, batching, sumcheck rounds | `sumcheck` |
| Step 4: code switch | `code_switch` |
| Step 5: base case | `base_case` |
| Claims and succinct linear forms | `claim` |
| Reed–Solomon codes and their encoding | `reed_solomon`, `witness` |
| Merkle commitments without salts | `merkle`, `oracle` |
| Round-by-round soundness errors | `soundness` |
| Fiat–Shamir transformation, proof of work | `transcript` |
| Choice of query counts and proofs of work | `tuning` |
| Fields: the `Field` trait, `GF(2^128)`, `GF(2^256)` | `field` |
| Hash functions: the `Hasher` trait, BLAKE3, Keccak-256 | `hash` |

`lib.rs` composes the stages into `Protocol::commit`, `prove` and `verify`.
Each of `sumcheck`, `code_switch` and `base_case` holds the prover and the
verifier of one reduction side by side.

## What the paper leaves to the implementation

* **Fields.** `GF(2^128)` is `GF(2)[x]/(x^128 + x^7 + x^2 + x + 1)`, and
  `GF(2^256)` is its extension by `u^2 + u + x^121`.
* **Evaluation domain.** The code of length `n` evaluates at the first `n`
  elements, in binary order, of the span of a Cantor basis (`reed_solomon`).
  Encoding is the additive FFT of Lin, Chung and Han.
* **Seeds.** A row has `t + 1` seed coordinates for query count `t`, which is
  the variant of the paper that commits without salts.
* **Batching in a code switch.** The coefficients are the powers of one
  challenge, and the out-of-domain sample is one point with a uniform nonzero
  vector on the pad, as in the paper's Reed–Solomon paragraph.
* **Base case.** The symbol of the mask at a queried position is computed by
  the verifier and authenticated, and is not part of the proof.
* **Fiat–Shamir.** The transcript binds the field, the hash function, the
  parameters, the commitment and the claim, and then every prover message.
  Challenges are digests of the transcript state and a counter.
* **Proof of work.** A proof of work before a verifier message is the
  smallest 8-byte nonce whose hash with the transcript state has a given
  number of leading zero bits. There is one before every round of queries
  (`pow_bits`), and optionally before every folding challenge of a sumcheck,
  before an out-of-domain sample and before the combination challenge of the
  base case. The batching challenge of a sumcheck has none.
* **Soundness analysis.** `soundness` evaluates the round-by-round errors of
  the paper's theorems with the Johnson bound for list sizes and the mutual
  correlated agreement bound of Ben-Sasson, Carmon, Haböck, Kopparty and Saraf
  for Reed–Solomon codes. Those errors bound the soundness error of the
  interactive protocol. The security target bounds the sum of the errors
  after each is divided by `2^bits` for its proof of work, which raises the
  cost of an attempt at the challenges of that round.

## Reproducing the paper's evaluation

`paper/tables.json` and `paper/frontiers.json` hold the configurations that the
paper's parameter search selected: for every row of its two tables, and for the
frontiers of its figure.

```sh
cargo run --release -- paper                 # writes paper/scaling-tables.tex, paper/frontiers.csv
uv run --with reportlab paper/plot_frontiers.py paper/frontiers.csv paper/size-frontiers.pdf
```

`zo0k search --num-variables N [--stages 1] [--baseline]` runs a search of the
same kind: it minimizes the expected argument size over the folding parameters
and domain sizes, under a limit on the IOP proof length. It takes seconds and
its results are within about one percent of the configurations above, which
come from a longer search. Both are heuristics that end in local optima. Over
`GF(2^256)` the search uses no proof of work other than that of the queries,
as the paper's configurations do.

Sizes are *IOP proof length* (all oracles and direct messages, including the
proof-of-work nonces) and *argument size* (the compiled proof). The argument
size depends on the sampled queries, so the tables report its expectation,
which `costs` computes in closed form.

The comparison without zero knowledge is a model, `Variant::Baseline`: the
same protocol without seeds, masks, auxiliary blocks, pads, the mask sum and
batching challenge of every sumcheck, and the mask of the base case. It has
the same accounting, and no prover or verifier.
