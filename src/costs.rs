//! Communication costs of a layout.
//!
//! The *IOP proof length* counts everything the prover sends in the
//! interactive oracle proof: all oracles, including the input oracle, and all
//! direct messages, among them the proof-of-work nonces. The *argument size*
//! is the length of the compiled, non-interactive proof: Merkle roots, direct
//! messages, and the opened columns with their authentication paths. The argument size depends on the
//! sampled queries, because repeated and neighbouring queries share data, so
//! this module computes its expectation.

use crate::hash::DIGEST_BYTES;
use crate::parameters::{Layout, Parameters};
use crate::transcript::NONCE_BYTES;
use serde::Serialize;

/// Sizes in bytes of the proofs of a layout.
#[derive(Clone, Debug, Serialize)]
pub struct Costs {
    /// All oracles.
    pub oracle_bytes: usize,
    /// The field elements that the prover sends directly.
    pub message_bytes: usize,
    /// The proof-of-work nonces.
    pub pow_nonce_bytes: usize,
    /// The IOP proof length: the sum of the three sizes above.
    pub iop_bytes: usize,
    /// The Merkle roots.
    pub root_bytes: usize,
    /// Expected size of the opened columns and their authentication paths.
    pub expected_opening_bytes: f64,
    /// Expected argument size: roots, direct messages, nonces and openings.
    pub expected_argument_bytes: f64,
}

impl Costs {
    /// The sizes for a field whose elements are encoded in `field_bytes`
    /// bytes.
    pub fn new(layout: &Layout, parameters: &Parameters, field_bytes: usize) -> Self {
        let count = layout.stages.len();
        let oracle_bytes = field_bytes * layout.oracle_len();
        let message_bytes = field_bytes * layout.message_len();
        // One nonce per verifier message that a proof of work precedes.
        let proofs_of_work = layout
            .rounds()
            .into_iter()
            .filter(|&(stage, round)| parameters.pow_bits_before(stage, round) > 0);
        let pow_nonce_bytes = NONCE_BYTES * proofs_of_work.count();
        let root_bytes = DIGEST_BYTES * count;
        // An opened column has one symbol per row. The symbol of the mask of
        // the base case is not sent: the verifier computes it.
        let expected_opening_bytes: f64 = layout
            .stages
            .iter()
            .map(|stage| {
                let opening = ExpectedOpening::new(stage.domain_size, stage.queries);
                (field_bytes * stage.rows()) as f64 * opening.leaves + DIGEST_BYTES as f64 * opening.digests
            })
            .sum();
        Self {
            oracle_bytes,
            message_bytes,
            pow_nonce_bytes,
            iop_bytes: oracle_bytes + message_bytes + pow_nonce_bytes,
            root_bytes,
            expected_opening_bytes,
            expected_argument_bytes: (root_bytes + message_bytes + pow_nonce_bytes) as f64
                + expected_opening_bytes,
        }
    }
}

/// Expected shape of the opening of a Merkle tree at independent uniform
/// positions.
#[derive(Clone, Copy, Debug)]
pub struct ExpectedOpening {
    /// Expected number of distinct opened leaves.
    pub leaves: f64,
    /// Expected number of digests in the opening.
    pub digests: f64,
}

impl ExpectedOpening {
    /// For a tree with `num_leaves` leaves, opened at
    /// `queries` positions that are uniform, independent and possibly equal.
    pub fn new(num_leaves: usize, queries: usize) -> Self {
        let n = num_leaves as f64;
        let q = queries as f64;
        // Probability that no query falls in a block of `size` leaves.
        let missed = |size: usize| (q * (-(size as f64) / n).ln_1p()).exp();
        let leaves = -(q * (-1.0 / n).ln_1p()).exp_m1() * n;
        // A node is in the opening when no query falls below it and some
        // query falls below its parent. A level consists of nodes over
        // `below` leaves each, except that the last node may have fewer, and
        // the last parent fewer children.
        let mut digests = 0.0;
        let mut below = 1;
        while below < num_leaves {
            let below_parent = 2 * below;
            let complete_parents = num_leaves / below_parent;
            if complete_parents != 0 {
                digests += (2 * complete_parents) as f64 * (missed(below) - missed(below_parent));
            }
            let rest = num_leaves - complete_parents * below_parent;
            if rest != 0 {
                digests += (rest / below) as f64 * (missed(below) - missed(rest));
                if !rest.is_multiple_of(below) {
                    digests += missed(rest % below) - missed(rest);
                }
            }
            below = below_parent;
        }
        Self { leaves, digests }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Blake3, Digest};
    use crate::merkle::MerkleTree;
    use crate::rng::Rng;

    #[test]
    fn expected_opening_is_exact_for_small_trees() {
        // Enumerate all query vectors and average the true opening size.
        for num_leaves in [1usize, 2, 3, 5, 8, 9, 11] {
            let tree = MerkleTree::new::<Blake3>(vec![Digest::default(); num_leaves]);
            for queries in 1..=3u32 {
                let (mut leaves, mut digests) = (0, 0);
                let total = num_leaves.pow(queries);
                for code in 0..total {
                    let mut positions: Vec<usize> =
                        (0..queries).map(|i| code / num_leaves.pow(i) % num_leaves).collect();
                    positions.sort_unstable();
                    positions.dedup();
                    leaves += positions.len();
                    digests += tree.open(&positions).len();
                }
                let expected = ExpectedOpening::new(num_leaves, queries as usize);
                let close = |a: f64, b: usize| (a - b as f64 / total as f64).abs() < 1e-9;
                assert!(close(expected.leaves, leaves), "{num_leaves} {queries}");
                assert!(close(expected.digests, digests), "{num_leaves} {queries}");
            }
        }
    }

    #[test]
    fn expected_opening_matches_sampled_openings_of_a_ragged_tree() {
        let (num_leaves, queries) = (9792, 69);
        let tree = MerkleTree::new::<Blake3>(vec![Digest::default(); num_leaves]);
        let mut rng = Rng::from_seed([30; 32]);
        let trials = 300;
        let mut digests = 0;
        for _ in 0..trials {
            let mut positions: Vec<usize> = (0..queries).map(|_| rng.below(num_leaves)).collect();
            positions.sort_unstable();
            positions.dedup();
            digests += tree.open(&positions).len();
        }
        let expected = ExpectedOpening::new(num_leaves, queries).digests;
        assert!((digests as f64 / trials as f64 / expected - 1.0).abs() < 0.01);
    }
}
