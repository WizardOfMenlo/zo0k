use crate::hash::DIGEST_BYTES;
use crate::parameters::{Layout, Parameters};
use crate::transcript::NONCE_BYTES;

#[derive(Clone, Debug)]
pub struct Costs {
    pub oracle_bytes: usize,
    pub message_bytes: usize,
    pub pow_nonce_bytes: usize,
    pub iop_bytes: usize,
    pub root_bytes: usize,
    pub expected_opening_bytes: f64,
    pub expected_argument_bytes: f64,
}

impl Costs {
    pub fn new(layout: &Layout, parameters: &Parameters, field_bytes: usize) -> Self {
        let count = layout.stages.len();
        let oracle_bytes = field_bytes * layout.oracle_len();
        let message_bytes = field_bytes * layout.message_len();
        let proofs_of_work = layout
            .rounds()
            .into_iter()
            .filter(|&(stage, round)| parameters.pow_bits_before(stage, round) > 0);
        let pow_nonce_bytes = NONCE_BYTES * proofs_of_work.count();
        let root_bytes = DIGEST_BYTES * count;
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

#[derive(Clone, Copy, Debug)]
pub struct ExpectedOpening {
    pub leaves: f64,
    pub digests: f64,
}

impl ExpectedOpening {
    pub fn new(num_leaves: usize, queries: usize) -> Self {
        let n = num_leaves as f64;
        let q = queries as f64;
        let missed = |size: usize| (q * (-(size as f64) / n).ln_1p()).exp();
        let leaves = -(q * (-1.0 / n).ln_1p()).exp_m1() * n;
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
