//! Merkle trees with batched openings.
//!
//! A tree commits to `n` leaf digests. Every node has two children, except
//! that the last node of a level has one when the level below has an odd
//! number of nodes. An opening of a set of leaves consists of the
//! digests that are needed to recompute the root and cannot be computed from
//! the opened leaves themselves: level by level from the leaves upwards, and
//! within a level by increasing position, every sibling of a computable node
//! that is not itself computable.
//!
//! Leaves are not salted. A leaf digest binds the position of the leaf, and
//! hides its content if the content is unpredictable (see the paper's
//! appendix "Hiding Merkle commitments without salts").

use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Domain, Hasher};
use rayon::prelude::*;

/// Below this many digests, a level is hashed on one thread.
const PARALLEL_THRESHOLD: usize = 1 << 10;

/// The digest of the leaf at `position` whose content is a column of field
/// elements.
pub fn leaf_digest<H: Hasher, F: Field>(position: usize, column: impl IntoIterator<Item = F>) -> Digest {
    let mut hasher = H::new(Domain::MerkleLeaf);
    hasher.update(&(position as u64).to_le_bytes());
    // Hash about one kibibyte at a time.
    let mut buffer = [0u8; 1024];
    let capacity = buffer.len() / F::BYTES * F::BYTES;
    let mut filled = 0;
    for element in column {
        element.write_bytes(&mut buffer[filled..filled + F::BYTES]);
        filled += F::BYTES;
        if filled == capacity {
            hasher.update(&buffer[..capacity]);
            filled = 0;
        }
    }
    hasher.update(&buffer[..filled]);
    hasher.finalize()
}

/// The digest of a node with one or two children.
fn node_digest<H: Hasher>(children: &[Digest]) -> Digest {
    H::hash(Domain::MerkleNode, children.as_flattened())
}

/// A Merkle tree, kept by the prover to answer openings.
pub struct MerkleTree {
    /// `levels[0]` are the leaf digests and the last level is the root.
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    /// The tree over the given leaf digests.
    ///
    /// # Panics
    /// Panics if there are no leaves.
    pub fn new<H: Hasher>(leaves: Vec<Digest>) -> Self {
        assert!(!leaves.is_empty());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let below = levels.last().unwrap();
            let level = if below.len() < PARALLEL_THRESHOLD {
                below.chunks(2).map(node_digest::<H>).collect()
            } else {
                below.par_chunks(2).with_min_len(1 << 8).map(node_digest::<H>).collect()
            };
            levels.push(level);
        }
        Self { levels }
    }

    /// The digest at the root.
    pub fn root(&self) -> Digest {
        self.levels.last().unwrap()[0]
    }

    /// The opening of the leaves at `positions`, which must be increasing.
    pub fn open(&self, positions: &[usize]) -> Vec<Digest> {
        let mut opening = Vec::new();
        let mut known = positions.to_vec();
        for level in &self.levels[..self.levels.len() - 1] {
            known = parents(&known, level.len(), |child, is_known| {
                if !is_known {
                    opening.push(level[child]);
                }
            });
        }
        opening
    }
}

/// The root of the tree with `num_leaves` leaves that is consistent with the
/// given leaves and with an opening, whose digests are requested one at a
/// time from `next_digest`.
///
/// The leaves are pairs of a position and a digest, by increasing position.
pub fn root_from_opening<H: Hasher>(
    num_leaves: usize,
    leaves: &[(usize, Digest)],
    mut next_digest: impl FnMut() -> Result<Digest>,
) -> Result<Digest> {
    let increasing = leaves.windows(2).all(|pair| pair[0].0 < pair[1].0);
    if leaves.is_empty() || !increasing || leaves.last().unwrap().0 >= num_leaves {
        return Err(Error::InvalidProof("invalid opened positions"));
    }
    let mut known: Vec<usize> = leaves.iter().map(|leaf| leaf.0).collect();
    let mut digests: Vec<Digest> = leaves.iter().map(|leaf| leaf.1).collect();
    let mut width = num_leaves;
    while width > 1 {
        let mut children = Vec::with_capacity(2);
        let mut above = Vec::with_capacity(known.len());
        let mut available = digests.iter();
        let mut failure = None;
        known = parents(&known, width, |child, is_known| {
            let digest = if is_known {
                Ok(*available.next().expect("one digest per known node"))
            } else {
                next_digest()
            };
            match digest {
                Ok(digest) => children.push(digest),
                Err(error) => failure = Some(error),
            }
            // The last child of a node is odd, or the last node of the level.
            if child % 2 == 1 || child + 1 == width {
                above.push(node_digest::<H>(&children));
                children.clear();
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        digests = above;
        width = width.div_ceil(2);
    }
    Ok(digests[0])
}

/// The parents of the `known` nodes of a level of `width` nodes, in
/// increasing order. Every child of each of these parents is reported to
/// `visit`, in increasing order, together with whether it is known.
fn parents(known: &[usize], width: usize, mut visit: impl FnMut(usize, bool)) -> Vec<usize> {
    let mut parents: Vec<usize> = known.iter().map(|node| node / 2).collect();
    parents.dedup();
    let mut known = known.iter().peekable();
    for &parent in &parents {
        for child in 2 * parent..(2 * parent + 2).min(width) {
            visit(child, known.next_if_eq(&&child).is_some());
        }
    }
    parents
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{BinaryField, Gf2_128, Gf2_256};
    use crate::hash::{Blake3, Keccak256};

    fn leaves<H: Hasher>(count: usize) -> Vec<Digest> {
        (0..count).map(|i| leaf_digest::<H, _>(i, [Gf2_256::from_base(i as u128)])).collect()
    }

    fn verify<H: Hasher>(
        tree: &MerkleTree,
        count: usize,
        positions: &[usize],
        opening: &[Digest],
    ) -> Result<bool> {
        let all = leaves::<H>(count);
        let opened: Vec<_> = positions.iter().map(|&p| (p, all[p])).collect();
        let mut opening = opening.iter();
        let root = root_from_opening::<H>(count, &opened, || {
            opening.next().copied().ok_or(Error::InvalidProof("truncated"))
        })?;
        Ok(root == tree.root() && opening.next().is_none())
    }

    fn openings_verify<H: Hasher>() {
        for count in 1..=20 {
            let tree = MerkleTree::new::<H>(leaves::<H>(count));
            // Every subset of the first few sizes, pairs of leaves otherwise.
            let subsets: Vec<Vec<usize>> = if count <= 10 {
                (1..1usize << count).map(|mask| (0..count).filter(|i| mask >> i & 1 == 1).collect()).collect()
            } else {
                (0..count - 1).map(|i| vec![i, count - 1]).collect()
            };
            for positions in subsets {
                let opening = tree.open(&positions);
                assert!(verify::<H>(&tree, count, &positions, &opening).unwrap(), "{count} {positions:?}");
            }
        }
    }

    #[test]
    fn openings_verify_for_every_shape_and_subset() {
        openings_verify::<Blake3>();
        openings_verify::<Keccak256>();
    }

    #[test]
    fn an_opening_has_one_digest_per_missing_sibling() {
        let tree = MerkleTree::new::<Blake3>(leaves::<Blake3>(8));
        assert_eq!(tree.open(&[0]).len(), 3);
        assert_eq!(tree.open(&[0, 1]).len(), 2);
        assert_eq!(tree.open(&[0, 7]).len(), 4);
        assert_eq!(tree.open(&(0..8).collect::<Vec<_>>()).len(), 0);
        // A level of five nodes: the last node of the level above has one child.
        let ragged = MerkleTree::new::<Blake3>(leaves::<Blake3>(5));
        assert_eq!(ragged.open(&[4]).len(), 1);
        assert_eq!(ragged.open(&[0]).len(), 3);
    }

    fn altered_openings<H: Hasher>() {
        let count = 13;
        let tree = MerkleTree::new::<H>(leaves::<H>(count));
        let positions = [2, 3, 9];
        let opening = tree.open(&positions);
        for i in 0..opening.len() {
            let mut bad = opening.clone();
            bad[i][0] ^= 1;
            assert!(!verify::<H>(&tree, count, &positions, &bad).unwrap());
        }
        assert!(verify::<H>(&tree, count, &positions, &opening[1..]).is_err());
        assert!(!verify::<H>(&tree, count, &[2, 3, 10], &opening).unwrap_or(false));
        let all = leaves::<H>(count);
        for bad in [vec![(3, all[3]), (2, all[2])], vec![(13, all[0])], vec![]] {
            assert!(root_from_opening::<H>(count, &bad, || Ok([0; 32])).is_err());
        }
    }

    #[test]
    fn altered_openings_are_rejected() {
        altered_openings::<Blake3>();
        altered_openings::<Keccak256>();
    }

    fn leaf_digests<H: Hasher, F: BinaryField>() {
        let column = [F::ONE, F::ZERO];
        assert_ne!(leaf_digest::<H, F>(0, column), leaf_digest::<H, F>(1, column));
        assert_ne!(leaf_digest::<H, F>(0, column), leaf_digest::<H, F>(0, [F::ONE]));
        // Columns longer than the hashing buffer.
        let long: Vec<F> = (0..70).map(F::from_base).collect();
        let mut other = long.clone();
        other[69] = F::ZERO;
        assert_ne!(leaf_digest::<H, F>(0, long), leaf_digest::<H, F>(0, other));
    }

    #[test]
    fn leaf_digests_bind_position_and_content() {
        leaf_digests::<Blake3, Gf2_128>();
        leaf_digests::<Blake3, Gf2_256>();
        leaf_digests::<Keccak256, Gf2_256>();
        // A leaf and a node with the same content have different digests.
        let children = [[1u8; 32], [2u8; 32]];
        let bytes = children.concat();
        assert_ne!(node_digest::<Blake3>(&children), Blake3::hash(Domain::MerkleLeaf, &bytes));
    }
}
