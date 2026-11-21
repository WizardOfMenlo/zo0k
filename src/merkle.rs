use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Domain, Hasher};
use crate::parallel;

const PARALLEL_THRESHOLD: usize = 1 << 10;

pub fn leaf_digest<H: Hasher, F: Field>(position: usize, column: impl IntoIterator<Item = F>) -> Digest {
    let mut hasher = H::new(Domain::MerkleLeaf);
    hasher.update(&(position as u64).to_le_bytes());
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

fn node_digest<H: Hasher>(children: &[Digest]) -> Digest {
    H::hash(Domain::MerkleNode, children.as_flattened())
}

pub struct MerkleTree {
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    pub fn new<H: Hasher>(leaves: Vec<Digest>) -> Self {
        assert!(!leaves.is_empty());
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            let below = levels.last().unwrap();
            let level = if below.len() < PARALLEL_THRESHOLD {
                below.chunks(2).map(node_digest::<H>).collect()
            } else {
                parallel::map_chunks(below, 2, 1 << 9, node_digest::<H>)
            };
            levels.push(level);
        }
        Self { levels }
    }

    pub fn root(&self) -> Digest {
        self.levels.last().unwrap()[0]
    }

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
