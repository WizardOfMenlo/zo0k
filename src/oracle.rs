use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Domain, Hasher};
use crate::merkle::{self, MerkleTree};
use crate::parallel;
use crate::transcript::{ProverTranscript, VerifierTranscript};

const PARALLEL_THRESHOLD: usize = 1 << 12;
const LEAF_BATCH: usize = 24;
const PARALLEL_LEAF_MIN: usize = 12;

pub struct Oracle<F> {
    rows: Vec<Vec<F>>,
    tree: MerkleTree,
}

impl<F: Field> Oracle<F> {
    pub fn commit<H: Hasher>(rows: Vec<Vec<F>>) -> Self {
        assert!(!rows.is_empty());
        let len = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == len));
        let tree = MerkleTree::from_rows::<H, F>(&rows);
        Self { tree, rows }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows.len(), self.rows[0].len())
    }

    pub fn root(&self) -> Digest {
        self.tree.root()
    }

    pub fn open<H: Hasher>(
        &self,
        transcript: &mut ProverTranscript<H>,
        positions: &[usize],
        width: usize,
    ) -> Opening<F> {
        let positions = distinct(positions);
        let columns: Vec<Vec<F>> = positions
            .iter()
            .map(|&position| self.rows[..width].iter().map(|row| row[position]).collect())
            .collect();
        for column in &columns {
            transcript.send_fields(column);
        }
        for digest in self.tree.open_with_rows::<H, F>(&self.rows, &positions) {
            transcript.send_digest(&digest);
        }
        Opening { positions, columns }
    }
}

pub struct Opening<F> {
    positions: Vec<usize>,
    columns: Vec<Vec<F>>,
}

fn hash_leaf_slice<H: Hasher, F: Field>(
    positions: &[usize],
    columns: &[Vec<F>],
    rests: &[Vec<F>],
    width: usize,
) -> Vec<(usize, Digest)> {
    assert!(positions.len() == columns.len());
    assert!(positions.len() == rests.len());
    let Some(first_rest) = rests.first() else {
        return Vec::new();
    };
    let rest_len = first_rest.len();
    let uniform = rests.iter().all(|rest| rest.len() == rest_len);
    if !uniform {
        return positions
            .iter()
            .zip(columns)
            .zip(rests)
            .map(|((&pos, col), rest)| {
                (pos, merkle::leaf_digest::<H, F>(pos, col.iter().copied().chain(rest.iter().copied())))
            })
            .collect();
    }
    let leaf_len_bytes = 8 + (width + rest_len) * F::BYTES;
    let mut scratch = vec![0u8; LEAF_BATCH * leaf_len_bytes];
    let mut digests = [[0u8; 32]; LEAF_BATCH];
    let mut leaves = Vec::with_capacity(positions.len());
    let mut done = 0;
    while done < positions.len() {
        let batch = (positions.len() - done).min(LEAF_BATCH);
        for lane in 0..batch {
            let index = done + lane;
            let base = lane * leaf_len_bytes;
            scratch[base..base + 8].copy_from_slice(&(positions[index] as u64).to_le_bytes());
            let mut offset = base + 8;
            for &element in &columns[index] {
                element.write_bytes(&mut scratch[offset..offset + F::BYTES]);
                offset += F::BYTES;
            }
            for &element in &rests[index] {
                element.write_bytes(&mut scratch[offset..offset + F::BYTES]);
                offset += F::BYTES;
            }
            assert!(offset == base + leaf_len_bytes);
        }
        let used_bytes = batch * leaf_len_bytes;
        H::hash_equal_len_batch(
            Domain::MerkleLeaf,
            &scratch[..used_bytes],
            leaf_len_bytes,
            &mut digests[..batch],
        );
        for lane in 0..batch {
            leaves.push((positions[done + lane], digests[lane]));
        }
        done += batch;
    }
    assert!(leaves.len() == positions.len());
    leaves
}

impl<F: Field> Opening<F> {
    pub fn read<H: Hasher>(
        transcript: &mut VerifierTranscript<H>,
        root: &Digest,
        len: usize,
        positions: &[usize],
        width: usize,
        complete: impl Fn(usize, &[F]) -> Vec<F> + Sync,
    ) -> Result<Self> {
        let positions = distinct(positions);
        let columns = positions.iter().map(|_| transcript.read_fields(width)).collect::<Result<Vec<_>>>()?;
        let rests: Vec<Vec<F>> = if positions.len() * width < PARALLEL_THRESHOLD {
            positions.iter().zip(&columns).map(|(&position, column)| complete(position, column)).collect()
        } else {
            parallel::map_range(positions.len(), 1, |index| complete(positions[index], &columns[index]))
        };
        let leaves: Vec<(usize, Digest)> = if positions.len() * width < PARALLEL_THRESHOLD {
            hash_leaf_slice::<H, F>(&positions, &columns, &rests, width)
        } else {
            let pieces = parallel::map_pieces(positions.len(), PARALLEL_LEAF_MIN, |range| {
                hash_leaf_slice::<H, F>(
                    &positions[range.clone()],
                    &columns[range.clone()],
                    &rests[range],
                    width,
                )
            });
            let mut flat = Vec::with_capacity(positions.len());
            for piece in pieces {
                flat.extend(piece);
            }
            flat
        };
        let computed = merkle::root_from_opening::<H>(len, &leaves, || transcript.read_digest())?;
        if computed != *root {
            return Err(Error::InvalidProof("an opening does not match its commitment"));
        }
        Ok(Self { positions, columns })
    }

    pub fn columns<'a>(&'a self, positions: &'a [usize]) -> impl Iterator<Item = &'a [F]> {
        positions.iter().map(|position| {
            let index = self.positions.binary_search(position).expect("the position was opened");
            &self.columns[index][..]
        })
    }
}

fn distinct(positions: &[usize]) -> Vec<usize> {
    let mut positions = positions.to_vec();
    positions.sort_unstable();
    positions.dedup();
    positions
}
