use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Hasher};
use crate::merkle::{self, MerkleTree};
use crate::parallel;
use crate::transcript::{ProverTranscript, VerifierTranscript};

const PARALLEL_THRESHOLD: usize = 1 << 12;

pub struct Oracle<F> {
    rows: Vec<Vec<F>>,
    tree: MerkleTree,
}

impl<F: Field> Oracle<F> {
    pub fn commit<H: Hasher>(rows: Vec<Vec<F>>) -> Self {
        let len = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == len));
        let leaves = parallel::map_range(len, 1 << 6, |position| {
            merkle::leaf_digest::<H, F>(position, rows.iter().map(|row| row[position]))
        });
        Self { tree: MerkleTree::new::<H>(leaves), rows }
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
        for digest in self.tree.open(&positions) {
            transcript.send_digest(&digest);
        }
        Opening { positions, columns }
    }
}

pub struct Opening<F> {
    positions: Vec<usize>,
    columns: Vec<Vec<F>>,
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
        let leaf = |(&position, column): (&usize, &Vec<F>)| {
            let rest = complete(position, column);
            (position, merkle::leaf_digest::<H, F>(position, column.iter().copied().chain(rest)))
        };
        let leaves: Vec<(usize, Digest)> = if positions.len() * width < PARALLEL_THRESHOLD {
            positions.iter().zip(&columns).map(leaf).collect()
        } else {
            parallel::map_range(positions.len(), 1, |index| leaf((&positions[index], &columns[index])))
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
