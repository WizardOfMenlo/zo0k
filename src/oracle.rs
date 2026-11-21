//! Oracles, committed by Merkle trees over their columns.
//!
//! An oracle is a list of codewords of the same length, its rows. Position
//! `x` of the oracle is the column of the symbols of all rows at `x`, and the
//! leaf of the Merkle tree at `x` is the digest of that column.
//!
//! An opening at a list of positions consists of the columns at the distinct
//! positions, in increasing order, followed by the Merkle opening. The prover
//! may send only the first symbols of every column when the verifier can
//! compute the others.

use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Hasher};
use crate::merkle::{self, MerkleTree};
use crate::parallel;
use crate::transcript::{ProverTranscript, VerifierTranscript};

/// The verifier hashes opened columns in parallel when they have at least
/// this many symbols in total.
const PARALLEL_THRESHOLD: usize = 1 << 12;

/// An oracle as the prover holds it.
pub struct Oracle<F> {
    rows: Vec<Vec<F>>,
    tree: MerkleTree,
}

impl<F: Field> Oracle<F> {
    /// Commit to an oracle with the given rows.
    ///
    /// # Panics
    /// Panics if there is no row, or if the rows differ in length.
    pub fn commit<H: Hasher>(rows: Vec<Vec<F>>) -> Self {
        let len = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == len));
        let leaves = parallel::map_range(len, 1 << 6, |position| {
            merkle::leaf_digest::<H, F>(position, rows.iter().map(|row| row[position]))
        });
        Self { tree: MerkleTree::new::<H>(leaves), rows }
    }

    /// The number of rows and their length.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows.len(), self.rows[0].len())
    }

    /// The commitment: the root of the Merkle tree.
    pub fn root(&self) -> Digest {
        self.tree.root()
    }

    /// Send an opening at the given positions, with the first `width` symbols
    /// of every column, and return these partial columns.
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

/// The columns of an oracle at a set of positions.
pub struct Opening<F> {
    /// The distinct opened positions, in increasing order.
    positions: Vec<usize>,
    columns: Vec<Vec<F>>,
}

impl<F: Field> Opening<F> {
    /// Read an opening of the oracle with the given commitment and length at
    /// the given positions, and check it against the commitment.
    ///
    /// The proof holds the first `width` symbols of every column, and
    /// `complete` computes the remaining ones from the position and these
    /// symbols. The returned columns are the partial ones.
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
        // A few short columns are hashed faster on one thread.
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

    /// The opened column at each of the given positions.
    ///
    /// # Panics
    /// Panics if a position was not opened.
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
