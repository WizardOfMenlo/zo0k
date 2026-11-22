use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Domain, Hasher};
use crate::parallel;
use std::ptr;

const PARALLEL_THRESHOLD: usize = 1 << 10;
const SUBTREE_HEIGHT: usize = 9;
const SUBTREE_LEAVES: usize = 1 << SUBTREE_HEIGHT;
const LEAF_TILE_MAX: usize = 8;
const LEAF_TILE_BYTES_MAX: usize = 16384;

#[derive(Clone, Copy)]
struct RawMut(*mut Digest);

unsafe impl Send for RawMut {}
unsafe impl Sync for RawMut {}

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

fn leaf_tile_len(leaf_len_bytes: usize) -> usize {
    assert!(leaf_len_bytes >= 8);
    let tile = if LEAF_TILE_MAX * leaf_len_bytes <= LEAF_TILE_BYTES_MAX { LEAF_TILE_MAX } else { 4 };
    assert!(tile == 4 || tile == LEAF_TILE_MAX);
    tile
}

fn hash_rows_range<H: Hasher, F: Field>(
    rows: &[Vec<F>],
    start: usize,
    count: usize,
    scratch: &mut [u8],
    out: &mut [Digest],
) {
    assert!(!rows.is_empty());
    assert!(out.len() == count);
    let leaf_len_bytes = 8 + rows.len() * F::BYTES;
    let tile_cap = scratch.len() / leaf_len_bytes;
    assert!(tile_cap >= 1);
    let mut done = 0;
    while done < count {
        let batch = (count - done).min(tile_cap);
        let base = start + done;
        for lane in 0..batch {
            let off = lane * leaf_len_bytes;
            scratch[off..off + 8].copy_from_slice(&((base + lane) as u64).to_le_bytes());
        }
        for (row_index, row) in rows.iter().enumerate() {
            let col_off = 8 + row_index * F::BYTES;
            let slice = &row[base..base + batch];
            for (lane, &element) in slice.iter().enumerate() {
                let off = lane * leaf_len_bytes + col_off;
                element.write_bytes(&mut scratch[off..off + F::BYTES]);
            }
        }
        let used_bytes = batch * leaf_len_bytes;
        H::hash_equal_len_batch(
            Domain::MerkleLeaf,
            &scratch[..used_bytes],
            leaf_len_bytes,
            &mut out[done..done + batch],
        );
        done += batch;
    }
    assert!(done == count);
}

fn level_lengths(len: usize, levels_count: usize) -> Vec<usize> {
    assert!(len > 0);
    assert!(levels_count >= 1);
    let mut lens = Vec::with_capacity(levels_count);
    let mut width = len;
    for _ in 0..levels_count {
        lens.push(width);
        width = width.div_ceil(2);
    }
    assert!(lens.len() == levels_count);
    lens
}

fn write_disjoint(dst: RawMut, start: usize, limit: usize, src: &[Digest]) {
    assert!(start + src.len() <= limit);
    assert!(!src.is_empty());
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), dst.0.add(start), src.len());
    }
}

fn build_subtree_levels<H: Hasher>(
    subtree: usize,
    mut cur_len: usize,
    buf_a: &mut [Digest; SUBTREE_LEAVES],
    buf_b: &mut [Digest; SUBTREE_LEAVES / 2],
    ptrs: &[(RawMut, usize)],
    write_first_depth: usize,
) {
    assert!(cur_len > 0 && cur_len <= SUBTREE_LEAVES);
    assert!(ptrs.len() == SUBTREE_HEIGHT + 1);
    for depth in 0..SUBTREE_HEIGHT {
        let next_len = cur_len.div_ceil(2);
        let shift = SUBTREE_HEIGHT - depth - 1;
        let start = subtree << shift;
        let (dst, limit) = ptrs[depth + 1];
        if depth % 2 == 0 {
            H::hash_nodes(&buf_a[..cur_len], &mut buf_b[..next_len]);
            if depth + 1 >= write_first_depth {
                write_disjoint(dst, start, limit, &buf_b[..next_len]);
            }
        } else {
            H::hash_nodes(&buf_b[..cur_len], &mut buf_a[..next_len]);
            if depth + 1 >= write_first_depth {
                write_disjoint(dst, start, limit, &buf_a[..next_len]);
            }
        }
        cur_len = next_len;
    }
    assert!(cur_len == 1);
}

fn hash_level<H: Hasher>(below: &[Digest]) -> Vec<Digest> {
    assert!(below.len() > 1);
    let out_len = below.len().div_ceil(2);
    if below.len() < PARALLEL_THRESHOLD {
        let mut level = vec![[0u8; 32]; out_len];
        H::hash_nodes(below, &mut level);
        return level;
    }
    let mut level: Vec<Digest> = Vec::with_capacity(out_len);
    let dst = RawMut(level.as_mut_ptr());
    parallel::map_pieces(out_len, 1 << 8, |range| {
        let child_start = range.start * 2;
        let child_end = (range.end * 2).min(below.len());
        let mut local = vec![[0u8; 32]; range.len()];
        H::hash_nodes(&below[child_start..child_end], &mut local);
        write_disjoint(dst, range.start, out_len, &local);
    });
    unsafe { level.set_len(out_len) };
    assert!(level.len() == out_len);
    level
}

fn finish_levels<H: Hasher>(levels: &mut Vec<Vec<Digest>>) {
    assert!(!levels.is_empty());
    for _ in 0..64 {
        let below_len = levels.last().expect("at least one level").len();
        if below_len <= 1 {
            break;
        }
        let level = hash_level::<H>(levels.last().expect("at least one level"));
        levels.push(level);
    }
    assert!(levels.last().expect("root level exists").len() == 1);
}

pub struct MerkleTree {
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    pub fn new<H: Hasher>(leaves: Vec<Digest>) -> Self {
        assert!(!leaves.is_empty());
        if leaves.len() < SUBTREE_LEAVES * 2 {
            let mut levels = vec![leaves];
            finish_levels::<H>(&mut levels);
            return Self { levels };
        }
        let len = leaves.len();
        let lens = level_lengths(len, SUBTREE_HEIGHT + 1);
        let mut levels: Vec<Vec<Digest>> = Vec::with_capacity(SUBTREE_HEIGHT + 1);
        levels.push(leaves);
        for &level_len in &lens[1..] {
            levels.push(Vec::with_capacity(level_len));
        }
        let ptrs: Vec<(RawMut, usize)> =
            levels.iter_mut().zip(&lens).map(|(level, &n)| (RawMut(level.as_mut_ptr()), n)).collect();
        let leaves_ref: &[Digest] = &levels[0];
        let num_subtrees = len.div_ceil(SUBTREE_LEAVES);
        parallel::map_pieces(num_subtrees, 1, |range| {
            let mut buf_a = [[0u8; 32]; SUBTREE_LEAVES];
            let mut buf_b = [[0u8; 32]; SUBTREE_LEAVES / 2];
            for subtree in range {
                let start = subtree << SUBTREE_HEIGHT;
                let end = ((subtree + 1) << SUBTREE_HEIGHT).min(len);
                let cur_len = end - start;
                buf_a[..cur_len].copy_from_slice(&leaves_ref[start..end]);
                build_subtree_levels::<H>(subtree, cur_len, &mut buf_a, &mut buf_b, &ptrs, 1);
            }
        });
        for (level, &n) in levels[1..].iter_mut().zip(&lens[1..]) {
            unsafe { level.set_len(n) };
        }
        finish_levels::<H>(&mut levels);
        Self { levels }
    }

    pub(crate) fn from_rows<H: Hasher, F: Field>(rows: &[Vec<F>]) -> Self {
        assert!(!rows.is_empty());
        let len = rows[0].len();
        assert!(len > 0);
        assert!(rows.iter().all(|row| row.len() == len));
        let leaf_len_bytes = 8 + rows.len() * F::BYTES;
        let tile = leaf_tile_len(leaf_len_bytes);
        if len < SUBTREE_LEAVES * 2 {
            let mut leaves = vec![[0u8; 32]; len];
            let mut scratch = vec![0u8; tile * leaf_len_bytes];
            hash_rows_range::<H, F>(rows, 0, len, &mut scratch, &mut leaves);
            return Self::new::<H>(leaves);
        }
        let lens = level_lengths(len, SUBTREE_HEIGHT + 1);
        let mut levels: Vec<Vec<Digest>> = lens.iter().map(|&n| Vec::with_capacity(n)).collect();
        let ptrs: Vec<(RawMut, usize)> =
            levels.iter_mut().zip(&lens).map(|(level, &n)| (RawMut(level.as_mut_ptr()), n)).collect();
        let num_subtrees = len.div_ceil(SUBTREE_LEAVES);
        parallel::map_pieces(num_subtrees, 1, |range| {
            let mut scratch = vec![0u8; tile * leaf_len_bytes];
            let mut buf_a = [[0u8; 32]; SUBTREE_LEAVES];
            let mut buf_b = [[0u8; 32]; SUBTREE_LEAVES / 2];
            for subtree in range {
                let start = subtree << SUBTREE_HEIGHT;
                let end = ((subtree + 1) << SUBTREE_HEIGHT).min(len);
                let cur_len = end - start;
                hash_rows_range::<H, F>(rows, start, cur_len, &mut scratch, &mut buf_a[..cur_len]);
                write_disjoint(ptrs[0].0, start, ptrs[0].1, &buf_a[..cur_len]);
                build_subtree_levels::<H>(subtree, cur_len, &mut buf_a, &mut buf_b, &ptrs, 1);
            }
        });
        for (level, &n) in levels.iter_mut().zip(&lens) {
            unsafe { level.set_len(n) };
        }
        finish_levels::<H>(&mut levels);
        Self { levels }
    }

    pub fn root(&self) -> Digest {
        self.levels.last().expect("at least one level")[0]
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
    let cap = leaves.len();
    let mut known: Vec<usize> = leaves.iter().map(|leaf| leaf.0).collect();
    let mut digests: Vec<Digest> = leaves.iter().map(|leaf| leaf.1).collect();
    let mut next_known: Vec<usize> = Vec::with_capacity(cap);
    let mut next_digests: Vec<Digest> = Vec::with_capacity(cap);
    let mut pairs: Vec<Digest> = Vec::with_capacity(cap * 2);
    let mut width = num_leaves;
    for _ in 0..64 {
        if width <= 1 {
            break;
        }
        next_known.clear();
        pairs.clear();
        let mut index = 0;
        while index < known.len() {
            let parent = known[index] / 2;
            let left_child = 2 * parent;
            if left_child + 1 < width {
                let left = if known[index] == left_child {
                    let digest = digests[index];
                    index += 1;
                    digest
                } else {
                    next_digest()?
                };
                let right = if index < known.len() && known[index] == left_child + 1 {
                    let digest = digests[index];
                    index += 1;
                    digest
                } else {
                    next_digest()?
                };
                pairs.push(left);
                pairs.push(right);
            } else {
                assert!(known[index] == left_child);
                pairs.push(digests[index]);
                index += 1;
            }
            next_known.push(parent);
        }
        assert!(next_known.len() == pairs.len().div_ceil(2));
        next_digests.resize(next_known.len(), [0u8; 32]);
        H::hash_nodes(&pairs, &mut next_digests);
        std::mem::swap(&mut known, &mut next_known);
        std::mem::swap(&mut digests, &mut next_digests);
        width = width.div_ceil(2);
    }
    assert!(digests.len() == 1);
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
