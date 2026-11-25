use crate::error::{Error, Result};
use crate::field::Field;
use crate::hash::{Digest, Domain, Hasher};
use crate::parallel;
use std::ptr;

const PARALLEL_THRESHOLD: usize = 1 << 10;
const SUBTREE_HEIGHT: usize = 9;
const SUBTREE_LEAVES: usize = 1 << SUBTREE_HEIGHT;
const SUBTREE_PIECE_MIN: usize = 4;
const PRUNE_LEAVES_SMALL: usize = 1 << 15;
const PRUNE_LEAVES_LARGE: usize = 1 << 18;
const PRUNE_SKIP_NARROW: usize = 5;
const PRUNE_SKIP_WIDE: usize = 3;
const PRUNE_LEAF_NARROW_MAX: usize = 1024;
const OPEN_PARALLEL_WORK_MIN: usize = 8192;
const LEAF_TILE_MAX: usize = 24;
const LEAF_TILE_BYTES_MAX: usize = 16384;

#[derive(Clone, Copy)]
struct RawMut(*mut Digest);

unsafe impl Send for RawMut {}
unsafe impl Sync for RawMut {}

impl RawMut {
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut(&self, start: usize, len: usize, limit: usize) -> &mut [Digest] {
        assert!(start + len <= limit);
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

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
    let tile = if LEAF_TILE_MAX * leaf_len_bytes <= LEAF_TILE_BYTES_MAX {
        LEAF_TILE_MAX
    } else if 8 * leaf_len_bytes <= LEAF_TILE_BYTES_MAX {
        8
    } else {
        4
    };
    assert!(tile == 4 || tile == 8 || tile == LEAF_TILE_MAX);
    tile
}

pub(crate) fn hash_rows_scratch<H: Hasher, F: Field>(
    rows: &[Vec<F>],
    start: usize,
    count: usize,
    scratch: &mut Vec<u8>,
    out: &mut [Digest],
) {
    assert!(!rows.is_empty());
    assert!(out.len() == count);
    let leaf_len_bytes = 8 + rows.len() * F::BYTES;
    let tile_cap = leaf_tile_len(leaf_len_bytes);
    let need_bytes = tile_cap * leaf_len_bytes;
    if scratch.len() < need_bytes {
        scratch.resize(need_bytes, 0);
    }
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

fn choose_skip_levels(len: usize, leaf_len_bytes: usize) -> usize {
    assert!(len > 0 && leaf_len_bytes >= 8);
    let skip = if len < PRUNE_LEAVES_SMALL {
        0
    } else if len >= PRUNE_LEAVES_LARGE && leaf_len_bytes <= PRUNE_LEAF_NARROW_MAX {
        PRUNE_SKIP_NARROW
    } else {
        PRUNE_SKIP_WIDE
    };
    assert!(skip <= SUBTREE_HEIGHT);
    skip
}

fn write_disjoint(dst: RawMut, start: usize, limit: usize, src: &[Digest]) {
    assert!(start + src.len() <= limit);
    assert!(!src.is_empty());
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), dst.0.add(start), src.len());
    }
}

fn build_subtree_from_depth<H: Hasher>(
    subtree: usize,
    mut cur_len: usize,
    first_depth: usize,
    buf_a: &mut [Digest; SUBTREE_LEAVES],
    buf_b: &mut [Digest; SUBTREE_LEAVES / 2],
    ptrs: &[(RawMut, usize)],
    write_first_depth: usize,
) {
    assert!(cur_len > 0 && cur_len <= SUBTREE_LEAVES);
    assert!(first_depth <= SUBTREE_HEIGHT && ptrs.len() == SUBTREE_HEIGHT + 1);
    for depth in first_depth..SUBTREE_HEIGHT {
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
        let out_slice = unsafe { dst.slice_mut(range.start, range.len(), out_len) };
        H::hash_nodes(&below[child_start..child_end], out_slice);
    });
    unsafe { level.set_len(out_len) };
    assert!(level.len() == out_len);
    level
}

fn append_subtree_pass<H: Hasher>(levels: &mut Vec<Vec<Digest>>) {
    let base_idx = levels.len() - 1;
    let len = levels[base_idx].len();
    assert!(len >= SUBTREE_LEAVES * 2);
    let lens = level_lengths(len, SUBTREE_HEIGHT + 1);
    for &level_len in &lens[1..] {
        levels.push(Vec::with_capacity(level_len));
    }
    let mut ptrs: Vec<(RawMut, usize)> = Vec::with_capacity(SUBTREE_HEIGHT + 1);
    ptrs.push((RawMut(ptr::null_mut()), 0));
    for (level, &n) in levels[base_idx + 1..].iter_mut().zip(&lens[1..]) {
        ptrs.push((RawMut(level.as_mut_ptr()), n));
    }
    let below_ref: &[Digest] = &levels[base_idx];
    let num_subtrees = len.div_ceil(SUBTREE_LEAVES);
    parallel::map_pieces(num_subtrees, SUBTREE_PIECE_MIN, |range| {
        let mut buf_a = [[0u8; 32]; SUBTREE_LEAVES];
        let mut buf_b = [[0u8; 32]; SUBTREE_LEAVES / 2];
        for subtree in range {
            let start = subtree << SUBTREE_HEIGHT;
            let end = ((subtree + 1) << SUBTREE_HEIGHT).min(len);
            let cur_len = end - start;
            let next_len = cur_len.div_ceil(2);
            H::hash_nodes(&below_ref[start..end], &mut buf_b[..next_len]);
            let dst_start = subtree << (SUBTREE_HEIGHT - 1);
            write_disjoint(ptrs[1].0, dst_start, ptrs[1].1, &buf_b[..next_len]);
            build_subtree_from_depth::<H>(subtree, next_len, 1, &mut buf_a, &mut buf_b, &ptrs, 1);
        }
    });
    for (level, &n) in levels[base_idx + 1..].iter_mut().zip(&lens[1..]) {
        unsafe { level.set_len(n) };
    }
    assert!(levels.last().expect("new top level").len() == lens[SUBTREE_HEIGHT]);
}

fn finish_levels<H: Hasher>(levels: &mut Vec<Vec<Digest>>) {
    assert!(!levels.is_empty());
    for _ in 0..8 {
        if levels.last().expect("at least one level").len() < SUBTREE_LEAVES * 2 {
            break;
        }
        append_subtree_pass::<H>(levels);
    }
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

struct LocalSubtree {
    nodes: Vec<Digest>,
    offsets: [usize; SUBTREE_HEIGHT],
}

fn build_local_tree<H: Hasher, F: Field>(
    rows: &[Vec<F>],
    block: usize,
    skip_levels: usize,
    num_leaves: usize,
    scratch: &mut Vec<u8>,
) -> LocalSubtree {
    assert!((1..=SUBTREE_HEIGHT).contains(&skip_levels));
    let start = block << skip_levels;
    let end = ((block + 1) << skip_levels).min(num_leaves);
    assert!(start < end);
    let mut cur_len = end - start;
    let mut offsets = [0usize; SUBTREE_HEIGHT];
    let mut total = cur_len;
    let mut w = cur_len;
    for offset in offsets.iter_mut().take(skip_levels).skip(1) {
        *offset = total;
        w = w.div_ceil(2);
        total += w;
    }
    let mut nodes = vec![[0u8; 32]; total];
    H::hash_rows_range(rows, start, cur_len, scratch, &mut nodes[..cur_len]);
    for depth in 1..skip_levels {
        let prev_off = offsets[depth - 1];
        let next_off = offsets[depth];
        let next_len = cur_len.div_ceil(2);
        let (head, tail) = nodes.split_at_mut(next_off);
        H::hash_nodes(&head[prev_off..prev_off + cur_len], &mut tail[..next_len]);
        cur_len = next_len;
    }
    assert!(nodes.len() == total);
    LocalSubtree { nodes, offsets }
}

pub struct MerkleTree {
    levels: Vec<Vec<Digest>>,
    num_leaves: usize,
    skip_levels: usize,
}

impl MerkleTree {
    pub fn new<H: Hasher>(leaves: Vec<Digest>) -> Self {
        assert!(!leaves.is_empty());
        let num_leaves = leaves.len();
        let mut levels = vec![leaves];
        finish_levels::<H>(&mut levels);
        assert!(levels[0].len() == num_leaves);
        Self { levels, num_leaves, skip_levels: 0 }
    }

    pub(crate) fn from_rows<H: Hasher, F: Field>(rows: &[Vec<F>]) -> Self {
        assert!(!rows.is_empty());
        let len = rows[0].len();
        assert!(len > 0);
        assert!(rows.iter().all(|row| row.len() == len));
        if len < SUBTREE_LEAVES * 2 {
            let mut leaves = vec![[0u8; 32]; len];
            let mut scratch = Vec::new();
            H::hash_rows_range(rows, 0, len, &mut scratch, &mut leaves);
            return Self::new::<H>(leaves);
        }
        let skip_levels = choose_skip_levels(len, 8 + rows.len() * F::BYTES);
        let lens = level_lengths(len, SUBTREE_HEIGHT + 1);
        let mut levels: Vec<Vec<Digest>> =
            lens[skip_levels..].iter().map(|&n| Vec::with_capacity(n)).collect();
        let mut ptrs: Vec<(RawMut, usize)> = vec![(RawMut(ptr::null_mut()), 0); skip_levels];
        for (level, &n) in levels.iter_mut().zip(&lens[skip_levels..]) {
            ptrs.push((RawMut(level.as_mut_ptr()), n));
        }
        assert!(ptrs.len() == SUBTREE_HEIGHT + 1);
        let num_subtrees = len.div_ceil(SUBTREE_LEAVES);
        parallel::map_pieces(num_subtrees, SUBTREE_PIECE_MIN, |range| {
            let mut scratch = Vec::new();
            let mut buf_a = [[0u8; 32]; SUBTREE_LEAVES];
            let mut buf_b = [[0u8; 32]; SUBTREE_LEAVES / 2];
            for subtree in range {
                let start = subtree << SUBTREE_HEIGHT;
                let end = ((subtree + 1) << SUBTREE_HEIGHT).min(len);
                let cur_len = end - start;
                H::hash_rows_range(rows, start, cur_len, &mut scratch, &mut buf_a[..cur_len]);
                if skip_levels == 0 {
                    write_disjoint(ptrs[0].0, start, ptrs[0].1, &buf_a[..cur_len]);
                }
                build_subtree_from_depth::<H>(
                    subtree,
                    cur_len,
                    0,
                    &mut buf_a,
                    &mut buf_b,
                    &ptrs,
                    skip_levels,
                );
            }
        });
        for (level, &n) in levels.iter_mut().zip(&lens[skip_levels..]) {
            unsafe { level.set_len(n) };
        }
        finish_levels::<H>(&mut levels);
        Self { levels, num_leaves: len, skip_levels }
    }

    pub fn root(&self) -> Digest {
        self.levels.last().expect("at least one level")[0]
    }

    pub fn open(&self, positions: &[usize]) -> Vec<Digest> {
        assert!(self.skip_levels == 0);
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

    pub(crate) fn open_with_rows<H: Hasher, F: Field>(
        &self,
        rows: &[Vec<F>],
        positions: &[usize],
    ) -> Vec<Digest> {
        assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(positions.last().is_none_or(|&last| last < self.num_leaves));
        if self.skip_levels == 0 || positions.is_empty() {
            return self.open(positions);
        }
        let k = self.skip_levels;
        let mut blocks: Vec<usize> = positions.iter().map(|&p| p >> k).collect();
        blocks.dedup();
        let work = blocks.len().saturating_mul(1usize << k).saturating_mul(rows.len());
        let local_trees: Vec<LocalSubtree> = if work >= OPEN_PARALLEL_WORK_MIN {
            parallel::map(&blocks, |&block| {
                let mut scratch = Vec::new();
                build_local_tree::<H, F>(rows, block, k, self.num_leaves, &mut scratch)
            })
        } else {
            let mut scratch = Vec::new();
            blocks
                .iter()
                .map(|&block| build_local_tree::<H, F>(rows, block, k, self.num_leaves, &mut scratch))
                .collect()
        };
        let mut opening = Vec::new();
        let mut known = positions.to_vec();
        let mut width = self.num_leaves;
        for shift in (1..=k).rev() {
            let depth = k - shift;
            let mask = (1usize << shift) - 1;
            known = parents(&known, width, |child, is_known| {
                if !is_known {
                    let blk = child >> shift;
                    let off = child & mask;
                    let bi = blocks.binary_search(&blk).expect("block was computed");
                    let tree = &local_trees[bi];
                    opening.push(tree.nodes[tree.offsets[depth] + off]);
                }
            });
            width = width.div_ceil(2);
        }
        assert!(width == self.levels[0].len());
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
