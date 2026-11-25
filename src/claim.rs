use crate::field::Field;
use crate::parallel;
use crate::parameters::MASK_LEN;
use crate::polynomial::{eq_table, inner_product, round_coefficients};
use crate::reed_solomon::RsField;
use crate::witness::Row;
use std::mem::MaybeUninit;

const EVAL_PARALLEL_OPS: usize = 1 << 13;
const EVAL_TERMS_PIECE_MIN: usize = 4;
const STACK_TERM_HALF_MAX: usize = 256;
const PARALLEL_EQ_DOT_MIN: usize = 1 << 14;

pub(crate) const fn l1_tile_log<F: Field>() -> u32 {
    if F::BYTES <= 16 { 10 } else { 9 }
}

pub(crate) const fn l1_tile_len<F: Field>() -> usize {
    1usize << l1_tile_log::<F>()
}

#[derive(Clone, Debug)]
pub struct Claim<F, M> {
    pub main: M,
    pub aux: Vec<F>,
    pub target: F,
}

#[derive(Clone, Debug)]
pub struct FoldedClaim<F, M> {
    pub main: M,
    pub aux: Vec<F>,
    pub mask: [F; MASK_LEN],
    pub target: F,
}

pub type ProverClaim<F> = Claim<F, Vec<F>>;
pub type ProverFoldedClaim<F> = FoldedClaim<F, Vec<F>>;

pub type VerifierClaim<F> = Claim<F, LinearForm<F>>;
pub type VerifierFoldedClaim<F> = FoldedClaim<F, LinearForm<F>>;

impl<F: Field> ProverFoldedClaim<F> {
    pub fn evaluate(&self, row: &Row<F>) -> F {
        inner_product(&self.main, &row.main)
            + inner_product(&self.aux, &row.aux)
            + inner_product(&self.mask, &row.mask)
    }
}

pub(crate) struct EqFactor<F> {
    pub hi: Vec<F>,
    pub lo: Vec<F>,
}

impl<F: Field> EqFactor<F> {
    pub(crate) fn new(point: &[F]) -> Self {
        let lo_bits = point.len().min(l1_tile_log::<F>() as usize);
        let split = point.len() - lo_bits;
        let (hi_point, lo_point) = point.split_at(split);
        let hi = eq_table(hi_point);
        let lo = eq_table(lo_point);
        assert_eq!(hi.len(), 1usize << split);
        assert_eq!(lo.len(), 1usize << lo_bits);
        Self { hi, lo }
    }

    pub(crate) fn len(&self) -> usize {
        self.hi.len() * self.lo.len()
    }

    pub(crate) fn inner_product(&self, values: &[F]) -> F {
        assert_eq!(values.len(), self.len());
        let tile = self.lo.len();
        let tiles = self.hi.len();
        assert!(tile >= 1 && tiles >= 1);
        if values.len() < PARALLEL_EQ_DOT_MIN || tiles < 2 {
            let mut sum = F::ZERO;
            for (t, &w) in self.hi.iter().enumerate() {
                sum += w * F::inner_product_slice(&self.lo, &values[t * tile..(t + 1) * tile]);
            }
            return sum;
        }
        let partials = parallel::fold_pieces(
            tiles,
            1,
            || F::ZERO,
            |sum, range| {
                for t in range {
                    *sum += self.hi[t] * F::inner_product_slice(&self.lo, &values[t * tile..(t + 1) * tile]);
                }
            },
        );
        partials.into_iter().fold(F::ZERO, |a, b| a + b)
    }

    pub(crate) fn to_dense_scaled(&self, scale: F) -> Vec<F> {
        let total = self.len();
        let tile = self.lo.len();
        assert!(total >= 1 && tile >= 1);
        let mut out = vec![F::ZERO; total];
        for (t, &w) in self.hi.iter().enumerate() {
            F::add_scaled_slice(&mut out[t * tile..(t + 1) * tile], &self.lo, scale * w);
        }
        out
    }
}

pub(crate) struct EqFoldedClaim<F> {
    pub eq: EqFactor<F>,
    pub scale: F,
    pub mask: [F; MASK_LEN],
    pub target: F,
}

#[derive(Clone, Copy, Debug)]
pub struct PowerRow<F> {
    pub scale: F,
    pub base: F,
}

pub trait MainCoefficients<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]);

    fn add_power_rows_positioned(&mut self, rows: &[(PowerRow<F>, Option<usize>)])
    where
        F: Copy,
    {
        let plain: Vec<PowerRow<F>> = rows.iter().map(|&(row, _)| row).collect();
        self.add_power_rows(&plain);
    }
}

fn square_pow2<F: Field>(mut value: F, log_exp: u32) -> F {
    assert!(log_exp <= 63);
    for _ in 0..log_exp {
        value = value.square();
    }
    assert!(log_exp < 64);
    value
}

fn reset_powers_at<F: Field>(
    powers: &mut Vec<F>,
    rows: &[PowerRow<F>],
    tile_steps: &[F],
    tile_log: u32,
    pos: usize,
) {
    assert_eq!(rows.len(), tile_steps.len());
    powers.clear();
    let tile_len = 1usize << tile_log;
    if pos.is_multiple_of(tile_len) {
        let tile_idx = (pos >> tile_log) as u64;
        for (row, &step) in rows.iter().zip(tile_steps) {
            powers.push(row.scale * step.pow(tile_idx));
        }
    } else {
        let exp = pos as u64;
        for row in rows {
            powers.push(row.scale * row.base.pow(exp));
        }
    }
    assert_eq!(powers.len(), rows.len());
}

fn add_power_rows_slice<F: Field>(target: &mut [F], rows: &[PowerRow<F>]) {
    if rows.is_empty() || target.is_empty() {
        return;
    }
    assert!(target.len() <= isize::MAX as usize);
    let tile_log = l1_tile_log::<F>();
    let tile_len = 1usize << tile_log;
    assert_eq!(tile_len, l1_tile_len::<F>());
    let tile_steps: Vec<F> = rows.iter().map(|row| square_pow2(row.base, tile_log)).collect();
    let piece_min = tile_len.max(1 << 11);
    parallel::for_each_piece_mut_with(
        target,
        piece_min,
        || (Vec::with_capacity(rows.len()), usize::MAX),
        |state, piece_start, piece| {
            if state.1 != piece_start {
                reset_powers_at(&mut state.0, rows, &tile_steps, tile_log, piece_start);
            }
            for tile in piece.chunks_mut(tile_len) {
                let full = tile.len() == tile_len;
                for ((row, power), &step) in rows.iter().zip(state.0.iter_mut()).zip(&tile_steps) {
                    F::add_powers_slice(tile, row.base, *power);
                    *power *= if full { step } else { row.base.pow(tile.len() as u64) };
                }
            }
            state.1 = piece_start + piece.len();
        },
    );
}

impl<F: Field> MainCoefficients<F> for Vec<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        add_power_rows_slice(self, rows);
    }
}

struct FusedPowerState<F> {
    lo_powers: Vec<F>,
    hi_powers: Vec<F>,
    next_tile: usize,
    constant: F,
    quadratic: F,
    scratch: Vec<F>,
}

impl<F: Field> FusedPowerState<F> {
    fn new(rows_len: usize, scratch_len: usize) -> Self {
        Self {
            lo_powers: Vec::with_capacity(rows_len),
            hi_powers: Vec::with_capacity(rows_len),
            next_tile: usize::MAX,
            constant: F::ZERO,
            quadratic: F::ZERO,
            scratch: vec![F::ZERO; scratch_len],
        }
    }

    fn sync_tile(&mut self, rows: &[PowerRow<F>], tile_steps: &[F], half_steps: &[F], tile_idx: usize) {
        assert!(rows.len() == tile_steps.len() && rows.len() == half_steps.len());
        if self.next_tile == tile_idx {
            return;
        }
        self.lo_powers.clear();
        self.hi_powers.clear();
        let exp = tile_idx as u64;
        for ((row, &step), &half_step) in rows.iter().zip(tile_steps).zip(half_steps) {
            let lo = row.scale * step.pow(exp);
            self.lo_powers.push(lo);
            self.hi_powers.push(lo * half_step);
        }
        self.next_tile = tile_idx;
        assert_eq!(self.lo_powers.len(), rows.len());
    }

    fn apply_and_advance(&mut self, lo_tile: &mut [F], hi_tile: &mut [F], rows: &[PowerRow<F>], steps: &[F]) {
        assert_eq!(lo_tile.len(), hi_tile.len());
        for (((row, lp), hp), &step) in
            rows.iter().zip(self.lo_powers.iter_mut()).zip(self.hi_powers.iter_mut()).zip(steps)
        {
            F::add_powers_slice(lo_tile, row.base, *lp);
            F::add_powers_slice(hi_tile, row.base, *hp);
            *lp *= step;
            *hp *= step;
        }
        self.next_tile += 1;
        assert_eq!(self.lo_powers.len(), rows.len());
    }
}

fn write_scaled_uninit<'a, F: Field>(dst: &'a mut [MaybeUninit<F>], src: &[F], scale: F) -> &'a mut [F] {
    let len = dst.len();
    assert_eq!(len, src.len());
    for slot in dst.iter_mut() {
        slot.write(F::ZERO);
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<F>(), len) };
    F::add_scaled_slice(slice, src, scale);
    assert_eq!(slice.len(), len);
    slice
}

fn scale_tile_with_scratch<F: Field>(tile: &mut [F], scale: F, scratch: &mut [F]) {
    assert_eq!(tile.len(), scratch.len());
    if scale == F::ONE {
        return;
    }
    if scale == F::ZERO {
        tile.fill(F::ZERO);
        return;
    }
    scratch.fill(F::ZERO);
    F::add_scaled_slice(scratch, tile, scale);
    tile.copy_from_slice(scratch);
    assert!(!tile.is_empty());
}

pub(crate) fn materialize_and_round_from_eq<F: Field>(
    eq: &EqFactor<F>,
    scale: F,
    rows: &[PowerRow<F>],
    next_main: &[F],
) -> (Vec<F>, (F, F)) {
    let total = eq.len();
    assert_eq!(next_main.len(), total);
    assert!(total >= 2 && total.is_power_of_two());
    let half = total / 2;
    let tile_len = eq.lo.len();
    if half < tile_len {
        let mut main_form = eq.to_dense_scaled(scale);
        add_power_rows_slice(&mut main_form, rows);
        let r0 = round_coefficients(&main_form, next_main);
        return (main_form, r0);
    }
    let tile_log = tile_len.trailing_zeros();
    let half_tiles = half / tile_len;
    assert_eq!(eq.hi.len(), 2 * half_tiles);
    let tile_steps: Vec<F> = rows.iter().map(|r| square_pow2(r.base, tile_log)).collect();
    let half_steps: Vec<F> = rows.iter().map(|r| square_pow2(r.base, half.trailing_zeros())).collect();
    let (main_lo, main_hi) = next_main.split_at(half);
    let mut main_form: Vec<F> = Vec::with_capacity(total);
    let (form_lo, form_hi) = main_form.spare_capacity_mut()[..total].split_at_mut(half);
    let chunks_min = (1usize << 11).div_ceil(tile_len).max(1);
    let states = parallel::fold_pair_chunks_mut_with(
        form_lo,
        form_hi,
        tile_len,
        chunks_min,
        || FusedPowerState::new(rows.len(), 0),
        |state, tile_start, lo_piece, hi_piece| {
            state.sync_tile(rows, &tile_steps, &half_steps, tile_start);
            for (k, (lo_u, hi_u)) in
                lo_piece.chunks_exact_mut(tile_len).zip(hi_piece.chunks_exact_mut(tile_len)).enumerate()
            {
                let t = tile_start + k;
                let lo_tile = write_scaled_uninit(lo_u, &eq.lo, scale * eq.hi[t]);
                let hi_tile = write_scaled_uninit(hi_u, &eq.lo, scale * eq.hi[half_tiles + t]);
                state.apply_and_advance(lo_tile, hi_tile, rows, &tile_steps);
                let off = t * tile_len;
                let (tc, tq) = F::round_coefficients_slice(
                    lo_tile,
                    hi_tile,
                    &main_lo[off..off + tile_len],
                    &main_hi[off..off + tile_len],
                );
                state.constant += tc;
                state.quadratic += tq;
            }
        },
    );
    unsafe { main_form.set_len(total) };
    let r0 = states.into_iter().fold((F::ZERO, F::ZERO), |a, s| (a.0 + s.constant, a.1 + s.quadratic));
    (main_form, r0)
}

pub(crate) fn scale_add_powers_and_round<F: Field>(
    main_form: &mut [F],
    scale: F,
    rows: &[PowerRow<F>],
    next_main: &[F],
) -> (F, F) {
    let total = main_form.len();
    assert_eq!(next_main.len(), total);
    assert!(total >= 2 && total.is_power_of_two());
    let half = total / 2;
    let tile_len = l1_tile_len::<F>();
    if half < tile_len {
        let mut scratch = vec![F::ZERO; total];
        scale_tile_with_scratch(main_form, scale, &mut scratch);
        add_power_rows_slice(main_form, rows);
        return round_coefficients(main_form, next_main);
    }
    let tile_log = l1_tile_log::<F>();
    let tile_steps: Vec<F> = rows.iter().map(|r| square_pow2(r.base, tile_log)).collect();
    let half_steps: Vec<F> = rows.iter().map(|r| square_pow2(r.base, half.trailing_zeros())).collect();
    let (form_lo, form_hi) = main_form.split_at_mut(half);
    let (main_lo, main_hi) = next_main.split_at(half);
    let need_scratch = if scale == F::ONE || scale == F::ZERO { 0 } else { tile_len };
    let chunks_min = (1usize << 11).div_ceil(tile_len).max(1);
    let states = parallel::fold_pair_chunks_mut_with(
        form_lo,
        form_hi,
        tile_len,
        chunks_min,
        || FusedPowerState::new(rows.len(), need_scratch),
        |state, tile_start, lo_piece, hi_piece| {
            state.sync_tile(rows, &tile_steps, &half_steps, tile_start);
            for (k, (lo_tile, hi_tile)) in
                lo_piece.chunks_exact_mut(tile_len).zip(hi_piece.chunks_exact_mut(tile_len)).enumerate()
            {
                if need_scratch > 0 {
                    scale_tile_with_scratch(lo_tile, scale, &mut state.scratch);
                    scale_tile_with_scratch(hi_tile, scale, &mut state.scratch);
                } else if scale == F::ZERO {
                    lo_tile.fill(F::ZERO);
                    hi_tile.fill(F::ZERO);
                }
                state.apply_and_advance(lo_tile, hi_tile, rows, &tile_steps);
                let off = (tile_start + k) * tile_len;
                let (tc, tq) = F::round_coefficients_slice(
                    lo_tile,
                    hi_tile,
                    &main_lo[off..off + tile_len],
                    &main_hi[off..off + tile_len],
                );
                state.constant += tc;
                state.quadratic += tq;
            }
        },
    );
    states.into_iter().fold((F::ZERO, F::ZERO), |a, s| (a.0 + s.constant, a.1 + s.quadratic))
}

#[derive(Clone, Copy, Debug)]
struct PowerTerm<F> {
    scale: F,
    base: F,
    position: Option<usize>,
    squares_start: usize,
}

#[derive(Clone, Debug)]
pub struct LinearForm<F> {
    num_variables: usize,
    min_variables: usize,
    terms: Vec<Term<F>>,
    power_terms: Vec<PowerTerm<F>>,
    squares: Vec<F>,
}

#[derive(Clone, Debug)]
struct Term<F> {
    scale: F,
    factors: Vec<[F; 2]>,
}

fn fold_step_in_place<F: Field>(lo: &mut [F], hi: &[F], zero: F, one: F) {
    assert_eq!(lo.len(), hi.len());
    if zero + one == F::ONE {
        F::fold_slice(lo, hi, one);
    } else {
        for (l, &h) in lo.iter_mut().zip(hi) {
            *l = *l * zero + h * one;
        }
    }
    assert!(lo.len() <= isize::MAX as usize);
}

fn eval_term_in_buffer<F: Field>(term: &Term<F>, folded: &mut [F], values_hi: &[F]) -> F {
    assert_eq!(folded.len(), values_hi.len());
    let [zero, one] = term.factors[0];
    fold_step_in_place(folded, values_hi, zero, one);
    let mut active = folded.len();
    for &[zero, one] in &term.factors[1..] {
        let next = active / 2;
        let (lo, hi) = folded[..active].split_at_mut(next);
        fold_step_in_place(lo, hi, zero, one);
        active = next;
    }
    assert_eq!(active, 1);
    term.scale * folded[0]
}

fn eval_term<F: Field>(term: &Term<F>, values: &[F]) -> F {
    assert_eq!(values.len(), 1usize << term.factors.len());
    assert!(term.factors.len() <= 60);
    if term.factors.is_empty() {
        return term.scale * values[0];
    }
    let half = values.len() / 2;
    if half <= STACK_TERM_HALF_MAX {
        let mut buf = [F::ZERO; STACK_TERM_HALF_MAX];
        buf[..half].copy_from_slice(&values[..half]);
        return eval_term_in_buffer(term, &mut buf[..half], &values[half..]);
    }
    let mut folded = values[..half].to_vec();
    eval_term_in_buffer(term, &mut folded, &values[half..])
}

fn eval_power_terms<F: Field>(
    terms: &[PowerTerm<F>],
    values: &[F],
    eval_one: impl Fn(&PowerTerm<F>, &[F]) -> F + Sync,
) -> F {
    assert!(!values.is_empty());
    if terms.is_empty() {
        return F::ZERO;
    }
    if values.len() == 1 {
        let sum: F = terms.iter().map(|term| term.scale).sum();
        assert!(terms.len() <= isize::MAX as usize);
        return sum * values[0];
    }
    let eval_slice =
        |slice: &[PowerTerm<F>]| -> F { slice.iter().map(|term| term.scale * eval_one(term, values)).sum() };
    let ops = terms.len().saturating_mul(values.len());
    assert!(values.len() >= 2);
    if ops < EVAL_PARALLEL_OPS || terms.len() < EVAL_TERMS_PIECE_MIN {
        eval_slice(terms)
    } else {
        parallel::fold_pieces(
            terms.len(),
            EVAL_TERMS_PIECE_MIN,
            || F::ZERO,
            |acc, range| *acc += eval_slice(&terms[range]),
        )
        .into_iter()
        .sum()
    }
}

impl<F: Field> LinearForm<F> {
    pub fn evaluation(point: &[F]) -> Self {
        Self::evaluation_with_min_vars(point, 0)
    }

    pub(crate) fn evaluation_with_min_vars(point: &[F], min_variables: usize) -> Self {
        assert!(min_variables <= point.len());
        let factors = point.iter().map(|&z| [F::ONE - z, z]).collect();
        let form = Self {
            num_variables: point.len(),
            min_variables,
            terms: vec![Term { scale: F::ONE, factors }],
            power_terms: Vec::new(),
            squares: Vec::new(),
        };
        assert_eq!(form.num_variables, point.len());
        form
    }

    pub(crate) fn fix_and_scale(&mut self, challenges: &[F], batching: F) {
        assert!(challenges.len() <= self.num_variables);
        let old_vars = self.num_variables;
        let new_vars = old_vars - challenges.len();
        assert!(new_vars >= self.min_variables);
        for term in &mut self.terms {
            let mut factor = batching;
            for ([zero, one], &challenge) in term.factors.drain(..challenges.len()).zip(challenges) {
                factor *= zero + (one - zero) * challenge;
            }
            term.scale *= factor;
        }
        let start_off = new_vars - self.min_variables;
        let end_off = old_vars - self.min_variables;
        for term in &mut self.power_terms {
            let mut factor = batching;
            if !challenges.is_empty() {
                let powers = &self.squares[term.squares_start + start_off..term.squares_start + end_off];
                for (&power, &challenge) in powers.iter().zip(challenges.iter().rev()) {
                    factor *= F::ONE + (power - F::ONE) * challenge;
                }
            }
            term.scale *= factor;
        }
        self.num_variables = new_vars;
        assert!(self.terms.iter().all(|term| term.factors.len() == self.num_variables));
    }

    pub fn fix(&mut self, challenges: &[F]) {
        self.fix_and_scale(challenges, F::ONE);
    }

    pub fn scale(&mut self, factor: F) {
        for term in &mut self.terms {
            term.scale *= factor;
        }
        for term in &mut self.power_terms {
            term.scale *= factor;
        }
    }

    pub fn evaluate(&self, values: &[F]) -> F {
        assert!(self.num_variables <= 60);
        assert_eq!(values.len(), 1usize << self.num_variables);
        let mut total = F::ZERO;
        for term in &self.terms {
            total += eval_term(term, values);
        }
        total
            + eval_power_terms(&self.power_terms, values, |term, vals| F::eval_horner_slice(vals, term.base))
    }

    pub fn to_dense(&self) -> Vec<F> {
        let mut dense = vec![F::ZERO; 1 << self.num_variables];
        let mut tensor = Vec::with_capacity(dense.len());
        for term in &self.terms {
            tensor.clear();
            tensor.push(term.scale);
            for &[zero, one] in term.factors.iter().rev() {
                for i in 0..tensor.len() {
                    tensor.push(tensor[i] * one);
                    if zero != F::ONE {
                        tensor[i] *= zero;
                    }
                }
            }
            dense.iter_mut().zip(&tensor).for_each(|(entry, &value)| *entry += value);
        }
        let power_rows: Vec<PowerRow<F>> =
            self.power_terms.iter().map(|term| PowerRow { scale: term.scale, base: term.base }).collect();
        dense.add_power_rows(&power_rows);
        assert_eq!(dense.len(), 1 << self.num_variables);
        dense
    }

    fn push_power_term(&mut self, row: PowerRow<F>, position: Option<usize>) {
        if row.scale == F::ZERO {
            return;
        }
        if let Some(existing) = self.power_terms.iter_mut().find(|term| term.base == row.base) {
            existing.scale += row.scale;
            if existing.position.is_none() {
                existing.position = position;
            }
            return;
        }
        let vars = self.num_variables;
        let min_vars = self.min_variables;
        assert!(min_vars <= vars);
        let squares_start = self.squares.len();
        let mut power = row.base;
        for step in 0..vars {
            if step >= min_vars {
                self.squares.push(power);
            }
            if step + 1 < vars {
                power = power.square();
            }
        }
        assert_eq!(self.squares.len(), squares_start + (vars - min_vars));
        self.power_terms.push(PowerTerm { scale: row.scale, base: row.base, position, squares_start });
    }
}

impl<F: RsField> LinearForm<F> {
    pub(crate) fn evaluate_rs(&self, values: &[F]) -> F {
        assert!(self.num_variables <= 60);
        assert_eq!(values.len(), 1usize << self.num_variables);
        let mut total = F::ZERO;
        for term in &self.terms {
            total += eval_term(term, values);
        }
        total
            + eval_power_terms(&self.power_terms, values, |term, vals| match term.position {
                Some(pos) => F::evaluate_at(vals, pos),
                None => F::eval_horner_slice(vals, term.base),
            })
    }
}

impl<F: Field> MainCoefficients<F> for LinearForm<F> {
    fn add_power_rows(&mut self, rows: &[PowerRow<F>]) {
        let stored = self.num_variables - self.min_variables;
        self.power_terms.reserve(rows.len());
        self.squares.reserve(rows.len().saturating_mul(stored));
        for &row in rows {
            self.push_power_term(row, None);
        }
        assert!(self.squares.len() >= self.power_terms.len() * stored);
        assert!(self.power_terms.len() <= self.power_terms.capacity());
    }

    fn add_power_rows_positioned(&mut self, rows: &[(PowerRow<F>, Option<usize>)]) {
        let stored = self.num_variables - self.min_variables;
        self.power_terms.reserve(rows.len());
        self.squares.reserve(rows.len().saturating_mul(stored));
        for &(row, position) in rows {
            self.push_power_term(row, position);
        }
        assert!(self.squares.len() >= self.power_terms.len() * stored);
        assert!(self.power_terms.len() <= self.power_terms.capacity());
    }
}

impl<F: Field> VerifierClaim<F> {
    pub fn evaluation(point: &[F], value: F) -> Self {
        Self { main: LinearForm::evaluation(point), aux: Vec::new(), target: value }
    }

    pub(crate) fn evaluation_with_min_vars(point: &[F], value: F, min_vars: usize) -> Self {
        Self { main: LinearForm::evaluation_with_min_vars(point, min_vars), aux: Vec::new(), target: value }
    }
}

impl<F: Field> ProverClaim<F> {
    pub fn evaluation(point: &[F], value: F) -> Self {
        Self { main: eq_table(point), aux: Vec::new(), target: value }
    }
}
