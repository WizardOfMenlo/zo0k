use crate::fields::{nonzero, uniform, uniforms};
use crate::prng::Prng;
use crate::report::{Ctx, Failure, ensure};
use zo0k::claim::{LinearForm, MainCoefficients, PowerRow, ProverClaim, VerifierClaim};
use zo0k::field::BinaryField;
use zo0k::mask::{Mask, linear_form};
use zo0k::parameters::MASK_LEN;
use zo0k::polynomial::{
    add_powers, eq_table, evaluate, evaluate_multilinear, fold, fold_rows, inner_product, round_coefficients,
};
use zo0k::{Field, Gf2_128, Gf2_256, Rng, RsField};

pub fn bits<F: Field>(index: usize, count: usize) -> Vec<F> {
    assert!(count < usize::BITS as usize);
    let point: Vec<F> =
        (0..count).rev().map(|i| if index >> i & 1 == 1 { F::ONE } else { F::ZERO }).collect();
    assert_eq!(point.len(), count);
    point
}

pub fn naive_eq<F: Field>(point: &[F], index: usize) -> F {
    assert!(point.len() < usize::BITS as usize);
    let count = point.len();
    let mut product = F::ONE;
    for (i, &z) in point.iter().enumerate() {
        let bit = index >> (count - 1 - i) & 1 == 1;
        product *= if bit { z } else { F::ONE + z };
    }
    product
}

pub fn naive_multilinear<F: Field>(table: &[F], point: &[F]) -> F {
    assert_eq!(table.len(), 1 << point.len());
    let mut sum = F::ZERO;
    for (index, &value) in table.iter().enumerate() {
        if value != F::ZERO {
            sum += value * naive_eq(point, index);
        }
    }
    sum
}

pub fn horner<F: Field>(coefficients: &[F], x: F) -> F {
    let mut value = F::ZERO;
    let mut power = F::ONE;
    for &coefficient in coefficients {
        value += coefficient * power;
        power *= x;
    }
    value
}

fn naive_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());
    let mut sum = F::ZERO;
    for (&x, &y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

pub fn polynomials<F: Field>(prng: &mut Prng) -> Result<(), String> {
    let k = prng.range(0, 10);
    let point: Vec<F> = uniforms(prng, k);
    let table = eq_table(&point);
    ensure(table.len() == 1 << k, || format!("eq table of {k} variables has {} entries", table.len()))?;
    ensure(table.iter().sum::<F>() == F::ONE, || format!("eq table of {k} variables does not sum to one"))?;
    let indices: Vec<usize> =
        if k <= 6 { (0..1 << k).collect() } else { (0..32).map(|_| prng.below(1 << k)).collect() };
    for &index in &indices {
        ensure(table[index] == naive_eq(&point, index), || format!("eq table entry {index} of {k}"))?;
    }
    ensure(eq_table(&bits::<F>(indices[0], k))[indices[0]] == F::ONE, || "eq table at a vertex".into())?;
    let log = prng.range(1, 10);
    let table: Vec<F> = uniforms(prng, 1 << log);
    let point: Vec<F> = uniforms(prng, log);
    let naive = naive_multilinear(&table, &point);
    ensure(evaluate_multilinear(&table, &point) == naive, || {
        format!("multilinear evaluation of {log} variables")
    })?;
    ensure(inner_product(&table, &eq_table(&point)) == naive, || {
        format!("eq inner product of {log} variables")
    })?;
    let index = prng.below(1 << log);
    ensure(evaluate_multilinear(&table, &bits(index, log)) == table[index], || {
        format!("evaluation at vertex {index}")
    })?;
    let (mut high, mut low) = (table.clone(), table.clone());
    fold(&mut high, F::ONE);
    fold(&mut low, F::ZERO);
    ensure(high == table[1 << (log - 1)..] && low == table[..1 << (log - 1)], || "folding at 0 or 1".into())?;
    let challenge = uniform::<F>(prng);
    let mut folded = table.clone();
    fold(&mut folded, challenge);
    let mut rest = point.clone();
    rest[0] = challenge;
    ensure(naive_multilinear(&folded, &rest[1..]) == naive_multilinear(&table, &rest), || {
        "folding fixes the first variable".into()
    })?;
    univariate::<F>(prng)
}

fn univariate<F: Field>(prng: &mut Prng) -> Result<(), String> {
    let (k, width) = (prng.range(0, 4), prng.range(1, 6));
    let table: Vec<F> = uniforms(prng, width << k);
    let challenges: Vec<F> = uniforms(prng, k);
    let mut folded = table.clone();
    for &challenge in &challenges {
        fold(&mut folded, challenge);
    }
    ensure(fold_rows(&table, &eq_table(&challenges)) == folded, || format!("fold_rows of {k} x {width}"))?;
    let len = 2 << prng.range(0, 8);
    let (mut a, mut b): (Vec<F>, Vec<F>) = (uniforms(prng, len), uniforms(prng, len));
    ensure(inner_product(&a, &b) == naive_inner_product(&a, &b), || format!("inner product of {len}"))?;
    let (constant, quadratic) = round_coefficients(&a, &b);
    let linear = inner_product(&a, &b) - constant - constant - quadratic;
    let x: F = uniform(prng);
    fold(&mut a, x);
    fold(&mut b, x);
    ensure(inner_product(&a, &b) == constant + linear * x + quadratic * x * x, || {
        format!("round coefficients of {len}")
    })?;
    let count = prng.range(0, 12);
    let (coefficients, x, scale): (Vec<F>, F, F) = (uniforms(prng, count), uniform(prng), uniform(prng));
    ensure(evaluate(&coefficients, x) == horner(&coefficients, x), || {
        format!("evaluation of degree {count}")
    })?;
    let mut powers = vec![F::ZERO; count];
    add_powers(&mut powers, x, scale);
    ensure(inner_product(&coefficients, &powers) == horner(&coefficients, x) * scale, || "add_powers".into())
}

pub fn large_polynomials() -> Result<(), String> {
    type F = Gf2_256;
    let mut prng = Prng::new(0x5eed_0001);
    let k = 16;
    let point: Vec<F> = uniforms(&mut prng, k);
    let table = eq_table(&point);
    ensure(table.iter().sum::<F>() == F::ONE, || "a large eq table does not sum to one".into())?;
    for index in [0, 1, 12345, (1 << k) / 3, (1 << k) - 1] {
        ensure(table[index] == naive_eq(&point, index), || format!("large eq table entry {index}"))?;
    }
    let values: Vec<F> = uniforms(&mut prng, 1 << k);
    let expected = naive_inner_product(&values, &table);
    ensure(inner_product(&values, &table) == expected, || "a large inner product".into())?;
    ensure(evaluate_multilinear(&values, &point) == expected, || "a large multilinear evaluation".into())?;
    let (constant, quadratic) = round_coefficients(&values, &table);
    let half = 1 << (k - 1);
    let naive_constant = naive_inner_product(&values[..half], &table[..half]);
    let differences = |v: &[F]| -> Vec<F> { (0..half).map(|i| v[half + i] - v[i]).collect() };
    let naive_quadratic = naive_inner_product(&differences(&values), &differences(&table));
    ensure((constant, quadratic) == (naive_constant, naive_quadratic), || "large round coefficients".into())?;
    let mut folded = values.clone();
    fold(&mut folded, point[0]);
    ensure(naive_inner_product(&folded, &eq_table(&point[1..])) == expected, || "a large fold".into())?;
    large_power_rows(&mut prng)
}

fn large_power_rows(prng: &mut Prng) -> Result<(), String> {
    type F = Gf2_128;
    let n = 12;
    let point: Vec<F> = uniforms(prng, n);
    let rows: Vec<PowerRow<F>> =
        (0..3).map(|_| PowerRow { scale: uniform(prng), base: uniform(prng) }).collect();
    let mut dense = eq_table(&point);
    dense.add_power_rows(&rows);
    let mut form = LinearForm::evaluation(&point);
    form.add_power_rows(&rows);
    ensure(form.to_dense() == dense, || "power rows over several blocks differ".into())?;
    let mut literal = eq_table(&point);
    for row in &rows {
        add_powers(&mut literal, row.base, row.scale);
    }
    ensure(dense == literal, || "power rows over several blocks differ from add_powers".into())
}

pub fn reed_solomon<F: RsField + BinaryField>(prng: &mut Prng) -> Result<(), String> {
    let len = prng.range(1, 700);
    let count = prng.range(0, len);
    let coefficients: Vec<F> = uniforms(prng, count);
    let codeword = F::encode(&coefficients, len);
    ensure(codeword.len() == len, || format!("a codeword of {len} has {} symbols", codeword.len()))?;
    let positions: Vec<usize> = if len <= 200 {
        (0..len).collect()
    } else {
        [0, 1, len - 2, len - 1].into_iter().chain((0..60).map(|_| prng.below(len))).collect()
    };
    for index in positions {
        let expected = horner(&coefficients, F::point(index));
        ensure(codeword[index] == expected, || {
            format!("{}: count {count}, len {len}, index {index}", F::NAME)
        })?;
        ensure(F::evaluate_at(&coefficients, index) == expected, || {
            format!("evaluate_at {index} of {count}")
        })?;
    }
    let shorter = prng.range(count.max(1), len);
    ensure(F::encode(&coefficients, shorter) == codeword[..shorter], || {
        format!("a code of {shorter} is not a prefix of {len}")
    })?;
    let other: Vec<F> = uniforms(prng, count);
    let scale: F = uniform(prng);
    let combined: Vec<F> = coefficients.iter().zip(&other).map(|(&x, &y)| x + y * scale).collect();
    let expected: Vec<F> = codeword.iter().zip(F::encode(&other, len)).map(|(&x, y)| x + y * scale).collect();
    ensure(F::encode(&combined, len) == expected, || {
        format!("encoding of {count} into {len} is not linear")
    })?;
    domain_points::<F>(prng)
}

fn domain_points<F: RsField + BinaryField>(prng: &mut Prng) -> Result<(), String> {
    let index = (prng.next_u64() & 0xffff_ffff) as usize;
    let expected: F = (0..32).filter(|bit| index >> bit & 1 == 1).map(|bit| F::point(1 << bit)).sum();
    ensure(F::point(index) == expected, || format!("point {index:#x} is not the sum of its basis elements"))?;
    ensure(F::point(index) == F::from_base(Gf2_128::point(index).limb()), || {
        format!("point {index:#x} is not in the subfield")
    })?;
    let shift = prng.range(1, 24);
    let mut value = F::point(index);
    for _ in 0..shift {
        value = value.square() + value;
    }
    ensure(value == F::point(index >> shift), || {
        format!("the subspace polynomial does not map {index:#x} to its shift by {shift}")
    })?;
    ensure(F::point(0) == F::ZERO && F::point(1) == F::ONE, || "the first points are not zero and one".into())
}

pub fn cantor_basis() -> Result<(), String> {
    let basis: Vec<u128> = (0..32).map(|i| Gf2_128::point(1 << i).limb()).collect();
    ensure(basis[0] == 1, || "the basis does not start with one".into())?;
    for pair in basis.windows(2) {
        let next = Gf2_128::new(pair[1]);
        ensure(next.square() + next == Gf2_128::new(pair[0]), || "the basis is not a Cantor basis".into())?;
    }
    let mut rows = basis;
    let mut rank = 0;
    for bit in 0..128 {
        if let Some(pivot) = (rank..rows.len()).find(|&i| rows[i] >> bit & 1 == 1) {
            rows.swap(rank, pivot);
            let pivot = rows[rank];
            for row in rows.iter_mut().skip(rank + 1) {
                if *row >> bit & 1 == 1 {
                    *row ^= pivot;
                }
            }
            rank += 1;
        }
    }
    ensure(rank == 32, || format!("the basis has rank {rank}"))
}

fn encoding_matches<F: RsField>(prng: &mut Prng, count: usize, len: usize) -> Result<(), String> {
    let coefficients: Vec<F> = uniforms(prng, count);
    let codeword = F::encode(&coefficients, len);
    ensure(codeword.len() == len, || format!("length of a codeword of {len}"))?;
    let positions: Vec<usize> = if len <= 512 {
        (0..len).collect()
    } else {
        [0, 1, len - 2, len - 1].into_iter().chain((0..300).map(|_| prng.below(len))).collect()
    };
    for index in positions {
        ensure(codeword[index] == horner(&coefficients, F::point(index)), || {
            format!("{}: count {count}, len {len}, index {index}", F::NAME)
        })?;
    }
    Ok(())
}

fn encoding_shapes<F: RsField>() -> Result<(), String> {
    let mut prng = Prng::new(0x5eed_0020);
    for len in 1..=70 {
        for count in 0..=len {
            encoding_matches::<F>(&mut prng, count, len)?;
        }
    }
    let shapes = [
        (1, 1000),
        (63, 64),
        (64, 64),
        (65, 129),
        (100, 4097),
        (1000, 5000),
        (1024, 4096),
        (1025, 4096),
        (3000, 9792),
        (4097, 8191),
        (5000, 5000),
        (8192, 40000),
        (9000, 40001),
        (3, 70000),
        (40000, 40001),
        (5000, 150001),
    ];
    for (count, len) in shapes {
        encoding_matches::<F>(&mut prng, count, len)?;
    }
    let coefficients: Vec<F> = uniforms(&mut prng, 300);
    let long = F::encode(&coefficients, 5000);
    for len in [300, 512, 513, 2049, 4999] {
        ensure(F::encode(&coefficients, len) == long[..len], || format!("a code of {len} is not a prefix"))?;
    }
    Ok(())
}

pub fn encodings() -> Result<(), String> {
    encoding_shapes::<Gf2_128>()?;
    encoding_shapes::<Gf2_256>()?;
    for index in [0usize, 1, 2, 255, 256, 0x1234_5678, 0xffff_ffff] {
        let expected: Gf2_256 =
            (0..32).filter(|bit| index >> bit & 1 == 1).map(|bit| Gf2_256::point(1 << bit)).sum();
        ensure(Gf2_256::point(index) == expected, || format!("point {index:#x}"))?;
    }
    Ok(())
}

fn mask_u<F: Field>(mask: &Mask<F>, point: &[F]) -> F {
    let folded = mask.fold(point);
    let value = inner_product(&linear_form(point), &folded);
    assert_eq!(linear_form(point).len(), MASK_LEN);
    value
}

pub fn masks<F: Field>(prng: &mut Prng) -> Result<(), String> {
    let k = prng.range(0, 6);
    let mask: Mask<F> = Mask::sample(k, &mut Rng::from_seed(prng.seed_bytes()));
    ensure(mask.folding() == k, || format!("a mask of {k} has folding {}", mask.folding()))?;
    let rows = mask.rows();
    ensure(rows.len() == 1 << k, || format!("a mask of {k} has {} rows", rows.len()))?;
    for (row, value) in rows.iter().enumerate() {
        ensure(*value == mask.fold(&bits(row, k)), || {
            format!("mask row {row} of {k} is not its value on the cube")
        })?;
    }
    let challenges: Vec<F> = uniforms(prng, k);
    let weights = eq_table(&challenges);
    let folded = mask.fold(&challenges);
    for e in 0..MASK_LEN {
        let column: Vec<F> = rows.iter().map(|row| row[e]).collect();
        ensure(fold_rows(&column, &weights) == [folded[e]], || {
            format!("mask column {e} of {k} does not fold to its value")
        })?;
    }
    let sum: F = challenges.iter().sum();
    ensure(mask_u(&mask, &challenges) == folded[0] + folded[1] * sum, || "the mask linear form".into())?;
    if k == 0 {
        return Ok(());
    }
    mask_sumcheck(prng, &mask, k)?;
    mask_map_is_onto::<F>(prng, k)
}

fn mask_sumcheck<F: Field>(prng: &mut Prng, mask: &Mask<F>, k: usize) -> Result<(), String> {
    assert!(k >= 1 && mask.folding() == k);
    let mut sumcheck = mask.sumcheck();
    let total: F = (0..1 << k).map(|b| mask_u(mask, &bits::<F>(b, k))).sum();
    ensure(sumcheck.sum() == total, || format!("the mask sum of {k} differs from the sum over the cube"))?;
    let mut prefix: Vec<F> = Vec::new();
    for round in 1..=k {
        let partial = |x: F| -> F {
            (0..1usize << (k - round))
                .map(|b| mask_u(mask, &[&prefix[..], &[x], &bits::<F>(b, k - round)].concat()))
                .sum()
        };
        let (constant, quadratic) = sumcheck.round_coefficients();
        let x: F = uniform(prng);
        let linear = partial(F::ONE) - constant - quadratic;
        ensure(partial(F::ZERO) == constant, || format!("mask round {round} of {k}: constant"))?;
        ensure(partial(x) == constant + linear * x + quadratic * x * x, || {
            format!("mask round {round} of {k}: polynomial")
        })?;
        let challenge: F = uniform(prng);
        sumcheck.bind(challenge);
        prefix.push(challenge);
    }
    Ok(())
}

pub fn rank<F: Field>(mut matrix: Vec<Vec<F>>) -> usize {
    let columns = matrix.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..matrix.len()).find(|&i| matrix[i][column] != F::ZERO) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = matrix[rank][column].inverse();
        let pivot_row: Vec<F> = matrix[rank].iter().map(|&entry| entry * inverse).collect();
        for (i, row) in matrix.iter_mut().enumerate() {
            if i != rank {
                let factor = row[column];
                row.iter_mut().zip(&pivot_row).for_each(|(entry, &p)| *entry -= factor * p);
            }
        }
        rank += 1;
    }
    assert!(rank <= matrix.len() && rank <= columns);
    rank
}

fn mask_map_is_onto<F: Field>(prng: &mut Prng, k: usize) -> Result<(), String> {
    assert!(k >= 1);
    let challenge_sets: [Vec<F>; 3] = [vec![F::ZERO; k], vec![F::ONE; k], uniforms(prng, k)];
    for (set, challenges) in challenge_sets.iter().enumerate() {
        let mut matrix = Vec::new();
        for _ in 0..MASK_LEN * (k + 1) {
            let mask: Mask<F> = Mask::sample(k, &mut Rng::from_seed(prng.seed_bytes()));
            let mut sumcheck = mask.sumcheck();
            let mut image = vec![sumcheck.sum()];
            for &challenge in challenges {
                let (constant, quadratic) = sumcheck.round_coefficients();
                image.extend([constant, quadratic]);
                sumcheck.bind(challenge);
            }
            matrix.push(image);
        }
        let found = rank(matrix);
        ensure(found == 1 + 2 * k, || {
            format!("the mask transcript map of {k} has rank {found} on challenge set {set}")
        })?;
    }
    Ok(())
}

pub fn claims<F: RsField>(prng: &mut Prng) -> Result<(), String> {
    let num_variables = prng.range(1, 10);
    let point: Vec<F> = uniforms(prng, num_variables);
    let mut form = LinearForm::evaluation(&point);
    let mut dense = eq_table(&point);
    ensure(form.to_dense() == dense, || "the evaluation form differs from the eq table".into())?;
    let rows: Vec<PowerRow<F>> = (0..prng.range(0, 5))
        .map(|i| PowerRow {
            scale: uniform(prng),
            base: if i % 2 == 0 { uniform(prng) } else { F::point(prng.below(1 << 20)) },
        })
        .collect();
    form.add_power_rows(&rows);
    dense.add_power_rows(&rows);
    ensure(form.to_dense() == dense, || format!("power rows differ on {num_variables} variables"))?;
    let mut literal = eq_table(&point);
    for row in &rows {
        add_powers(&mut literal, row.base, row.scale);
    }
    ensure(dense == literal, || "power rows differ from add_powers".into())?;
    let fixed_count = prng.range(0, num_variables);
    let challenges: Vec<F> = uniforms(prng, fixed_count);
    form.fix(&challenges);
    for &challenge in &challenges {
        fold(&mut dense, challenge);
    }
    ensure(form.to_dense() == dense, || {
        format!("fixing {} variables differs from folding", challenges.len())
    })?;
    let factor: F = uniform(prng);
    form.scale(factor);
    dense.iter_mut().for_each(|entry| *entry *= factor);
    let some = prng.range(0, rows.len());
    form.add_power_rows(&rows[..some]);
    dense.add_power_rows(&rows[..some]);
    ensure(form.to_dense() == dense, || "scaling and more power rows".into())?;
    let rest = num_variables - challenges.len();
    form.fix(&uniforms::<F>(prng, rest));
    ensure(form.to_dense().len() == 1, || "a fully fixed form is not a scalar".into())?;
    honest_claim::<F>(prng)
}

fn honest_claim<F: RsField>(prng: &mut Prng) -> Result<(), String> {
    let n = prng.range(0, 8);
    let (table, point): (Vec<F>, Vec<F>) = (uniforms(prng, 1 << n), uniforms(prng, n));
    let value = naive_multilinear(&table, &point);
    let prover = ProverClaim::evaluation(&point, value);
    let verifier = VerifierClaim::evaluation(&point, value);
    ensure(inner_product(&prover.main, &table) == value, || "the evaluation claim does not hold".into())?;
    ensure(verifier.main.to_dense() == prover.main && verifier.target == value, || {
        "the claims differ".into()
    })?;
    let wrong = value + nonzero::<F>(prng);
    ensure(inner_product(&prover.main, &table) != wrong, || "a wrong value satisfies the claim".into())
}

fn per_field<F: RsField + BinaryField>(ctx: &mut Ctx) -> Result<(), Failure> {
    ctx.check("polynomial_identities", polynomials::<F>)?;
    ctx.check("reed_solomon_encoding", reed_solomon::<F>)?;
    ctx.check("mask_rows_fold_and_sumcheck", masks::<F>)?;
    ctx.check("claim_representations_agree", claims::<F>)?;
    Ok(())
}

pub fn scenario(ctx: &mut Ctx) -> Result<(), Failure> {
    if ctx.prng.chance(50) { per_field::<Gf2_128>(ctx) } else { per_field::<Gf2_256>(ctx) }
}
