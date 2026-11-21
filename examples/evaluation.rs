use std::time::Instant;
use zo0k::{Blake3, Field, Gf2_128, Parameters, Protocol, Rng};

fn main() -> Result<(), zo0k::Error> {
    type F = Gf2_128;
    let num_variables = 18;
    let parameters = Parameters::standard(num_variables, 100, F::SIZE)?;
    let protocol = Protocol::<F, Blake3>::new(parameters)?;

    let mut rng = Rng::from_entropy();
    let table: Vec<F> = rng.fields(1 << num_variables);
    let point: Vec<F> = rng.fields(num_variables);

    let start = Instant::now();
    let (commitment, witness) = protocol.commit(table, &mut rng)?;
    println!("commit: {:.1?}", start.elapsed());

    let start = Instant::now();
    let (value, proof) = protocol.prove(&commitment, witness, &point, &mut rng)?;
    println!("prove:  {:.1?}, {} bytes", start.elapsed(), proof.0.len());

    let start = Instant::now();
    protocol.verify(&commitment, &point, value, &proof)?;
    println!("verify: {:.1?}", start.elapsed());
    Ok(())
}
