//! Command-line interface: benchmarks, verification, parameter analysis and
//! the tables of the paper.

mod bench;
mod paper;
mod search;

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use zo0k::tuning::{Analysis, optimize_queries};
use zo0k::{Blake3, Field, FieldSize, Gf2_128, Gf2_256, Hasher, Keccak256, Parameters, Variant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
#[command(version, about = "Zero-knowledge IOPP for constrained interleaved Reed-Solomon codes")]
struct Cli {
    /// Number of worker threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// The field of the protocol.
    #[arg(long, global = true, value_enum, default_value_t = FieldChoice::Gf2_256)]
    field: FieldChoice,
    /// The hash function of Merkle trees and of the Fiat-Shamir transformation.
    #[arg(long, global = true, value_enum, default_value_t = HashChoice::Blake3)]
    hash: HashChoice,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum FieldChoice {
    #[value(name = Gf2_256::NAME)]
    Gf2_256,
    #[value(name = Gf2_128::NAME)]
    Gf2_128,
}

impl FieldChoice {
    fn size(self) -> FieldSize {
        match self {
            Self::Gf2_256 => Gf2_256::SIZE,
            Self::Gf2_128 => Gf2_128::SIZE,
        }
    }
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum HashChoice {
    #[value(name = Blake3::NAME)]
    Blake3,
    #[value(name = Keccak256::NAME)]
    Keccak256,
}

/// Call a function that is generic over the field and the hash function with
/// the types of the given choices.
macro_rules! with_types {
    ($field:expr, $hash:expr, $($function:ident)::+, $($argument:expr),*) => {
        match ($field, $hash) {
            ($crate::FieldChoice::Gf2_256, $crate::HashChoice::Blake3) => {
                $($function)::+::<zo0k::Gf2_256, zo0k::Blake3>($($argument),*)
            }
            ($crate::FieldChoice::Gf2_256, $crate::HashChoice::Keccak256) => {
                $($function)::+::<zo0k::Gf2_256, zo0k::Keccak256>($($argument),*)
            }
            ($crate::FieldChoice::Gf2_128, $crate::HashChoice::Blake3) => {
                $($function)::+::<zo0k::Gf2_128, zo0k::Blake3>($($argument),*)
            }
            ($crate::FieldChoice::Gf2_128, $crate::HashChoice::Keccak256) => {
                $($function)::+::<zo0k::Gf2_128, zo0k::Keccak256>($($argument),*)
            }
        }
    };
}
use with_types;

#[derive(Subcommand)]
enum Command {
    /// Commit to a random table, prove an evaluation, verify it, and report
    /// times and sizes.
    Bench(bench::Bench),
    /// Verify a proof saved by `bench --out`.
    Verify {
        /// The directory that holds `statement.json` and `proof.bin`.
        directory: PathBuf,
    },
    /// Print the layout, soundness errors and sizes of parameters as JSON.
    /// Fails after printing if the soundness error is above the target.
    Analyze {
        #[command(flatten)]
        source: Source,
        /// Analyze the protocol without zero knowledge instead.
        #[arg(long)]
        baseline: bool,
    },
    /// Recompute the tables and the frontier data of the paper's evaluation
    /// from its configurations.
    Paper(paper::Paper),
    /// Search for a schedule with a small argument within a limit on the
    /// IOP proof length, and print its parameters.
    Search(search::Search),
}

/// Parse an integer between `MIN` and `MAX`.
fn in_range<const MIN: usize, const MAX: usize>(text: &str) -> std::result::Result<usize, String> {
    match text.parse() {
        Ok(value) if (MIN..=MAX).contains(&value) => Ok(value),
        _ => Err(format!("expected an integer between {MIN} and {MAX}")),
    }
}

/// Where parameters come from.
#[derive(Args)]
struct Source {
    /// A JSON file with parameters.
    #[arg(long, group = "source")]
    params: Option<PathBuf>,
    /// The configuration of a row of the paper's tables, such as
    /// `sublinear-16` or `polylog-24`.
    #[arg(long, value_name = "ROW", group = "source")]
    paper: Option<String>,
    /// Standard parameters for tables of 2^N entries.
    #[arg(long, value_name = "N", group = "source", value_parser = in_range::<1, 33>)]
    num_variables: Option<usize>,
    /// Security target of the standard parameters, in bits.
    #[arg(long, default_value_t = 100, conflicts_with_all = ["params", "paper"])]
    security: u32,
    /// Replace the query counts by optimized ones.
    #[arg(long)]
    optimize_queries: bool,
}

impl Source {
    fn parameters(&self, variant: Variant, field: FieldSize) -> Result<Parameters> {
        let mut parameters = match (&self.params, &self.paper, self.num_variables) {
            (Some(path), _, _) => serde_json::from_slice(&fs::read(path)?)?,
            (_, Some(row), _) => {
                let selected = paper::table_row(row)?;
                match variant {
                    Variant::ZeroKnowledge => selected.zero_knowledge,
                    Variant::Baseline => selected.baseline,
                }
            }
            (_, _, Some(num_variables)) => {
                // The standard schedule, with query counts for the variant.
                let mut parameters = Parameters::standard(num_variables, self.security, field)?;
                if variant == Variant::Baseline {
                    optimize_queries(&mut parameters, variant, field)?;
                }
                parameters
            }
            _ => return Err("give --params FILE, --paper ROW or --num-variables N".into()),
        };
        if self.optimize_queries {
            optimize_queries(&mut parameters, variant, field)?;
        }
        Ok(parameters)
    }
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    let field = cli.field.size();
    match &cli.command {
        Command::Bench(arguments) => with_types!(cli.field, cli.hash, bench::bench, arguments),
        Command::Verify { directory } => bench::verify(directory),
        Command::Analyze { source, baseline } => {
            let variant = if *baseline { Variant::Baseline } else { Variant::ZeroKnowledge };
            let parameters = source.parameters(variant, field)?;
            let analysis = Analysis::ignoring_target(&parameters, variant, field)?;
            println!("{}", serde_json::to_string_pretty(&analysis)?);
            Ok(analysis.soundness.check(parameters.security_bits)?)
        }
        Command::Paper(arguments) => paper::paper(arguments),
        Command::Search(arguments) => search::search(arguments, field),
    }
}
