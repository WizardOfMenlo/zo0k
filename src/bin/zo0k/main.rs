mod args;
mod bench;
mod paper;
mod search;

use args::{Command, Invocation, Matches};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use zo0k::json::ToJson;
use zo0k::tuning::{Analysis, optimize_queries};
use zo0k::{Blake3, Field, FieldSize, Gf2_128, Gf2_256, Hasher, Keccak256, Parameters, Variant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldChoice {
    Gf2_256,
    Gf2_128,
}

impl FieldChoice {
    const NAMES: [&str; 2] = [Gf2_256::NAME, Gf2_128::NAME];

    fn from_name(name: &str) -> Option<Self> {
        let choice = match name {
            Gf2_256::NAME => Self::Gf2_256,
            Gf2_128::NAME => Self::Gf2_128,
            _ => return None,
        };
        assert!(Self::NAMES.contains(&name));
        Some(choice)
    }

    fn size(self) -> FieldSize {
        match self {
            Self::Gf2_256 => Gf2_256::SIZE,
            Self::Gf2_128 => Gf2_128::SIZE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HashChoice {
    Blake3,
    Keccak256,
}

impl HashChoice {
    const NAMES: [&str; 2] = [Blake3::NAME, Keccak256::NAME];

    fn from_name(name: &str) -> Option<Self> {
        let choice = match name {
            Blake3::NAME => Self::Blake3,
            Keccak256::NAME => Self::Keccak256,
            _ => return None,
        };
        assert!(Self::NAMES.contains(&name));
        Some(choice)
    }
}

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

struct Source {
    params: Option<PathBuf>,
    paper: Option<String>,
    num_variables: Option<usize>,
    security: u32,
    optimize_queries: bool,
}

impl Source {
    fn new(matches: &Matches) -> Self {
        let source = Self {
            params: matches.path("params"),
            paper: matches.text("paper"),
            num_variables: matches.count("num-variables"),
            security: matches.bits("security").expect("a default security target"),
            optimize_queries: matches.flag("optimize-queries"),
        };
        let given = [source.params.is_some(), source.paper.is_some(), source.num_variables.is_some()];
        assert!(given.iter().filter(|&&is_given| is_given).count() <= 1);
        source
    }

    fn parameters(&self, variant: Variant, field: FieldSize) -> Result<Parameters> {
        let mut parameters = match (&self.params, &self.paper, self.num_variables) {
            (Some(path), _, _) => zo0k::json::from_slice(&fs::read(path)?)?,
            (_, Some(row), _) => {
                let selected = paper::table_row(row)?;
                match variant {
                    Variant::ZeroKnowledge => selected.zero_knowledge,
                    Variant::Baseline => selected.baseline,
                }
            }
            (_, _, Some(num_variables)) => {
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
        assert!(self.num_variables.is_none_or(|num_variables| parameters.num_variables == num_variables));
        assert!(self.num_variables.is_none() || parameters.security_bits == self.security);
        Ok(parameters)
    }
}

fn main() {
    let arguments: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let invocation = match args::parse(&arguments) {
        Ok(invocation) => invocation,
        Err(exit) => {
            assert!(exit.code == 0 || exit.code == 2);
            assert!(!exit.text.is_empty());
            if exit.code == 0 {
                print!("{}", exit.text);
            } else {
                eprint!("{}", exit.text);
            }
            std::process::exit(exit.code);
        }
    };
    if let Err(error) = run(&invocation) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(invocation: &Invocation) -> Result<()> {
    if let Some(threads) = invocation.threads {
        if threads > zo0k::parallel::THREADS_MAX {
            return Err(format!("--threads is at most {}", zo0k::parallel::THREADS_MAX).into());
        }
        zo0k::parallel::set_threads(threads);
    }
    let field = invocation.field.size();
    let matches = &invocation.matches;
    assert!(invocation.threads.is_none_or(|threads| threads <= usize::MAX / 2));
    assert!(field.bits == 128 || field.bits == 256);
    match invocation.command {
        Command::Bench => {
            let arguments = bench::Bench::new(matches);
            with_types!(invocation.field, invocation.hash, bench::bench, &arguments)
        }
        Command::Verify => bench::verify(matches.positional.as_ref().expect("a directory")),
        Command::Analyze => {
            let source = Source::new(matches);
            let variant = if matches.flag("baseline") { Variant::Baseline } else { Variant::ZeroKnowledge };
            let parameters = source.parameters(variant, field)?;
            let analysis = Analysis::ignoring_target(&parameters, variant, field)?;
            println!("{}", analysis.to_json().to_string_pretty());
            Ok(analysis.soundness.check(parameters.security_bits)?)
        }
        Command::Paper => paper::paper(&paper::Paper::new(matches)),
        Command::Search => search::search(&search::Search::new(matches), field),
    }
}
