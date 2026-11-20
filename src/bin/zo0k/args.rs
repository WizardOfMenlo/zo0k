use crate::{FieldChoice, HashChoice};
use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Flag,
    Path,
    Text,
    Range { min: usize, max: usize },
    Count,
    Bits { max: u32 },
    Float,
    Field,
    Hash,
}

pub struct Spec {
    pub name: &'static str,
    pub value_name: &'static str,
    pub kind: Kind,
    pub help: &'static str,
    pub default: Option<&'static str>,
    pub required: bool,
}

impl Spec {
    const fn new(name: &'static str, value_name: &'static str, kind: Kind, help: &'static str) -> Self {
        Self { name, value_name, kind, help, default: None, required: false }
    }

    const fn with_default(self, default: &'static str) -> Self {
        Self { default: Some(default), ..self }
    }

    const fn required(self) -> Self {
        Self { required: true, ..self }
    }

    fn display(&self) -> String {
        assert!(!self.name.is_empty());
        if self.kind == Kind::Flag {
            format!("--{}", self.name)
        } else {
            assert!(!self.value_name.is_empty());
            format!("--{} <{}>", self.name, self.value_name)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Typed {
    Flag,
    Path(PathBuf),
    Text(String),
    Count(usize),
    Bits(u32),
    Float(f64),
    Field(FieldChoice),
    Hash(HashChoice),
}

pub const GLOBAL: &[Spec] = &[
    Spec::new("threads", "THREADS", Kind::Count, "Number of worker threads (default: all cores)"),
    Spec::new("field", "FIELD", Kind::Field, "The field of the protocol").with_default("gf2_256"),
    Spec::new(
        "hash",
        "HASH",
        Kind::Hash,
        "The hash function of Merkle trees and of the Fiat-Shamir transformation",
    )
    .with_default("blake3"),
];

const SOURCE: &[Spec] = &[
    Spec::new("params", "PARAMS", Kind::Path, "A JSON file with parameters"),
    Spec::new(
        "paper",
        "ROW",
        Kind::Text,
        "The configuration of a row of the paper's tables, such as `sublinear-16` or `polylog-24`",
    ),
    Spec::new(
        "num-variables",
        "N",
        Kind::Range { min: 1, max: 33 },
        "Standard parameters for tables of 2^N entries",
    ),
    Spec::new(
        "security",
        "SECURITY",
        Kind::Bits { max: u32::MAX },
        "Security target of the standard parameters, in bits",
    )
    .with_default("100"),
    Spec::new("optimize-queries", "", Kind::Flag, "Replace the query counts by optimized ones"),
];

const SOURCE_CONFLICTS: &[(&str, &str)] = &[
    ("params", "paper"),
    ("params", "num-variables"),
    ("paper", "num-variables"),
    ("params", "security"),
    ("paper", "security"),
];

const BENCH: &[Spec] = &[
    Spec::new(
        "repeats",
        "REPEATS",
        Kind::Count,
        "Number of proofs to generate; the report has the median times",
    )
    .with_default("3"),
    Spec::new("out", "OUT", Kind::Path, "Save the statement and the last proof to this new directory"),
];

const ANALYZE: &[Spec] =
    &[Spec::new("baseline", "", Kind::Flag, "Analyze the protocol without zero knowledge instead")];

const PAPER: &[Spec] = &[
    Spec::new(
        "configurations",
        "CONFIGURATIONS",
        Kind::Path,
        "Directory with `tables.json` and `frontiers.json`, the configurations of the paper",
    )
    .with_default("paper"),
    Spec::new("out", "OUT", Kind::Path, "Directory for `scaling-tables.tex` and `frontiers.csv`")
        .with_default("paper"),
];

const SEARCH: &[Spec] = &[
    Spec::new("num-variables", "N", Kind::Range { min: 1, max: 40 }, "Tables have 2^N entries").required(),
    Spec::new(
        "stages",
        "STAGES",
        Kind::Range { min: 1, max: 8 },
        "Number of stages. One gives the sublinear protocol. By default every number up to \
         `--max-stages` is tried",
    ),
    Spec::new(
        "max-stages",
        "MAX_STAGES",
        Kind::Range { min: 1, max: 8 },
        "Largest number of stages that is tried without `--stages`",
    )
    .with_default("6"),
    Spec::new(
        "budget",
        "BUDGET",
        Kind::Float,
        "Limit on the IOP proof length, as a multiple of the size of the table",
    )
    .with_default("5"),
    Spec::new("security", "SECURITY", Kind::Bits { max: u32::MAX }, "Security target in bits")
        .with_default("100"),
    Spec::new(
        "pow-bits",
        "POW_BITS",
        Kind::Bits { max: u32::MAX },
        "Bits of proof of work before every round of queries",
    )
    .with_default("24"),
    Spec::new(
        "max-pow-bits",
        "MAX_POW_BITS",
        Kind::Bits { max: 40 },
        "Limit on the bits of proof of work before each other verifier message. Such a message gets \
         the fewest bits that the security target needs, which is none over a large field",
    )
    .with_default("24"),
    Spec::new(
        "johnson-slack",
        "JOHNSON_SLACK",
        Kind::Float,
        "Distance of the proximity parameters from the Johnson bound. A schedule that needs more than \
         `--max-pow-bits` at this slack gets the smallest larger slack at which it does not",
    )
    .with_default("0.0002"),
    Spec::new("baseline", "", Kind::Flag, "Search for the protocol without zero knowledge"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Bench,
    Verify,
    Analyze,
    Paper,
    Search,
}

pub struct CommandSpec {
    pub command: Command,
    pub name: &'static str,
    pub about: &'static str,
    pub options: &'static [&'static [Spec]],
    pub positional: Option<(&'static str, &'static str)>,
    pub conflicts: &'static [(&'static str, &'static str)],
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        command: Command::Bench,
        name: "bench",
        about: "Commit to a random table, prove an evaluation, verify it, and report times and sizes",
        options: &[SOURCE, BENCH],
        positional: None,
        conflicts: SOURCE_CONFLICTS,
    },
    CommandSpec {
        command: Command::Verify,
        name: "verify",
        about: "Verify a proof saved by `bench --out`",
        options: &[],
        positional: Some(("DIRECTORY", "The directory that holds `statement.json` and `proof.bin`")),
        conflicts: &[],
    },
    CommandSpec {
        command: Command::Analyze,
        name: "analyze",
        about: "Print the layout, soundness errors and sizes of parameters as JSON. Fails after printing if \
                the soundness error is above the target",
        options: &[SOURCE, ANALYZE],
        positional: None,
        conflicts: SOURCE_CONFLICTS,
    },
    CommandSpec {
        command: Command::Paper,
        name: "paper",
        about: "Recompute the tables and the frontier data of the paper's evaluation from its configurations",
        options: &[PAPER],
        positional: None,
        conflicts: &[],
    },
    CommandSpec {
        command: Command::Search,
        name: "search",
        about: "Search for a schedule with a small argument within a limit on the IOP proof length, and print \
                its parameters",
        options: &[SEARCH],
        positional: None,
        conflicts: &[],
    },
];

const ABOUT: &str = "Zero-knowledge IOPP for constrained interleaved Reed-Solomon codes";

const HELP_ABOUT: &str = "Print this message or the help of the given subcommand(s)";

pub struct Exit {
    pub text: String,
    pub code: i32,
}

impl Exit {
    fn usage(message: String) -> Self {
        assert!(!message.is_empty());
        Self { text: format!("error: {message}\n\nFor more information, try '--help'.\n"), code: 2 }
    }

    fn help(text: String) -> Self {
        assert!(text.ends_with('\n'));
        Self { text, code: 0 }
    }
}

pub struct Matches {
    specs: Vec<&'static Spec>,
    values: Vec<(&'static str, Typed)>,
    pub positional: Option<PathBuf>,
}

impl Matches {
    fn new(groups: &[&'static [Spec]]) -> Self {
        let specs: Vec<&'static Spec> = groups.iter().flat_map(|group| group.iter()).collect();
        assert!(!specs.is_empty());
        for (index, spec) in specs.iter().enumerate() {
            assert!(specs[..index].iter().all(|other| other.name != spec.name));
        }
        Self { specs, values: Vec::new(), positional: None }
    }

    fn spec(&self, name: &str) -> Option<&'static Spec> {
        self.specs.iter().copied().find(|spec| spec.name == name)
    }

    pub fn is_present(&self, name: &str) -> bool {
        assert!(self.spec(name).is_some(), "an option of the command");
        self.values.iter().any(|(present, _)| *present == name)
    }

    pub fn explicit(&self, name: &str) -> Option<Typed> {
        assert!(self.spec(name).is_some(), "an option of the command");
        self.values.iter().find(|(present, _)| *present == name).map(|(_, value)| value.clone())
    }

    pub fn value(&self, name: &str) -> Option<Typed> {
        let spec = self.spec(name).expect("an option of the command");
        let default = || spec.default.map(|text| parse_value(spec, text).ok().expect("a valid default"));
        self.explicit(name).or_else(default)
    }

    pub fn flag(&self, name: &str) -> bool {
        assert_eq!(self.spec(name).map(|spec| spec.kind), Some(Kind::Flag));
        self.is_present(name)
    }

    pub fn path(&self, name: &str) -> Option<PathBuf> {
        match self.value(name) {
            Some(Typed::Path(path)) => Some(path),
            None => None,
            Some(other) => unreachable!("{other:?} is not a path"),
        }
    }

    pub fn text(&self, name: &str) -> Option<String> {
        match self.value(name) {
            Some(Typed::Text(text)) => Some(text),
            None => None,
            Some(other) => unreachable!("{other:?} is not a text"),
        }
    }

    pub fn count(&self, name: &str) -> Option<usize> {
        match self.value(name) {
            Some(Typed::Count(count)) => Some(count),
            None => None,
            Some(other) => unreachable!("{other:?} is not a count"),
        }
    }

    pub fn bits(&self, name: &str) -> Option<u32> {
        match self.value(name) {
            Some(Typed::Bits(bits)) => Some(bits),
            None => None,
            Some(other) => unreachable!("{other:?} is not a number of bits"),
        }
    }

    pub fn float(&self, name: &str) -> Option<f64> {
        match self.value(name) {
            Some(Typed::Float(value)) => Some(value),
            None => None,
            Some(other) => unreachable!("{other:?} is not a number"),
        }
    }

    fn take_option(&mut self, arguments: &[String], index: usize) -> Result<usize, Exit> {
        let token = arguments[index].as_str();
        assert!(token.starts_with('-'));
        let (name, inline) = match token.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (token, None),
        };
        let Some(spec) = name.strip_prefix("--").and_then(|name| self.spec(name)) else {
            return Err(Exit::usage(format!("unexpected argument '{token}' found")));
        };
        if self.values.iter().any(|(present, _)| *present == spec.name) {
            return Err(Exit::usage(format!(
                "the argument '{}' cannot be used multiple times",
                spec.display()
            )));
        }
        let mut next = index + 1;
        let typed = if spec.kind == Kind::Flag {
            if let Some(value) = inline {
                let message = format!(
                    "unexpected value '{value}' for '{}' found; no more were expected",
                    spec.display()
                );
                return Err(Exit::usage(message));
            }
            Typed::Flag
        } else {
            let value = match inline {
                Some(value) => value,
                None => match arguments.get(index + 1).filter(|value| !is_option(value)) {
                    Some(value) => {
                        next += 1;
                        value.as_str()
                    }
                    None => return Err(missing_value(spec)),
                },
            };
            parse_value(spec, value)?
        };
        self.values.push((spec.name, typed));
        assert!(next > index && next <= arguments.len());
        Ok(next)
    }
}

fn is_option(token: &str) -> bool {
    token.starts_with('-') && token.len() > 1
}

fn missing_value(spec: &Spec) -> Exit {
    Exit::usage(format!("a value is required for '{}' but none was supplied", spec.display()))
}

fn parse_value(spec: &Spec, text: &str) -> Result<Typed, Exit> {
    assert!(spec.kind != Kind::Flag);
    let invalid = |reason: &dyn std::fmt::Display| {
        Exit::usage(format!("invalid value '{text}' for '{}': {reason}", spec.display()))
    };
    let typed = match spec.kind {
        Kind::Flag => unreachable!("a flag has no value"),
        Kind::Path if text.is_empty() => return Err(missing_value(spec)),
        Kind::Path => Typed::Path(PathBuf::from(text)),
        Kind::Text => Typed::Text(text.to_string()),
        Kind::Range { min, max } => match text.parse::<usize>() {
            Ok(value) if (min..=max).contains(&value) => Typed::Count(value),
            _ => return Err(invalid(&format!("expected an integer between {min} and {max}"))),
        },
        Kind::Count => Typed::Count(text.parse().map_err(|error| invalid(&error))?),
        Kind::Bits { max } => {
            let value: i64 = text.parse().map_err(|error| invalid(&error))?;
            let bits = u32::try_from(value).ok().filter(|bits| *bits <= max);
            Typed::Bits(bits.ok_or_else(|| invalid(&format!("{value} is not in 0..={max}")))?)
        }
        Kind::Float => Typed::Float(text.parse().map_err(|error| invalid(&error))?),
        Kind::Field => match FieldChoice::from_name(text) {
            Some(field) => Typed::Field(field),
            None => return Err(invalid_choice(spec, text, &FieldChoice::NAMES)),
        },
        Kind::Hash => match HashChoice::from_name(text) {
            Some(hash) => Typed::Hash(hash),
            None => return Err(invalid_choice(spec, text, &HashChoice::NAMES)),
        },
    };
    Ok(typed)
}

fn invalid_choice(spec: &Spec, text: &str, names: &[&str]) -> Exit {
    assert!(!names.is_empty());
    let possible = names.join(", ");
    Exit::usage(format!("invalid value '{text}' for '{}'\n  [possible values: {possible}]", spec.display()))
}

pub struct Invocation {
    pub command: Command,
    pub threads: Option<usize>,
    pub field: FieldChoice,
    pub hash: HashChoice,
    pub matches: Matches,
}

pub fn parse(arguments: &[OsString]) -> Result<Invocation, Exit> {
    let mut texts = Vec::with_capacity(arguments.len());
    for argument in arguments {
        let Some(text) = argument.to_str() else {
            return Err(Exit::usage("invalid UTF-8 was detected in one or more arguments".to_string()));
        };
        texts.push(text.to_string());
    }
    assert_eq!(texts.len(), arguments.len());
    let mut top = Matches::new(&[GLOBAL]);
    let mut index = 0;
    while index < texts.len() {
        match texts[index].as_str() {
            "-h" | "--help" => return Err(Exit::help(main_help())),
            "-V" | "--version" => {
                return Err(Exit::help(format!("zo0k {}\n", env!("CARGO_PKG_VERSION"))));
            }
            token if is_option(token) => index = top.take_option(&texts, index)?,
            _ => break,
        }
    }
    let Some(name) = texts.get(index) else {
        return Err(Exit { text: main_help(), code: 2 });
    };
    if name == "help" {
        return Err(help_command(&texts[index + 1..]));
    }
    let Some(spec) = COMMANDS.iter().find(|spec| spec.name == name) else {
        return Err(Exit::usage(format!("unrecognized subcommand '{name}'")));
    };
    let matches = parse_command(spec, &texts[index + 1..])?;
    let global = |name: &str| matches.explicit(name).or_else(|| top.value(name));
    let threads = global("threads").map(|value| match value {
        Typed::Count(threads) => threads,
        other => unreachable!("{other:?} is not a count"),
    });
    let Some(Typed::Field(field)) = global("field") else { unreachable!("a field") };
    let Some(Typed::Hash(hash)) = global("hash") else { unreachable!("a hash") };
    Ok(Invocation { command: spec.command, threads, field, hash, matches })
}

fn parse_command(spec: &CommandSpec, arguments: &[String]) -> Result<Matches, Exit> {
    let mut groups: Vec<&'static [Spec]> = spec.options.to_vec();
    groups.push(GLOBAL);
    let mut matches = Matches::new(&groups);
    assert!(matches.values.is_empty());
    let mut positional_only = false;
    let mut index = 0;
    while index < arguments.len() {
        let token = arguments[index].as_str();
        if !positional_only && token == "--" {
            positional_only = true;
            index += 1;
            continue;
        }
        if !positional_only && (token == "-h" || token == "--help") {
            return Err(Exit::help(command_help(spec)));
        }
        if !positional_only && is_option(token) {
            index = matches.take_option(arguments, index)?;
            continue;
        }
        match spec.positional {
            Some((value_name, _)) if matches.positional.is_none() => {
                if token.is_empty() {
                    return Err(Exit::usage(format!(
                        "a value is required for '<{value_name}>' but none was supplied"
                    )));
                }
                matches.positional = Some(PathBuf::from(token));
            }
            _ => return Err(Exit::usage(format!("unexpected argument '{token}' found"))),
        }
        index += 1;
    }
    check_command(spec, &matches)?;
    assert!(matches.values.len() <= arguments.len());
    Ok(matches)
}

fn check_command(spec: &CommandSpec, matches: &Matches) -> Result<(), Exit> {
    assert!(matches.positional.is_none() || spec.positional.is_some());
    assert!(spec.conflicts.iter().all(|(first, second)| first != second));
    let missing = |argument: String| {
        Exit::usage(format!("the following required arguments were not provided:\n  {argument}"))
    };
    for option in spec.options.iter().flat_map(|group| group.iter()) {
        if option.required && !matches.is_present(option.name) {
            return Err(missing(option.display()));
        }
    }
    if let (Some((value_name, _)), None) = (spec.positional, &matches.positional) {
        return Err(missing(format!("<{value_name}>")));
    }
    for (first, second) in spec.conflicts {
        if matches.is_present(first) && matches.is_present(second) {
            let display = |name: &str| matches.spec(name).expect("an option").display();
            let message =
                format!("the argument '{}' cannot be used with '{}'", display(first), display(second));
            return Err(Exit::usage(message));
        }
    }
    Ok(())
}

fn help_command(arguments: &[String]) -> Exit {
    assert!(COMMANDS.iter().all(|spec| spec.name != "help"));
    match arguments {
        [] => Exit::help(main_help()),
        [name] if name == "help" => Exit::help(format!(
            "{HELP_ABOUT}\n\nUsage: zo0k help [COMMAND]\n\nArguments:\n  [COMMAND]  Print help for the subcommand\n"
        )),
        [name] => match COMMANDS.iter().find(|spec| spec.name == name) {
            Some(spec) => Exit::help(command_help(spec)),
            None => Exit::usage(format!("unrecognized subcommand '{name}'")),
        },
        [name, extra, ..] => {
            let known = COMMANDS.iter().any(|spec| spec.name == name);
            Exit::usage(format!("unrecognized subcommand '{}'", if known { extra } else { name }))
        }
    }
}

fn option_rows(specs: &[&Spec]) -> Vec<(String, String)> {
    let mut rows = Vec::with_capacity(specs.len() + 2);
    for spec in specs {
        let mut help = spec.help.to_string();
        if let Some(default) = spec.default {
            help += &format!(" [default: {default}]");
        }
        match spec.kind {
            Kind::Field => help += &format!(" [possible values: {}]", FieldChoice::NAMES.join(", ")),
            Kind::Hash => help += &format!(" [possible values: {}]", HashChoice::NAMES.join(", ")),
            _ => {}
        }
        rows.push((format!("    {}", spec.display()), help));
    }
    rows.push(("-h, --help".to_string(), "Print help".to_string()));
    assert_eq!(rows.len(), specs.len() + 1);
    rows
}

fn format_rows(title: &str, rows: &[(String, String)]) -> String {
    assert!(!rows.is_empty());
    let width = rows.iter().map(|(left, _)| left.len()).max().unwrap_or(0);
    let mut text = format!("{title}:\n");
    for (left, right) in rows {
        text += &format!("  {left:<width$}  {right}\n");
    }
    assert!(text.len() > title.len());
    text
}

fn main_help() -> String {
    let mut commands: Vec<(String, String)> =
        COMMANDS.iter().map(|spec| (spec.name.to_string(), spec.about.to_string())).collect();
    commands.push(("help".to_string(), HELP_ABOUT.to_string()));
    let mut options = option_rows(&GLOBAL.iter().collect::<Vec<_>>());
    options.push(("-V, --version".to_string(), "Print version".to_string()));
    let text = format!(
        "{ABOUT}\n\nUsage: zo0k [OPTIONS] <COMMAND>\n\n{}\n{}",
        format_rows("Commands", &commands),
        format_rows("Options", &options)
    );
    assert!(text.ends_with('\n'));
    text
}

fn command_help(spec: &CommandSpec) -> String {
    let mut specs: Vec<&Spec> = spec.options.iter().flat_map(|group| group.iter()).collect();
    let mut usage = format!("zo0k {} [OPTIONS]", spec.name);
    for option in specs.iter().filter(|option| option.required) {
        usage += &format!(" {}", option.display());
    }
    let mut text = format!("{}\n\n", spec.about);
    if let Some((value_name, help)) = spec.positional {
        usage += &format!(" <{value_name}>");
        text += &format!("Usage: {usage}\n\n");
        text += &format_rows("Arguments", &[(format!("<{value_name}>"), help.to_string())]);
        text += "\n";
    } else {
        text += &format!("Usage: {usage}\n\n");
    }
    specs.extend(GLOBAL.iter());
    text += &format_rows("Options", &option_rows(&specs));
    assert!(text.ends_with('\n'));
    text
}
