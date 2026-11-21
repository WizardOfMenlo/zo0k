use crate::Result;
use crate::args::Matches;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;
use zo0k::costs::Costs;
use zo0k::json::{self, FromJson, Shape, Unknown, Value, required};
use zo0k::tuning::Analysis;
use zo0k::{Field, FieldSize, Gf2_256, Layout, Parameters, Variant};

const FIELD: FieldSize = Gf2_256::SIZE;

pub struct Paper {
    configurations: PathBuf,
    out: PathBuf,
}

impl Paper {
    pub fn new(matches: &Matches) -> Self {
        let paper = Self {
            configurations: matches.path("configurations").expect("a default directory"),
            out: matches.path("out").expect("a default directory"),
        };
        assert!(!paper.configurations.as_os_str().is_empty());
        assert!(!paper.out.as_os_str().is_empty());
        paper
    }
}

const TABLES: &str = include_str!("../../../paper/tables.json");

pub fn table_row(name: &str) -> Result<Pair<Parameters>> {
    let rows: Vec<Row> = json::from_slice(TABLES.as_bytes())?;
    let names: Vec<String> =
        rows.iter().map(|row| format!("{}-{}", row.protocol, row.num_variables)).collect();
    let index = names
        .iter()
        .position(|row| row == name)
        .ok_or_else(|| format!("no table row `{name}`; the rows are {}", names.join(", ")))?;
    Ok(rows.into_iter().nth(index).unwrap().selected)
}

struct Row {
    protocol: String,
    num_variables: usize,
    selected: Pair<Parameters>,
}

const ROW_FIELDS: [&str; 4] = ["protocol", "num_variables", "zero_knowledge", "baseline"];

impl FromJson for Row {
    fn from_json(value: &Value) -> json::Result<Self> {
        let [protocol, num_variables, zero_knowledge, baseline] =
            json::fields(value, ROW_FIELDS, Unknown::Ignore, Shape::Object)?;
        let protocol = required(protocol, ROW_FIELDS[0])?;
        let num_variables = required(num_variables, ROW_FIELDS[1])?;
        let selected = Pair {
            zero_knowledge: required(zero_knowledge, ROW_FIELDS[2])?,
            baseline: required(baseline, ROW_FIELDS[3])?,
        };
        Ok(Self { protocol, num_variables, selected })
    }
}

pub struct Pair<T> {
    pub zero_knowledge: T,
    pub baseline: T,
}

const PAIR_FIELDS: [&str; 2] = ["zero_knowledge", "baseline"];

impl<T: FromJson> FromJson for Pair<T> {
    fn from_json(value: &Value) -> json::Result<Self> {
        let [zero_knowledge, baseline] =
            json::fields(value, PAIR_FIELDS, Unknown::Ignore, Shape::ObjectOrArray)?;
        let zero_knowledge = required(zero_knowledge, PAIR_FIELDS[0])?;
        let baseline = required(baseline, PAIR_FIELDS[1])?;
        Ok(Self { zero_knowledge, baseline })
    }
}

struct Sizes {
    num_variables: usize,
    zero_knowledge: Costs,
    baseline: Costs,
    baseline_iop_same_schedule: usize,
}

impl Sizes {
    fn new(row: &Row) -> Result<Self> {
        let zero_knowledge =
            Analysis::new(&row.selected.zero_knowledge, Variant::ZeroKnowledge, FIELD)?.costs;
        let baseline = Analysis::new(&row.selected.baseline, Variant::Baseline, FIELD)?.costs;
        let same_schedule = Layout::new(&row.selected.zero_knowledge, Variant::Baseline)?;
        let baseline_iop_same_schedule =
            Costs::new(&same_schedule, &row.selected.zero_knowledge, FIELD.bytes).iop_bytes;
        let budget = (5 * FIELD.bytes) << row.num_variables;
        if zero_knowledge.iop_bytes > budget || baseline.iop_bytes > budget {
            return Err(format!("a configuration for 2^{} is above the budget", row.num_variables).into());
        }
        Ok(Self { num_variables: row.num_variables, zero_knowledge, baseline, baseline_iop_same_schedule })
    }

    fn iop_overhead_percent(&self) -> f64 {
        100.0 * (self.zero_knowledge.iop_bytes as f64 / self.baseline_iop_same_schedule as f64 - 1.0)
    }

    fn argument_overhead_percent(&self) -> f64 {
        100.0 * (self.zero_knowledge.expected_argument_bytes / self.baseline.expected_argument_bytes - 1.0)
    }
}

const PROTOCOLS: [(&str, &str, &str, f64); 2] = [
    ("sublinear", "Sublinear protocol", "MiB", (1u64 << 20) as f64),
    ("polylog", "Polylog protocol", "GiB", (1u64 << 30) as f64),
];

const CAPTION: &str = "under a $5\\times$ input IOP budget. Argument overhead is relative to the best \
non-ZK configuration under the same budget; the non-ZK IOP proof length and the IOP overhead are for the \
non-ZK protocol on the same schedule.";

fn tables(rows: &[(String, Sizes)]) -> String {
    const SETUP: &str =
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{7pt}\n\\renewcommand{\\arraystretch}{1.05}\n";
    let mut tex = String::from("\\begin{table}[p]\n");
    for (index, (protocol, title, unit, unit_bytes)) in PROTOCOLS.into_iter().enumerate() {
        if index > 0 {
            tex += "\\par\\medskip\n";
        }
        tex += SETUP;
        tex += "\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n";
        writeln!(
            tex,
            "$\\log_2 n$ & \\multicolumn{{3}}{{c}}{{IOP length ({unit})}} & \
             \\multicolumn{{3}}{{c}}{{Argument size (KiB)}} \\\\"
        )
        .unwrap();
        tex += "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n";
        tex += " & Non-ZK & ZK & Overhead & Non-ZK & ZK & Overhead \\\\\n\\midrule\n";
        for (_, sizes) in rows.iter().filter(|(row_protocol, _)| row_protocol == protocol) {
            writeln!(
                tex,
                "{} & {:.2} & {:.2} & {:.1}\\% & {:.2} & {:.2} & {:.1}\\% \\\\",
                sizes.num_variables,
                sizes.baseline_iop_same_schedule as f64 / unit_bytes,
                sizes.zero_knowledge.iop_bytes as f64 / unit_bytes,
                sizes.iop_overhead_percent(),
                sizes.baseline.expected_argument_bytes / 1024.0,
                sizes.zero_knowledge.expected_argument_bytes / 1024.0,
                sizes.argument_overhead_percent(),
            )
            .unwrap();
        }
        tex += "\\bottomrule\n\\end{tabular}\n";
        writeln!(tex, "\\caption{{{title} {CAPTION}}}\n\\label{{tab:implementation-{protocol}}}").unwrap();
    }
    tex + "\\end{table}\n"
}

fn frontier(points: &mut Vec<(usize, f64, bool)>) {
    points.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut smallest = f64::INFINITY;
    points.retain(|&(_, argument_bytes, _)| {
        let keep = argument_bytes < smallest;
        smallest = smallest.min(argument_bytes);
        keep
    });
}

pub fn paper(arguments: &Paper) -> Result<()> {
    let read = |file: &str| fs::read(arguments.configurations.join(file));
    let table_rows: Vec<Row> = json::from_slice(&read("tables.json")?)?;
    let frontiers: BTreeMap<String, Pair<Vec<Parameters>>> = json::from_slice(&read("frontiers.json")?)?;
    let rows = table_rows
        .iter()
        .map(|row| Ok((row.protocol.clone(), Sizes::new(row)?)))
        .collect::<Result<Vec<_>>>()?;

    let mut csv = String::from("protocol,variant,iop_bytes,expected_argument_bytes,selected\n");
    for (protocol, ..) in PROTOCOLS {
        let candidates = frontiers.get(protocol).ok_or("a frontier is missing")?;
        let (_, selected) =
            rows.iter().rfind(|(row_protocol, _)| row_protocol == protocol).ok_or("no row")?;
        for (variant, name, candidates, selected) in [
            (Variant::Baseline, "baseline", &candidates.baseline, &selected.baseline),
            (Variant::ZeroKnowledge, "zero_knowledge", &candidates.zero_knowledge, &selected.zero_knowledge),
        ] {
            let mut points = vec![(selected.iop_bytes, selected.expected_argument_bytes, true)];
            for parameters in candidates {
                let costs = Analysis::new(parameters, variant, FIELD)?.costs;
                let is_selected =
                    (costs.iop_bytes, costs.expected_argument_bytes) == (points[0].0, points[0].1);
                if !is_selected {
                    points.push((costs.iop_bytes, costs.expected_argument_bytes, false));
                }
            }
            frontier(&mut points);
            if !points.iter().any(|point| point.2) {
                return Err(
                    format!("the selected {name} {protocol} configuration is not on its frontier").into()
                );
            }
            for (iop_bytes, argument_bytes, selected) in points {
                writeln!(csv, "{protocol},{name},{iop_bytes},{argument_bytes:.3},{selected}").unwrap();
            }
        }
    }
    fs::create_dir_all(&arguments.out)?;
    fs::write(arguments.out.join("scaling-tables.tex"), tables(&rows))?;
    fs::write(arguments.out.join("frontiers.csv"), csv)?;

    println!(
        "{:<10} {:>2}  {:>13} {:>8}  {:>12} {:>8}",
        "protocol", "n", "IOP bytes", "overhead", "argument KiB", "overhead"
    );
    for (protocol, sizes) in &rows {
        println!(
            "{protocol:<10} {:>2}  {:>13} {:>7.2}%  {:>12.2} {:>7.2}%",
            sizes.num_variables,
            sizes.zero_knowledge.iop_bytes,
            sizes.iop_overhead_percent(),
            sizes.zero_knowledge.expected_argument_bytes / 1024.0,
            sizes.argument_overhead_percent(),
        );
    }
    Ok(())
}
