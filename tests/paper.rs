//! The pinned configurations of the paper's evaluation reproduce its tables
//! and frontier data.

use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn the_paper_command_reproduces_the_committed_tables_and_frontiers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = std::env::temp_dir().join(format!("zo0k-paper-test-{}", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_zo0k"))
        .arg("paper")
        .arg("--configurations")
        .arg(root.join("paper"))
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let read = |directory: &Path, file: &str| fs::read_to_string(directory.join(file)).unwrap();

    let file = "scaling-tables.tex";
    assert!(read(&out, file) == read(&root.join("paper"), file), "paper/{file} is out of date");

    // Expected sizes go through elementary functions, whose last bits depend
    // on the platform, so they are compared up to the last printed digit.
    let file = "frontiers.csv";
    let (actual, expected) = (read(&out, file), read(&root.join("paper"), file));
    assert_eq!(actual.lines().count(), expected.lines().count());
    for (actual, expected) in actual.lines().zip(expected.lines()).skip(1) {
        let (actual, expected): (Vec<&str>, Vec<&str>) =
            (actual.split(',').collect(), expected.split(',').collect());
        assert_eq!((&actual[..3], actual[4]), (&expected[..3], expected[4]));
        let size = |fields: &[&str]| fields[3].parse::<f64>().unwrap();
        assert!((size(&actual) - size(&expected)).abs() < 0.002, "{actual:?} and {expected:?}");
    }
    fs::remove_dir_all(out).unwrap();
}
