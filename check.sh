#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mode="${1:-full}"
case "$mode" in
  full | quick) ;;
  *) echo "usage: ./check.sh [quick|full]" >&2; exit 2 ;;
esac

step() { printf '\n== %s\n' "$*"; }

step "build"
cargo build --release --locked --bins
step "lint"
target/release/lint
step "fmt"
cargo fmt --check
step "clippy"
cargo clippy --locked --all-targets -- -D warnings
step "sim: carry-less arithmetic"
target/release/sim

if [ "$mode" = quick ]; then
  printf '\ncheck quick: PASS\n'
  exit 0
fi

case "$(uname -m)" in
  arm64 | aarch64) portable_flags="-C target-feature=-aes,-sha3" ;;
  x86_64) portable_flags="-C target-cpu=x86-64" ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac
step "sim: portable arithmetic"
RUSTFLAGS="$portable_flags" CARGO_TARGET_DIR=target/portable cargo build --release --locked --bins
target/portable/release/sim --fresh 200
step "msrv: rust 1.88"
cargo +1.88 check --locked --all-targets

printf '\ncheck full: PASS\n'
