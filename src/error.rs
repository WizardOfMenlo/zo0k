//! Errors of the library.

use std::fmt;

/// The result of an operation of the library.
pub type Result<T> = std::result::Result<T, Error>;

/// Why an operation of the library failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The parameters do not describe a protocol, or its analyzed soundness
    /// error is above the target.
    InvalidParameters(String),
    /// An input of the prover or the verifier does not fit the parameters.
    InvalidInput(&'static str),
    /// The verifier rejects the proof.
    InvalidProof(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters(reason) => write!(f, "invalid parameters: {reason}"),
            Self::InvalidInput(reason) => write!(f, "invalid input: {reason}"),
            Self::InvalidProof(reason) => write!(f, "invalid proof: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
