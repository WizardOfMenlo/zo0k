use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidParameters(String),
    InvalidInput(&'static str),
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
