//! Errors that cross layer boundaries.

use std::fmt;

/// Why a request could not be served.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A downstream request the proxy will not serve: a malformed body, or a
    /// well-formed one asking for something the upstream cannot do. Surfaces
    /// as the dialect's own 400 envelope.
    Request(String),
    /// A provider failure safe to expose through a downstream error envelope.
    /// `account_unavailable` marks a failure of the credentials rather than
    /// of the request, so the account pool may retry it on another login.
    Provider {
        status: u16,
        message: String,
        account_unavailable: bool,
    },
    /// An upstream stream or value that was not what its protocol promises.
    /// Answered as 502.
    Upstream(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn request(message: impl Into<String>) -> Self {
        Error::Request(message.into())
    }

    pub fn provider(status: u16, message: impl Into<String>) -> Self {
        Error::Provider {
            status,
            message: message.into(),
            account_unavailable: false,
        }
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        Error::Upstream(message.into())
    }

    /// The HTTP status the client is answered with.
    pub fn status(&self) -> u16 {
        match self {
            Error::Request(_) => 400,
            Error::Provider { status, .. } => *status,
            Error::Upstream(_) => 502,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Error::Request(message)
            | Error::Upstream(message)
            | Error::Provider { message, .. } => message,
        }
    }

    pub fn account_unavailable(&self) -> bool {
        matches!(
            self,
            Error::Provider {
                account_unavailable: true,
                ..
            }
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}
