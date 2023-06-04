use thiserror::Error;

/// Why provided file is assumed to be uncompressed.
#[derive(Error, Debug)]
pub enum UncompressedError {
    #[error("IoError({0:?})")]
    IoError(#[from] std::io::Error),
    /// Repeated bytes(usually zero-filled).
    #[error("Repeated bytes")]
    RepeatedBytes,
    /// This file has little variants(mostly ascii texts)
    #[error("Little byte variants: {0}")]
    LittleVariant(usize),
    /// A small set of bytes make up 90% of the sample.
    #[error("Small byte core set: {0}")]
    SmallCoreSet(usize),
    /// Low shannon entropy.
    #[error("Low shannon entropy: {0}")]
    LowEntropy(usize),
}
