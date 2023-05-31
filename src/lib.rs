//! Fast and heuristically detects whether content is compressible.
use std::io::{BufReader, Read, Seek, SeekFrom};

use thiserror::Error;

/// Why we assume provided file is uncompressed.
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

struct Samples {
    samples: Vec<u8>,
}

impl Samples {
    const SAMPLING_READ_SIZE: usize = 16;
    const SAMPLING_INTERVAL: usize = 256;
    const MAX_SAMPLE_COUNT: usize = 512;
    fn sample<R: Read + Seek>(content: &mut BufReader<R>) -> std::io::Result<Self> {
        use std::io::ErrorKind;
        let mut samples = Vec::new();
        let mut sample_buffer = [0u8; Self::SAMPLING_READ_SIZE];

        for _ in 0..Self::MAX_SAMPLE_COUNT {
            let result = content.read_exact(&mut sample_buffer);
            match result {
                Ok(()) => {
                    samples.extend_from_slice(&sample_buffer);
                }
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => return Err(e),
            }
            content.seek_relative(Self::SAMPLING_INTERVAL as i64)?;
        }

        debug_assert_eq!(samples.len() % Self::SAMPLING_READ_SIZE, 0);

        Ok(Samples { samples })
    }

    fn check_repeated_bytes(&self) -> bool {
        let len = self.samples.len() / 2;
        &self.samples[..len] == &self.samples[len..len * 2]
    }

    fn word_bucket(&self) -> [usize; 0x100] {
        let mut bucket = [0usize; 0x100];
        for &sample in self.samples.iter() {
            bucket[sample as usize] += 1;
        }
        bucket
    }
}

/// How many byte variants occurred.
fn num_variant(word_bucket: &[usize; 0x100]) -> usize {
    word_bucket.iter().filter(|&&x| x > 0).count()
}

/// How many byte variants make up 90% of the sample.
fn num_core_set(sorted_word_bucket: &[usize; 0x100], sample_size: usize) -> usize {
    let threshold = sample_size * 9 / 10;
    let mut core_set_num = 0;
    for (i, &count) in sorted_word_bucket.iter().enumerate() {
        if core_set_num >= threshold {
            return i;
        }
        core_set_num += count;
    }
    0x100
}

fn log2_pow4(x: usize) -> u32 {
    fn log2(x: usize) -> u32 {
        usize::BITS.wrapping_sub(x.leading_zeros()).wrapping_sub(1)
    }

    log2(x * x * x * x)
}

fn shannon_entropy(sorted_word_bucket: &[usize; 0x100], sample_size: usize) -> usize {
    let entropy_max = 8 * log2_pow4(2);
    let mut entropy_sum = 0;
    let base = log2_pow4(sample_size);
    for &count in sorted_word_bucket {
        // We can early exit here since word bucket is reversely sorted.
        if count == 0 {
            break;
        }
        let logp = log2_pow4(count);
        entropy_sum += count * (base - logp) as usize;
    }
    entropy_sum /= sample_size;
    entropy_sum * 100 / entropy_max as usize
}

pub fn compressed<R: Read + Seek>(content: &mut BufReader<R>) -> Result<(), UncompressedError> {
    const WORD_VARIANT_THRESHOLD: usize = 64;
    const BYTE_CORE_SET_LOW: usize = 64;
    const BYTE_CORE_SET_HIGH: usize = 200;
    const ENTROPY_LVL_HIGH: usize = 80;

    let samples = Samples::sample(content)?;

    if samples.check_repeated_bytes() {
        return Err(UncompressedError::RepeatedBytes);
    }

    let word_bucket = samples.word_bucket();

    let num_variants = num_variant(&word_bucket);
    if num_variants < WORD_VARIANT_THRESHOLD {
        return Err(UncompressedError::LittleVariant(num_variants));
    }

    let mut sorted_word_bucket = word_bucket;
    sorted_word_bucket.sort_by(|a, b| b.cmp(a));
    let sample_size = samples.samples.len();

    let num_core_set = num_core_set(&sorted_word_bucket, sample_size);
    if num_core_set <= BYTE_CORE_SET_LOW {
        return Err(UncompressedError::SmallCoreSet(num_core_set));
    }
    if num_core_set >= BYTE_CORE_SET_HIGH {
        return Ok(());
    }

    let shannon_entropy = shannon_entropy(&sorted_word_bucket, sample_size);
    if shannon_entropy <= ENTROPY_LVL_HIGH {
        return Err(UncompressedError::LowEntropy(shannon_entropy));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    #[test]
    fn test_shannon_entropy() {
        let mut word_bucket = [0; 0x100];
        word_bucket[0x30] = 233;
        word_bucket[0x31] = 234;
        word_bucket[0x32] = 235;

        let mut sorted_word_bucket = word_bucket;
        sorted_word_bucket.sort_by(|a, b| b.cmp(a));

        let entropy = shannon_entropy(&sorted_word_bucket, 233 + 234 + 235);
        assert_eq!(entropy, 18);
    }

    #[test]
    fn test_repeated_bytes() {
        let zeroed = vec![0u8; 0x10000];
        let zeroed = Cursor::new(zeroed);
        let mut reader = BufReader::new(zeroed);
        assert!(matches!(
            compressed(&mut reader),
            Err(UncompressedError::RepeatedBytes)
        ));

        let repeated = vec![2u8; 0x10000];
        let repeated = Cursor::new(repeated);
        let mut reader = BufReader::new(repeated);
        assert!(matches!(
            compressed(&mut reader),
            Err(UncompressedError::RepeatedBytes)
        ));

        let repeated: Vec<u8> = (0..0x10000).map(|_| [32, 42]).flatten().collect();
        let repeated = Cursor::new(repeated);
        let mut reader = BufReader::new(repeated);
        assert!(matches!(
            compressed(&mut reader),
            Err(UncompressedError::RepeatedBytes)
        ));
    }

    #[test]
    fn test_little_variant() {
        let ascii_text: Vec<u8> = (0..0x10000)
            .map(|_| rand::random::<u8>() % (0x7E - 0x40) + 0x40)
            .collect();
        let mut reader = BufReader::new(Cursor::new(ascii_text));
        match compressed(&mut reader) {
            Err(UncompressedError::LittleVariant(x)) => {
                assert_eq!(x, 0x7E - 0x40);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_small_core_set() {
        let mostly_ascii_text: Vec<u8> = (0..280)
            .map(|_| rand::random::<u8>() % (0x7E - 0x40) + 0x40)
            .chain((0..28).map(|_| rand::random::<u8>()))
            .collect();
        let mostly_ascii_text = mostly_ascii_text.repeat(1000);
        let mut reader = BufReader::new(Cursor::new(mostly_ascii_text));
        match compressed(&mut reader) {
            Err(UncompressedError::SmallCoreSet(x)) => {
                assert!(x < 0x7E - 0x40);
            }
            x => panic!("{:?}", x),
        }
    }

    #[test]
    fn test_shannon_entropy_low() {
        let mostly_ascii_text: Vec<u8> = (0..50)
            .map(|_| rand::random::<u8>() % (0x7F - 0x70) + 0x70)
            .chain((0..100).map(|_| rand::random::<u8>() % (0x7F - 0x20) + 0x20))
            .chain((0..28).map(|_| rand::random::<u8>()))
            .collect();
        let mostly_ascii_text = mostly_ascii_text.repeat(1000);
        let mut reader = BufReader::new(Cursor::new(mostly_ascii_text));
        match compressed(&mut reader) {
            Err(UncompressedError::LowEntropy(_)) => {}
            x => panic!("{:?}", x),
        }
    }

    #[test]
    fn test_random() {
        let mostly_ascii_text: Vec<u8> = (0..0x10000).map(|_| rand::random::<u8>()).collect();
        let mut reader = BufReader::new(Cursor::new(mostly_ascii_text));
        assert!(matches!(compressed(&mut reader), Ok(())));
    }
}
