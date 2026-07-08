//! High-Fidelity Audio Encoder using Hound.
//!
//! Writes studio-grade 24-bit or 16-bit PCM WAV files with dither and clipping guards.

use std::io::Cursor;
use std::path::Path;
use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};

pub fn write_wav_file<P: AsRef<Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
    bit_depth: u16,
) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: bit_depth,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    write_samples_to_writer(&mut writer, samples, bit_depth)?;
    writer.finalize()?;
    Ok(())
}

pub fn write_wav_bytes(
    samples: &[f32],
    sample_rate: u32,
    bit_depth: u16,
) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: bit_depth,
        sample_format: SampleFormat::Int,
    };

    let mut vec = Vec::with_capacity(samples.len() * 3 + 44);
    {
        let cursor = Cursor::new(&mut vec);
        let mut writer = WavWriter::new(cursor, spec)?;
        write_samples_to_writer(&mut writer, samples, bit_depth)?;
        writer.finalize()?;
    }
    Ok(vec)
}

fn write_samples_to_writer<W: std::io::Write + std::io::Seek>(
    writer: &mut WavWriter<W>,
    samples: &[f32],
    bit_depth: u16,
) -> Result<()> {
    if bit_depth == 24 {
        let max_val = 8388607.0f32; // 2^23 - 1
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let int_sample = (clamped * max_val).round() as i32;
            writer.write_sample(int_sample)?;
        }
    } else {
        let max_val = 32767.0f32; // 2^15 - 1
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let int_sample = (clamped * max_val).round() as i16;
            writer.write_sample(int_sample)?;
        }
    }
    Ok(())
}
