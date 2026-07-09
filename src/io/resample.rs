//! High-Quality Band-Limited Sinc Resampling using Rubato.
//!
//! Converts arbitrary microphone recording sample rates (16kHz, 44.1kHz, 96kHz)
//! to the 48kHz broadcast studio standard with zero aliasing.

use anyhow::Result;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub fn resample_to_target(
    input: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>> {
    if source_rate == target_rate || input.is_empty() {
        return Ok(input.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024;
    let mut resampler = SincFixedIn::<f32>::new(
        target_rate as f64 / source_rate as f64,
        2.0,
        params,
        chunk_size,
        1, // 1 channel (mono)
    )?;

    let mut output = Vec::with_capacity(
        ((input.len() as f64) * (target_rate as f64 / source_rate as f64)).ceil() as usize,
    );

    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let mut chunk = input[pos..end].to_vec();
        // Pad last chunk with zeros if smaller than chunk_size
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let waves_in = vec![chunk];
        let waves_out = resampler.process(&waves_in, None)?;

        if let Some(out_chan) = waves_out.get(0) {
            output.extend_from_slice(out_chan);
        }

        pos += chunk_size;
    }

    // Trim output to accurate length
    let expected_len = ((input.len() as f64) * (target_rate as f64 / source_rate as f64)).round() as usize;
    if output.len() > expected_len {
        output.truncate(expected_len);
    }

    Ok(output)
}
