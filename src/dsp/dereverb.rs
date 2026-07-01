//! Room Reverberation Suppression (Dereverberation).
//!
//! Suppresses acoustic room reflections and decay tails using statistical
//! multi-band spectral decay modeling, transforming hollow, boxy room recordings
//! into tight, intimate studio speech.

use std::sync::Arc;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use rustfft::num_complex::Complex;

pub struct Dereverberator {
    fft_size: usize,
    hop_size: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    strength: f32,
}

impl Dereverberator {
    pub fn new(strength: f32) -> Self {
        let fft_size = 1024; // ~21.3ms at 48kHz (optimal temporal resolution for reverb tails)
        let hop_size = fft_size / 4; // 75% overlap

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);

        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / (fft_size as f32)).cos())
            })
            .collect();

        Self {
            fft_size,
            hop_size,
            r2c,
            c2r,
            window,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Processes an audio buffer to attenuate room reverb and echo tails.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if self.strength <= 0.01 || input.len() < self.fft_size {
            return input.to_vec();
        }

        let num_frames = (input.len() - self.fft_size) / self.hop_size + 1;
        let num_bins = self.fft_size / 2 + 1;

        let mut output = vec![0.0f32; input.len() + self.fft_size];
        let mut window_weights = vec![0.0f32; input.len() + self.fft_size];

        let mut in_scratch = vec![0.0f32; self.fft_size];
        let mut out_scratch = vec![0.0f32; self.fft_size];
        let mut spec_scratch = vec![Complex::new(0.0f32, 0.0f32); num_bins];

        // Reverberation decay model: decay factor alpha per frequency band
        // High frequencies decay faster in air, low/mid frequencies linger longer
        let mut decay_tail = vec![0.0f32; num_bins];
        let base_decay = (0.75 + 0.15 * (1.0 - self.strength)).clamp(0.65, 0.92);

        let suppression_gain = self.strength * 0.7;

        for f in 0..num_frames {
            let start = f * self.hop_size;
            for j in 0..self.fft_size {
                in_scratch[j] = input[start + j] * self.window[j];
            }

            if self.r2c.process(&mut in_scratch, &mut spec_scratch).is_err() {
                continue;
            }

            for k in 0..num_bins {
                let current_mag = spec_scratch[k].norm();

                // If current energy is lower than the predicted reverb tail,
                // it is likely lingering late reverberation from previous phonemes
                let estimated_tail = decay_tail[k];
                let is_tail = current_mag < estimated_tail * 1.5;

                let gain = if is_tail && current_mag > 1e-7 {
                    let excess = (estimated_tail - current_mag).max(0.0);
                    let suppression = (1.0 - suppression_gain * (excess / (estimated_tail + 1e-6))).clamp(0.2, 1.0);
                    suppression
                } else {
                    1.0
                };

                // Update decay state for next frame
                decay_tail[k] = (current_mag * 0.8).max(estimated_tail * base_decay);

                spec_scratch[k] *= gain;
            }

            if self.c2r.process(&mut spec_scratch, &mut out_scratch).is_ok() {
                let norm = 1.0 / (self.fft_size as f32);
                for j in 0..self.fft_size {
                    let w = self.window[j];
                    output[start + j] += out_scratch[j] * norm * w;
                    window_weights[start + j] += w * w;
                }
            }
        }

        let mut final_output = vec![0.0f32; input.len()];
        for i in 0..input.len() {
            let w = window_weights[i];
            if w > 1e-4 {
                final_output[i] = output[i] / w;
            } else {
                final_output[i] = input[i];
            }
        }

        final_output
    }
}
