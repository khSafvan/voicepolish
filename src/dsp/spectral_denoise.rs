//! Fast Multi-Band Spectral Denoising & Hum Removal.
//!
//! Implements spectral subtraction with continuous noise floor tracking,
//! over-subtraction factor, and musical noise suppression via spectral floor bounds.
//! Also features notch filters for 50Hz/60Hz mains hum and harmonics.

use std::sync::Arc;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use rustfft::num_complex::Complex;

pub struct SpectralDenoiser {
    fft_size: usize,
    hop_size: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    sample_rate: f32,
    noise_reduction_db: f32,
}

impl SpectralDenoiser {
    pub fn new(sample_rate: f32, noise_reduction_db: f32) -> Self {
        let fft_size = 2048; // ~42.6ms at 48kHz (optimal frequency resolution for speech & hum)
        let hop_size = fft_size / 4; // 75% overlap for artifact-free reconstruction

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let c2r = planner.plan_fft_inverse(fft_size);

        // Periodic Hann window
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
            sample_rate,
            noise_reduction_db: noise_reduction_db.clamp(3.0, 35.0),
        }
    }

    /// Processes an input audio buffer, removing background noise and hum.
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if input.len() < self.fft_size {
            return input.to_vec();
        }

        let num_frames = (input.len() - self.fft_size) / self.hop_size + 1;
        let mut output = vec![0.0f32; input.len() + self.fft_size];
        let mut window_weights = vec![0.0f32; input.len() + self.fft_size];

        let num_bins = self.fft_size / 2 + 1;
        let mut noise_floor = vec![1e-5f32; num_bins];

        // 1. First pass: estimate initial noise floor using the quietest 10% frames
        let mut frame_energies = Vec::with_capacity(num_frames);
        for f in 0..num_frames {
            let start = f * self.hop_size;
            let end = start + self.fft_size;
            let energy: f32 = input[start..end].iter().map(|&x| x * x).sum();
            frame_energies.push((energy, f));
        }
        frame_energies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let num_noise_frames = (num_frames / 10).clamp(5, 50).min(num_frames);
        let mut noise_accum = vec![0.0f32; num_bins];

        let mut in_scratch = vec![0.0f32; self.fft_size];
        let mut spec_scratch = vec![Complex::new(0.0f32, 0.0f32); num_bins];

        for i in 0..num_noise_frames {
            let f = frame_energies[i].1;
            let start = f * self.hop_size;
            for j in 0..self.fft_size {
                in_scratch[j] = input[start + j] * self.window[j];
            }
            if self.r2c.process(&mut in_scratch, &mut spec_scratch).is_ok() {
                for (k, c) in spec_scratch.iter().enumerate() {
                    noise_accum[k] += c.norm();
                }
            }
        }

        for k in 0..num_bins {
            noise_floor[k] = (noise_accum[k] / num_noise_frames as f32).max(1e-6);
        }

        // Notch hum frequencies (50Hz / 60Hz and first 3 harmonics)
        let bin_width = self.sample_rate / (self.fft_size as f32);
        let hum_freqs = [50.0, 60.0, 100.0, 120.0, 150.0, 180.0];
        let mut is_hum_bin = vec![false; num_bins];
        for &hf in &hum_freqs {
            let target_bin = (hf / bin_width).round() as usize;
            for b in target_bin.saturating_sub(1)..=(target_bin + 1).min(num_bins - 1) {
                is_hum_bin[b] = true;
            }
        }

        // Floor gain: controls musical noise vs noise reduction depth
        let min_gain = 10.0_f32.powf(-self.noise_reduction_db / 20.0);
        let oversubtraction = 1.35f32;

        // 2. Second pass: perform spectral subtraction & Wiener smoothing
        let mut out_scratch = vec![0.0f32; self.fft_size];
        let mut prev_gain = vec![1.0f32; num_bins];

        for f in 0..num_frames {
            let start = f * self.hop_size;
            for j in 0..self.fft_size {
                in_scratch[j] = input[start + j] * self.window[j];
            }

            if self.r2c.process(&mut in_scratch, &mut spec_scratch).is_err() {
                continue;
            }

            // Continuous slow tracking of noise floor during quiet moments
            let current_energy: f32 = in_scratch.iter().map(|&x| x * x).sum();
            let is_speech_quiet = current_energy < frame_energies[num_noise_frames].0 * 2.0;

            for k in 0..num_bins {
                let mag = spec_scratch[k].norm();
                if is_speech_quiet {
                    // Slow exponential moving average
                    noise_floor[k] = 0.98 * noise_floor[k] + 0.02 * mag;
                }

                // Spectral subtraction with over-subtraction
                let noise_mag = noise_floor[k] * oversubtraction;
                let mut gain = if mag > 1e-7 {
                    ((mag - noise_mag) / mag).max(min_gain)
                } else {
                    min_gain
                };

                // Extra attenuation on detected mains hum bins
                if is_hum_bin[k] && gain < 0.6 {
                    gain *= 0.2;
                }

                // Temporal smoothing (Decision-Directed approach to eliminate musical noise)
                let smoothed_gain = 0.65 * prev_gain[k] + 0.35 * gain;
                prev_gain[k] = smoothed_gain;

                spec_scratch[k] *= smoothed_gain;
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

        // 3. Normalize overlap-add
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
