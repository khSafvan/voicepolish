//! Dynamic Split-Band De-Esser.
//!
//! Tames harsh vocal sibilants ("s", "sh", "ch", "z") in the 4.5 kHz – 8.5 kHz range
//! using sidechain band detection and dynamic attenuation, preventing ear fatigue
//! while preserving natural speech clarity.

use crate::dsp::biquad::Biquad;

pub struct DeEsser {
    detector_filter: Biquad,
    notch_filter: Biquad,
    attack_coeff: f32,
    release_coeff: f32,
    threshold_db: f32,
    envelope: f32,
}

impl DeEsser {
    pub fn new(sample_rate: f32, threshold_db: f32, max_reduction_db: f32) -> Self {
        // Sibilance center frequency: 6.0 kHz
        let center_freq = 6000.0f32;
        let q = 1.6f32;

        let detector_filter = Biquad::bandpass(center_freq, q, sample_rate);
        let notch_filter = Biquad::peak_eq(center_freq, -max_reduction_db, q, sample_rate);

        // Fast attack to catch explosive sibilants, smooth release
        let attack_ms = 1.0f32;
        let release_ms = 40.0f32;

        let attack_coeff = (-1.0 / (attack_ms * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / (release_ms * 0.001 * sample_rate)).exp();

        Self {
            detector_filter,
            notch_filter,
            attack_coeff,
            release_coeff,
            threshold_db,
            envelope: 0.0,
        }
    }

    /// Process a block of audio samples in-place.
    pub fn process_block(&mut self, audio: &mut [f32]) {
        let threshold_linear = 10.0_f32.powf(self.threshold_db / 20.0);

        for sample in audio.iter_mut() {
            let x = *sample;

            // 1. Detect energy in sibilant band
            let sibilant = self.detector_filter.process_sample(x);
            let sibilant_level = sibilant.abs();

            // 2. Track detector envelope
            let coeff = if sibilant_level > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = sibilant_level + coeff * (self.envelope - sibilant_level);

            // 3. Compute dynamic attenuation factor
            let notch_sample = self.notch_filter.process_sample(x);

            if self.envelope > threshold_linear {
                let over = (self.envelope / threshold_linear).min(4.0);
                let blend = ((over - 1.0) / 3.0).clamp(0.0, 1.0);
                // Dynamic blend between original and notched sample
                *sample = (1.0 - blend) * x + blend * notch_sample;
            } else {
                *sample = x;
            }
        }
    }
}
