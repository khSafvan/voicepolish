//! Broadcast Feed-Forward Compressor with Soft Knee.
//!
//! Levels out inconsistent microphone distance and vocal dynamics, providing
//! that authoritative, radio/podcast vocal presence without pumping artifacts.

pub struct Compressor {
    threshold_db: f32,
    ratio: f32,
    knee_width_db: f32,
    attack_coeff: f32,
    release_coeff: f32,
    makeup_gain_linear: f32,
    envelope_db: f32,
}

impl Compressor {
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_db: f32,
        sample_rate: f32,
    ) -> Self {
        let attack_coeff = (-1.0 / ((attack_ms.max(0.5)) * 0.001 * sample_rate)).exp();
        let release_coeff = (-1.0 / ((release_ms.max(5.0)) * 0.001 * sample_rate)).exp();
        let makeup_gain_linear = 10.0_f32.powf(makeup_db / 20.0);

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            knee_width_db: 5.0, // 5 dB soft knee
            attack_coeff,
            release_coeff,
            makeup_gain_linear,
            envelope_db: -100.0,
        }
    }

    /// Process audio buffer in-place.
    pub fn process_block(&mut self, audio: &mut [f32]) {
        for sample in audio.iter_mut() {
            let input_abs = sample.abs().max(1e-6);
            let input_db = 20.0 * input_abs.log10();

            // 1. Soft knee gain computer
            let half_knee = self.knee_width_db / 2.0;
            let gain_reduction_db = if input_db < self.threshold_db - half_knee {
                0.0
            } else if input_db > self.threshold_db + half_knee {
                (input_db - self.threshold_db) * (1.0 - 1.0 / self.ratio)
            } else {
                // Quadratic interpolation inside the knee
                let delta = input_db - self.threshold_db + half_knee;
                (delta * delta / (2.0 * self.knee_width_db)) * (1.0 - 1.0 / self.ratio)
            };

            // 2. Ballistics filter (Attack / Release)
            let coeff = if gain_reduction_db > self.envelope_db {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope_db = gain_reduction_db + coeff * (self.envelope_db - gain_reduction_db);

            // 3. Apply gain reduction and makeup
            let reduction_linear = 10.0_f32.powf(-self.envelope_db / 20.0);
            *sample *= reduction_linear * self.makeup_gain_linear;
        }
    }
}
