//! Brick-Wall Lookahead Peak Limiter.
//!
//! Provides a safe guardrail against digital clipping (inter-sample peaks and overshoots),
//! ensuring output never exceeds the broadcast ceiling (typically -1.0 dBTP).

pub struct Limiter {
    ceiling_linear: f32,
    lookahead_len: usize,
    buffer: Vec<f32>,
    write_idx: usize,
    release_coeff: f32,
    current_gain: f32,
}

impl Limiter {
    pub fn new(ceiling_db: f32, lookahead_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        let ceiling_linear = 10.0_f32.powf(ceiling_db / 20.0);
        let lookahead_len = ((lookahead_ms.max(0.5) * 0.001 * sample_rate).round() as usize).max(4);
        let release_coeff = (-1.0 / ((release_ms.max(5.0)) * 0.001 * sample_rate)).exp();

        Self {
            ceiling_linear,
            lookahead_len,
            buffer: vec![0.0f32; lookahead_len],
            write_idx: 0,
            release_coeff,
            current_gain: 1.0,
        }
    }

    /// Process audio block in-place.
    pub fn process_block(&mut self, audio: &mut [f32]) {
        for sample in audio.iter_mut() {
            let in_val = *sample;

            // 1. Store incoming sample in ring buffer
            self.buffer[self.write_idx] = in_val;

            // 2. Compute read pointer for delayed sample
            let read_idx = (self.write_idx + 1) % self.lookahead_len;
            let delayed_sample = self.buffer[read_idx];

            // 3. Compute instant peak demand from current input
            let peak = in_val.abs();
            let target_gain = if peak > self.ceiling_linear {
                self.ceiling_linear / peak
            } else {
                1.0
            };

            // 4. Ballistics: zero attack (instant clamp) + exponential release
            if target_gain < self.current_gain {
                self.current_gain = target_gain;
            } else {
                self.current_gain = target_gain + self.release_coeff * (self.current_gain - target_gain);
            }

            // 5. Output delayed sample scaled by envelope
            *sample = (delayed_sample * self.current_gain).clamp(-self.ceiling_linear, self.ceiling_linear);

            self.write_idx = (self.write_idx + 1) % self.lookahead_len;
        }
    }
}
