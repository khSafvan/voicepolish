//! Direct Form II Transposed Biquad Filter Implementation.
//!
//! Provides numerically stable second-order IIR filtering for:
//! - Butterworth High-Pass and Low-Pass filters
//! - Parametric Peaking EQ filters (presence/cuts)
//! - Low-Shelf (warmth) and High-Shelf (air) filters
//! - Band-Pass filters (for frequency-band detection/de-essing)

use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
    // State variables (Direct Form II Transposed)
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Creates a raw biquad with normalized coefficients (a0 is normalized to 1.0).
    pub fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Resets the internal delay state to prevent pop/click artifacts.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Butterworth High-Pass Filter (12 dB/octave slope per biquad).
    pub fn highpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let f = freq_hz.clamp(10.0, sample_rate * 0.49);
        let q = q.max(0.1);
        let w0 = 2.0 * PI * f / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;

        Self::new(
            ((1.0 + cos_w0) / 2.0) / a0,
            (-(1.0 + cos_w0)) / a0,
            ((1.0 + cos_w0) / 2.0) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }

    /// Butterworth Low-Pass Filter (12 dB/octave slope per biquad).
    #[allow(dead_code)]
    pub fn lowpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let f = freq_hz.clamp(10.0, sample_rate * 0.49);
        let q = q.max(0.1);
        let w0 = 2.0 * PI * f / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;

        Self::new(
            ((1.0 - cos_w0) / 2.0) / a0,
            (1.0 - cos_w0) / a0,
            ((1.0 - cos_w0) / 2.0) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }

    /// Parametric Peaking / Bell EQ filter.
    pub fn peak_eq(freq_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let f = freq_hz.clamp(10.0, sample_rate * 0.49);
        let q = q.max(0.1);
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha / a;

        Self::new(
            (1.0 + alpha * a) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha * a) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha / a) / a0,
        )
    }

    /// Low-Shelf Filter (boost/cut frequencies below `freq_hz`).
    pub fn low_shelf(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let f = freq_hz.clamp(10.0, sample_rate * 0.49);
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let s = 1.0; // slope parameter
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).max(0.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// High-Shelf Filter (boost/cut frequencies above `freq_hz`).
    pub fn high_shelf(freq_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let f = freq_hz.clamp(10.0, sample_rate * 0.49);
        let a = 10.0_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * f / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let s = 1.0;
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).max(0.0).sqrt();
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Band-Pass Filter with constant skirt gain (peak gain = Q).
    pub fn bandpass(freq_hz: f32, q: f32, sample_rate: f32) -> Self {
        let f = freq_hz.clamp(10.0, sample_rate * 0.49);
        let q = q.max(0.1);
        let w0 = 2.0 * PI * f / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;

        Self::new(
            (alpha * q) / a0,
            0.0,
            (-alpha * q) / a0,
            (-2.0 * cos_w0) / a0,
            (1.0 - alpha) / a0,
        )
    }

    /// Process a single input sample through the filter.
    #[inline(always)]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        // Denormal protection
        if self.z1.abs() < 1e-25 {
            self.z1 = 0.0;
        }
        if self.z2.abs() < 1e-25 {
            self.z2 = 0.0;
        }
        y
    }

    /// Process a slice of audio samples in place.
    #[inline]
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}
