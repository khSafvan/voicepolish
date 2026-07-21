//! Main Audio Enhancement Pipeline.
//!
//! Chains noise reduction, dereverberation, corrective filtering, dynamic control,
//! tonal coloration, true-peak limiting, and loudness normalization in optimal order.

use anyhow::Result;
use crate::config::PipelineConfig;
use crate::dsp::{
    Biquad, Compressor, DeEsser, Dereverberator, Limiter, LoudnessMeter, SpectralDenoiser,
};
use crate::io::resample_to_target;

#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub input_duration_sec: f32,
    pub original_sample_rate: u32,
    pub input_lufs: f32,
    pub input_peak_db: f32,
    pub output_lufs: f32,
    pub output_peak_db: f32,
    pub loudness_gain_applied_db: f32,
}

pub struct VoicePipeline {
    config: PipelineConfig,
    sample_rate: f32,
}

impl VoicePipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            sample_rate: 48000.0,
        }
    }

    /// Processes an input audio buffer through the full mastering chain.
    /// Input: mono float samples at `input_sr`.
    /// Output: polished 48kHz float samples + processing metrics.
    pub fn process(&self, input_samples: &[f32], input_sr: u32) -> Result<(Vec<f32>, PipelineStats)> {
        let sr = self.sample_rate;
        let input_duration_sec = input_samples.len() as f32 / input_sr as f32;

        // 1. Resample to 48kHz internal studio standard
        let audio_48k = if input_sr != (sr as u32) {
            resample_to_target(input_samples, input_sr, sr as u32)?
        } else {
            input_samples.to_vec()
        };

        // Measure input metrics
        let mut meter = LoudnessMeter::new(sr);
        let input_lufs = meter.measure_integrated_loudness(&audio_48k);
        let input_peak_db = LoudnessMeter::measure_true_peak(&audio_48k);

        let mut processed = audio_48k.clone();

        // 2. Stage 1: Spectral Denoising & Mains Hum Removal
        if self.config.enable_denoise {
            let denoiser = SpectralDenoiser::new(sr, self.config.denoise_amount_db);
            processed = denoiser.process(&processed);
        }

        // 3. Stage 2: Dereverberation / Room Echo Suppression
        if self.config.enable_dereverb {
            let dereverb = Dereverberator::new(self.config.dereverb_strength);
            processed = dereverb.process(&processed);
        }

        // 4. Stage 3: High-Pass Filter (removes sub-audible table thumps & plosives)
        if self.config.enable_hpf {
            let mut hpf = Biquad::highpass(self.config.hpf_freq, 0.707, sr);
            hpf.process_block(&mut processed);
        }

        // 5. Stage 4: Dynamic De-Esser (removes harsh sibilance)
        if self.config.enable_deesser {
            let mut deesser = DeEsser::new(
                sr,
                self.config.deesser_threshold_db,
                self.config.deesser_reduction_db,
            );
            deesser.process_block(&mut processed);
        }

        // 6. Stage 5: 3-Band Parametric EQ
        if self.config.enable_eq {
            // Warmth (low-shelf at 200 Hz)
            let mut low_shelf = Biquad::low_shelf(200.0, self.config.eq_warmth_db, sr);
            low_shelf.process_block(&mut processed);

            // Presence / Vocal articulation (bell at 3.0 kHz)
            let mut presence_bell = Biquad::peak_eq(3000.0, self.config.eq_presence_db, 1.4, sr);
            presence_bell.process_block(&mut processed);

            // Air & brightness sheen (high-shelf at 10.0 kHz)
            let mut high_shelf = Biquad::high_shelf(10000.0, self.config.eq_air_db, sr);
            high_shelf.process_block(&mut processed);
        }

        // 7. Stage 6: Broadcast Dynamics Compressor
        if self.config.enable_compressor {
            let mut comp = Compressor::new(
                self.config.comp_threshold_db,
                self.config.comp_ratio,
                self.config.comp_attack_ms,
                self.config.comp_release_ms,
                self.config.comp_makeup_db,
                sr,
            );
            comp.process_block(&mut processed);
        }

        // 8. Stage 7: True-Peak Brick-wall Limiter
        if self.config.enable_limiter {
            let mut limiter = Limiter::new(
                self.config.limiter_ceiling_db,
                4.0,  // 4ms lookahead
                60.0, // 60ms smooth release
                sr,
            );
            limiter.process_block(&mut processed);
        }

        // 9. Stage 8: Loudness Normalization (ITU-R BS.1770 / EBU R128)
        let mut gain_applied_db = 0.0;
        if let Some(target_lufs) = self.config.target_lufs {
            gain_applied_db = meter.normalize_to_target(&mut processed, target_lufs);

            // Final safety catch limiter after normalization gain boost
            if gain_applied_db > 0.0 && self.config.enable_limiter {
                let mut final_guard = Limiter::new(
                    self.config.limiter_ceiling_db,
                    2.0,
                    40.0,
                    sr,
                );
                final_guard.process_block(&mut processed);
            }
        }

        // 10. Dry / Wet Blend
        let mix = self.config.dry_wet_mix.clamp(0.0, 1.0);
        if (mix - 1.0).abs() > 0.001 {
            for (i, p) in processed.iter_mut().enumerate() {
                if i < audio_48k.len() {
                    *p = (1.0 - mix) * audio_48k[i] + mix * (*p);
                }
            }
        }

        // 11. Final output metrics
        let output_lufs = meter.measure_integrated_loudness(&processed);
        let output_peak_db = LoudnessMeter::measure_true_peak(&processed);

        let stats = PipelineStats {
            input_duration_sec,
            original_sample_rate: input_sr,
            input_lufs,
            input_peak_db,
            output_lufs,
            output_peak_db,
            loudness_gain_applied_db: gain_applied_db,
        };

        Ok((processed, stats))
    }
}
