//! Pipeline Configuration and Parameters.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    // Denoising & Hum Removal
    pub enable_denoise: bool,
    pub denoise_amount_db: f32,

    // Dereverberation
    pub enable_dereverb: bool,
    pub dereverb_strength: f32,

    // Subsonic High-Pass Filter
    pub enable_hpf: bool,
    pub hpf_freq: f32,

    // Dynamic De-Esser
    pub enable_deesser: bool,
    pub deesser_threshold_db: f32,
    pub deesser_reduction_db: f32,

    // 3-Band Parametric EQ
    pub enable_eq: bool,
    pub eq_warmth_db: f32,
    pub eq_presence_db: f32,
    pub eq_air_db: f32,

    // Dynamics Compressor
    pub enable_compressor: bool,
    pub comp_threshold_db: f32,
    pub comp_ratio: f32,
    pub comp_attack_ms: f32,
    pub comp_release_ms: f32,
    pub comp_makeup_db: f32,

    // True-Peak Limiter
    pub enable_limiter: bool,
    pub limiter_ceiling_db: f32,

    // Loudness Normalization
    pub target_lufs: Option<f32>,

    // Dry / Wet Blend (1.0 = fully polished)
    pub dry_wet_mix: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_denoise: true,
            denoise_amount_db: 14.0,

            enable_dereverb: true,
            dereverb_strength: 0.40,

            enable_hpf: true,
            hpf_freq: 80.0,

            enable_deesser: true,
            deesser_threshold_db: -20.0,
            deesser_reduction_db: 5.0,

            enable_eq: true,
            eq_warmth_db: 1.5,
            eq_presence_db: 1.5,
            eq_air_db: 1.0,

            enable_compressor: true,
            comp_threshold_db: -20.0,
            comp_ratio: 2.8,
            comp_attack_ms: 15.0,
            comp_release_ms: 120.0,
            comp_makeup_db: 3.0,

            enable_limiter: true,
            limiter_ceiling_db: -1.0,

            target_lufs: Some(-16.0), // Standard podcast master target
            dry_wet_mix: 1.0,
        }
    }
}
