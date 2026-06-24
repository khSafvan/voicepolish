//! Production Voice Presets.

use crate::config::PipelineConfig;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum Preset {
    /// Standard podcast release (-16 LUFS, punchy, warm, clear).
    Podcast,
    /// High-energy YouTube master (-14 LUFS, prominent presence, clean top end).
    Youtube,
    /// European Broadcast Standard (-23 LUFS EBU R128, high dynamic range).
    Broadcast,
    /// Audible/ACX compliant (-20 LUFS, -3 dBTP safety ceiling, smooth compression).
    Audiobook,
    /// Classic FM radio tone (heavy compression, deep proximity warmth).
    WarmRadio,
    /// Surgical cleanup only (denoising, dereverb, hum removal without coloration).
    CleanOnly,
}

impl Preset {
    pub fn config(self) -> PipelineConfig {
        match self {
            Preset::Podcast => PipelineConfig {
                target_lufs: Some(-16.0),
                limiter_ceiling_db: -1.0,
                eq_warmth_db: 1.5,
                eq_presence_db: 1.5,
                eq_air_db: 1.0,
                comp_ratio: 2.8,
                comp_threshold_db: -20.0,
                comp_makeup_db: 3.0,
                ..Default::default()
            },

            Preset::Youtube => PipelineConfig {
                target_lufs: Some(-14.0),
                limiter_ceiling_db: -1.0,
                eq_warmth_db: 1.0,
                eq_presence_db: 2.2, // Extra forward presence
                eq_air_db: 1.5,
                comp_ratio: 3.2,
                comp_threshold_db: -22.0,
                comp_makeup_db: 4.5,
                ..Default::default()
            },

            Preset::Broadcast => PipelineConfig {
                target_lufs: Some(-23.0),
                limiter_ceiling_db: -1.0,
                eq_warmth_db: 0.5,
                eq_presence_db: 0.8,
                eq_air_db: 0.5,
                comp_ratio: 2.0, // Gentle broadcast leveling
                comp_threshold_db: -18.0,
                comp_makeup_db: 1.0,
                ..Default::default()
            },

            Preset::Audiobook => PipelineConfig {
                target_lufs: Some(-20.0),
                limiter_ceiling_db: -3.0, // Strict ACX requirement: max -3.0 dBTP
                eq_warmth_db: 0.8,
                eq_presence_db: 0.5,
                eq_air_db: 0.0, // Avoid harshness during multi-hour listening
                comp_ratio: 2.2,
                comp_threshold_db: -20.0,
                comp_makeup_db: 2.0,
                ..Default::default()
            },

            Preset::WarmRadio => PipelineConfig {
                target_lufs: Some(-16.0),
                limiter_ceiling_db: -0.5,
                eq_warmth_db: 3.0, // Rich low-end proximity effect
                eq_presence_db: 2.0,
                eq_air_db: 1.0,
                comp_ratio: 4.0, // Heavy radio broadcast squeeze
                comp_threshold_db: -24.0,
                comp_makeup_db: 5.0,
                ..Default::default()
            },

            Preset::CleanOnly => PipelineConfig {
                enable_eq: false,
                enable_compressor: false,
                enable_deesser: false,
                target_lufs: None, // Keep natural input levels
                ..Default::default()
            },
        }
    }
}
