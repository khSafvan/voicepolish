//! Interface to AI Sound Layer Separation and Background Music Remover.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use super::ai::find_python_env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGains {
    pub remove_music: bool,
    pub vocals: f32,
    pub music: Option<f32>,
    pub instruments: f32,
    pub drums: f32,
    pub bass: f32,
}

impl Default for LayerGains {
    fn default() -> Self {
        Self {
            remove_music: false,
            vocals: 1.0,
            music: None,
            instruments: 1.0,
            drums: 1.0,
            bass: 1.0,
        }
    }
}

impl LayerGains {
    pub fn is_active(&self) -> bool {
        self.remove_music
            || self.music.is_some()
            || (self.vocals - 1.0).abs() > 0.001
            || (self.instruments - 1.0).abs() > 0.001
            || (self.drums - 1.0).abs() > 0.001
            || (self.bass - 1.0).abs() > 0.001
    }

    pub fn vocals_only() -> Self {
        Self {
            remove_music: true,
            vocals: 1.0,
            music: Some(0.0),
            instruments: 0.0,
            drums: 0.0,
            bass: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDetail {
    pub rms: f32,
    pub gain: f32,
    pub kept: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationResult {
    pub status: String,
    pub duration_sec: f32,
    pub processing_time_sec: f32,
    pub speed_multiplier: f32,
    pub music_presence_percent: f32,
    pub music_detected: bool,
    pub layers: HashMap<String, LayerDetail>,
    #[serde(skip)]
    pub audio_bytes: Vec<u8>,
}

pub fn find_separate_script() -> Option<PathBuf> {
    let mut candidates = vec![
        PathBuf::from("voicepolish/ai_separate.py"),
        PathBuf::from("ai_separate.py"),
        PathBuf::from("./voicepolish/ai_separate.py"),
    ];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(target_dir) = exe.parent() {
            if let Ok(canon_root) = target_dir.join("../..").canonicalize() {
                candidates.push(canon_root.join("ai_separate.py"));
            }
            if let Ok(canon_root) = target_dir.join("..").canonicalize() {
                candidates.push(canon_root.join("ai_separate.py"));
            }
            candidates.push(target_dir.join("ai_separate.py"));
        }
    }

    for p in candidates {
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub fn is_separator_available() -> bool {
    find_python_env().is_some() && find_separate_script().is_some()
}

pub fn separate_layers(
    input_audio_bytes: &[u8],
    gains: &LayerGains,
) -> Result<SeparationResult> {
    let python = find_python_env().ok_or_else(|| {
        anyhow!("Python virtual environment (ai_env) not found")
    })?;

    let script = find_separate_script().ok_or_else(|| {
        anyhow!("Layer separation script (ai_separate.py) not found")
    })?;

    let temp_dir = std::env::temp_dir();
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let in_file = temp_dir.join(format!("vp_sep_in_{}.wav", id));
    let out_file = temp_dir.join(format!("vp_sep_out_{}.wav", id));

    fs::write(&in_file, input_audio_bytes)?;

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg(&in_file)
        .arg(&out_file)
        .arg("--gain-vocals")
        .arg(gains.vocals.to_string())
        .arg("--gain-other")
        .arg(gains.instruments.to_string())
        .arg("--gain-drums")
        .arg(gains.drums.to_string())
        .arg("--gain-bass")
        .arg(gains.bass.to_string())
        .arg("--json");

    if gains.remove_music {
        cmd.arg("--remove-music");
    }
    if let Some(m_gain) = gains.music {
        cmd.arg("--gain-music").arg(m_gain.to_string());
    }

    let output = cmd.output()?;
    let _ = fs::remove_file(&in_file);

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&out_file);
        return Err(anyhow!("AI sound layer separation failed: {err}"));
    }

    let out_bytes = fs::read(&out_file)?;
    let _ = fs::remove_file(&out_file);

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let mut result: SeparationResult = serde_json::from_str(stdout_str.trim())
        .map_err(|e| anyhow!("Failed to parse separator JSON: {e} | stdout: {stdout_str}"))?;

    result.audio_bytes = out_bytes;
    Ok(result)
}

pub fn export_stems(
    input_audio_bytes: &[u8],
) -> Result<(SeparationResult, HashMap<String, Vec<u8>>)> {
    let python = find_python_env().ok_or_else(|| {
        anyhow!("Python virtual environment (ai_env) not found")
    })?;

    let script = find_separate_script().ok_or_else(|| {
        anyhow!("Layer separation script (ai_separate.py) not found")
    })?;

    let temp_dir = std::env::temp_dir();
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let in_file = temp_dir.join(format!("vp_stem_in_{}.wav", id));
    let stems_dir = temp_dir.join(format!("vp_stems_{}", id));

    fs::write(&in_file, input_audio_bytes)?;

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg(&in_file)
        .arg("--stems-dir")
        .arg(&stems_dir)
        .arg("--json");

    let output = cmd.output()?;
    let _ = fs::remove_file(&in_file);

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_dir_all(&stems_dir);
        return Err(anyhow!("Stem export failed: {err}"));
    }

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let result: SeparationResult = serde_json::from_str(stdout_str.trim())
        .map_err(|e| anyhow!("Failed to parse separator JSON: {e}"))?;

    let mut stems = HashMap::new();
    let stem_names = ["vocals", "music", "other", "drums", "bass"];
    for name in stem_names {
        let stem_file = stems_dir.join(format!("{}.wav", name));
        if stem_file.exists() {
            if let Ok(bytes) = fs::read(&stem_file) {
                stems.insert(name.to_string(), bytes);
            }
        }
    }

    let _ = fs::remove_dir_all(&stems_dir);
    Ok((result, stems))
}
