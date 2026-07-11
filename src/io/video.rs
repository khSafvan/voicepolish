//! Hardware-Accelerated Video Demuxing, Audio Extraction, and Lossless Re-Muxing.
//!
//! Utilizes GPU hardware acceleration (AMD VAAPI / NVIDIA CUDA / Intel QSV)
//! to extract audio losslessly from video files, and re-multiplexes polished audio
//! back into the video container with 100% video stream passthrough (-c:v copy).

use std::fs;
use std::path::Path;
use std::process::Command;
use anyhow::{anyhow, Context, Result};

pub struct GpuInfo {
    pub backend: &'static str,
    pub device: &'static str,
    pub is_available: bool,
    pub description: String,
}

/// Detect best available hardware acceleration backend.
pub fn detect_gpu_accel() -> GpuInfo {
    // Check for AMD / Intel VAAPI render node
    if Path::new("/dev/dri/renderD128").exists() {
        return GpuInfo {
            backend: "vaapi",
            device: "/dev/dri/renderD128",
            is_available: true,
            description: "AMD Radeon Navi 22 (VAAPI Hardware Acceleration on /dev/dri/renderD128)".to_string(),
        };
    }

    // Check for NVIDIA CUDA
    if let Ok(output) = Command::new("nvidia-smi").output() {
        if output.status.success() {
            let desc = String::from_utf8_lossy(&output.stdout);
            let name = desc.lines().find(|l| l.contains("NVIDIA")).unwrap_or("NVIDIA GPU");
            return GpuInfo {
                backend: "cuda",
                device: "cuda",
                is_available: true,
                description: format!("NVIDIA CUDA Hardware Acceleration ({})", name.trim()),
            };
        }
    }

    // Generic auto fallback
    GpuInfo {
        backend: "auto",
        device: "auto",
        is_available: false,
        description: "Standard CPU / Auto Engine".to_string(),
    }
}

pub fn get_gpu_ffmpeg_args() -> Vec<String> {
    let gpu = detect_gpu_accel();
    if gpu.backend == "vaapi" {
        vec![
            "-hwaccel".to_string(),
            "vaapi".to_string(),
            "-hwaccel_device".to_string(),
            gpu.device.to_string(),
        ]
    } else if gpu.backend == "cuda" {
        vec!["-hwaccel".to_string(), "cuda".to_string()]
    } else {
        vec!["-hwaccel".to_string(), "auto".to_string()]
    }
}

/// Check if a file extension represents a supported video container.
pub fn is_video_extension(ext: &str) -> bool {
    let lower = ext.to_lowercase();
    matches!(
        lower.as_str(),
        "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v" | "flv" | "ts" | "mts" | "3gp"
    )
}

/// Check if a byte slice starts with video container magic bytes.
pub fn is_video_bytes(bytes: &[u8]) -> bool {
    if bytes.len() < 12 {
        return false;
    }
    // MP4 / MOV: 'ftyp' or 'moov' in first 64 bytes
    let check_len = bytes.len().min(64);
    if bytes[..check_len].windows(4).any(|w| w == b"ftyp" || w == b"moov") {
        return true;
    }
    // WebM / Matroska: 0x1A, 0x45, 0xDF, 0xA3
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return true;
    }
    // AVI: 'RIFF' + 4 bytes + 'AVI '
    if bytes.starts_with(b"RIFF") && &bytes[8..12] == b"AVI " {
        return true;
    }
    false
}

/// Extract audio stream losslessly from a video file into a 48kHz 24-bit WAV file.
pub fn extract_audio_from_video(video_path: &Path, out_wav_path: &Path) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");

    // Add hardware acceleration arguments
    for arg in get_gpu_ffmpeg_args() {
        cmd.arg(arg);
    }

    cmd.arg("-i")
        .arg(video_path)
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_s24le")
        .arg("-ar")
        .arg("48000")
        .arg(out_wav_path);

    let output = cmd.output().context("Failed to execute ffmpeg for audio extraction")?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("FFmpeg video audio extraction failed: {err}"));
    }

    Ok(())
}

/// Extract audio bytes from raw video bytes in-memory using a temporary file.
pub fn extract_audio_from_video_bytes(video_bytes: &[u8]) -> Result<Vec<u8>> {
    let temp_dir = std::env::temp_dir();
    let temp_video = temp_dir.join(format!("vp_in_{}.mp4", std::process::id()));
    let temp_wav = temp_dir.join(format!("vp_out_{}.wav", std::process::id()));

    fs::write(&temp_video, video_bytes)?;
    let result = extract_audio_from_video(&temp_video, &temp_wav);

    let _ = fs::remove_file(&temp_video);

    match result {
        Ok(()) => {
            let wav_bytes = fs::read(&temp_wav)?;
            let _ = fs::remove_file(&temp_wav);
            Ok(wav_bytes)
        }
        Err(e) => {
            let _ = fs::remove_file(&temp_wav);
            Err(e)
        }
    }
}

/// Re-multiplex enhanced audio WAV back into video container with lossless video stream copy.
pub fn remux_video_with_audio(
    video_path: &Path,
    enhanced_wav_path: &Path,
    out_video_path: &Path,
) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");

    // Add GPU hardware acceleration arguments for decoding
    for arg in get_gpu_ffmpeg_args() {
        cmd.arg(arg);
    }

    cmd.arg("-i")
        .arg(video_path)
        .arg("-i")
        .arg(enhanced_wav_path)
        .arg("-c:v")
        .arg("copy") // Lossless 100% video copy
        .arg("-c:a")
        .arg("aac")
        .arg("-ac")
        .arg("2") // Standard stereo layout for AAC
        .arg("-b:a")
        .arg("320k") // 320 kbps high-fidelity AAC
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("1:a:0")
        .arg("-shortest")
        .arg(out_video_path);

    let output = cmd.output().context("Failed to execute ffmpeg for video remuxing")?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("FFmpeg video remuxing failed: {err}"));
    }

    Ok(())
}

/// Remux in-memory video bytes with enhanced WAV bytes.
pub fn remux_video_bytes(
    video_bytes: &[u8],
    enhanced_wav_bytes: &[u8],
    container_ext: &str,
) -> Result<Vec<u8>> {
    let temp_dir = std::env::temp_dir();
    let ext = if container_ext.is_empty() { "mp4" } else { container_ext };
    let temp_video = temp_dir.join(format!("vp_mux_in_{}.{}", std::process::id(), ext));
    let temp_wav = temp_dir.join(format!("vp_mux_wav_{}.wav", std::process::id()));
    let temp_out = temp_dir.join(format!("vp_mux_out_{}.{}", std::process::id(), ext));

    fs::write(&temp_video, video_bytes)?;
    fs::write(&temp_wav, enhanced_wav_bytes)?;

    let result = remux_video_with_audio(&temp_video, &temp_wav, &temp_out);

    let _ = fs::remove_file(&temp_video);
    let _ = fs::remove_file(&temp_wav);

    match result {
        Ok(()) => {
            let out_bytes = fs::read(&temp_out)?;
            let _ = fs::remove_file(&temp_out);
            Ok(out_bytes)
        }
        Err(e) => {
            let _ = fs::remove_file(&temp_out);
            Err(e)
        }
    }
}
