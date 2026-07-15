//! Interface to Resemble Enhance Studio Neural Flow Matching Model.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use anyhow::{anyhow, Result};

pub fn find_python_env() -> Option<PathBuf> {
    let mut candidates = vec![
        PathBuf::from("voicepolish/ai_env/bin/python"),
        PathBuf::from("ai_env/bin/python"),
        PathBuf::from("./voicepolish/ai_env/bin/python"),
    ];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(target_dir) = exe.parent() {
            if let Ok(canon_root) = target_dir.join("../..").canonicalize() {
                candidates.push(canon_root.join("ai_env/bin/python"));
            }
            if let Ok(canon_root) = target_dir.join("..").canonicalize() {
                candidates.push(canon_root.join("ai_env/bin/python"));
            }
            candidates.push(target_dir.join("ai_env/bin/python"));
        }
    }

    for p in candidates {
        // Note: Do NOT call canonicalize() on the python binary path itself!
        // A virtualenv python binary is a symlink to base python; resolving
        // the symlink loses the virtualenv environment and packages.
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub fn find_enhance_script() -> Option<PathBuf> {
    let mut candidates = vec![
        PathBuf::from("voicepolish/ai_enhance.py"),
        PathBuf::from("ai_enhance.py"),
        PathBuf::from("./voicepolish/ai_enhance.py"),
    ];

    if let Ok(exe) = std::env::current_exe() {
        if let Some(target_dir) = exe.parent() {
            if let Ok(canon_root) = target_dir.join("../..").canonicalize() {
                candidates.push(canon_root.join("ai_enhance.py"));
            }
            if let Ok(canon_root) = target_dir.join("..").canonicalize() {
                candidates.push(canon_root.join("ai_enhance.py"));
            }
            candidates.push(target_dir.join("ai_enhance.py"));
        }
    }

    for p in candidates {
        if p.exists() {
            return Some(p);
        }
    }
    None
}

pub fn is_ai_enhancer_available() -> bool {
    find_python_env().is_some() && find_enhance_script().is_some()
}

pub fn enhance_with_ai(
    input_wav_bytes: &[u8],
    nfe: u32,
    lambd: f32,
    tau: f32,
) -> Result<Vec<u8>> {
    let python = find_python_env().ok_or_else(|| {
        anyhow!("Resemble Enhance Python virtual environment (ai_env) not found")
    })?;

    let script = find_enhance_script().ok_or_else(|| {
        anyhow!("Resemble Enhance script (ai_enhance.py) not found")
    })?;

    let temp_dir = std::env::temp_dir();
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let in_file = temp_dir.join(format!("vp_in_{}.wav", id));
    let out_file = temp_dir.join(format!("vp_out_{}.wav", id));

    fs::write(&in_file, input_wav_bytes)?;

    let output = Command::new(&python)
        .arg(&script)
        .arg(&in_file)
        .arg(&out_file)
        .arg("--nfe")
        .arg(nfe.to_string())
        .arg("--lambd")
        .arg(lambd.to_string())
        .arg("--tau")
        .arg(tau.to_string())
        .output()?;

    let _ = fs::remove_file(&in_file);

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&out_file);
        return Err(anyhow!("AI enhancement failed: {err}"));
    }

    let out_bytes = fs::read(&out_file)?;
    let _ = fs::remove_file(&out_file);

    Ok(out_bytes)
}
