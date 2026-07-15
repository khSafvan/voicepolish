#!/usr/bin/env python3
"""
AI Studio Sound Layer Separation & Background Music Remover.
Powered by Hybrid Demucs (HDemucs) Deep Neural Network.
Separates mixed audio into 4 stems: Vocals (Voice/Speech), Drums, Bass, and Other (Melodic Instruments/Music).
Allows isolating voice, removing background music, or custom layer balancing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Suppress verbose warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import torch
import torchaudio

# Configure thread pool for Ryzen 7 7700X
torch.set_num_threads(min(torch.get_num_threads(), 8))

# Global cached model
_CACHED_MODEL = None
_CACHED_BUNDLE = None


def get_separation_model(device: str = "cpu"):
    global _CACHED_MODEL, _CACHED_BUNDLE
    if _CACHED_MODEL is None:
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
        model = bundle.get_model()
        model.to(device)
        model.eval()
        _CACHED_MODEL = model
        _CACHED_BUNDLE = bundle
    return _CACHED_MODEL, _CACHED_BUNDLE


def separate_audio(
    wav: torch.Tensor,
    sr: int,
    device: str = "cpu",
    chunk_duration: float = 10.0,
    overlap: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """
    Separates input audio tensor into 4 stems: [drums, bass, other, vocals].
    wav shape: (channels, samples).
    Returns: (4, channels, samples) at model sample rate (44100 Hz).
    """
    model, bundle = get_separation_model(device=device)
    model_sr = bundle.sample_rate  # 44100 Hz

    # Resample to model sample rate if necessary
    if sr != model_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
        wav = resampler(wav)

    channels, total_samples = wav.shape
    if channels == 1:
        # Duplicate mono to stereo for Demucs
        stereo_wav = wav.repeat(2, 1)
    else:
        stereo_wav = wav[:2, :]

    chunk_samples = int(chunk_duration * model_sr)
    hop_samples = int((chunk_duration - overlap) * model_sr)

    # Output stems: [4, 2, total_samples]
    out_stems = torch.zeros(4, 2, total_samples, device=device)
    weight = torch.zeros(total_samples, device=device)
    hann_window = torch.hann_window(chunk_samples, device=device)

    with torch.no_grad():
        for start in range(0, total_samples, hop_samples):
            end = min(start + chunk_samples, total_samples)
            actual_len = end - start
            chunk = stereo_wav[:, start:end].to(device)

            if actual_len < chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - actual_len))
                w = hann_window.clone()
                w[actual_len:] = 0
            else:
                w = hann_window

            chunk_batch = chunk.unsqueeze(0)  # (1, 2, chunk_samples)
            chunk_out = model(chunk_batch).squeeze(0)  # (4, 2, chunk_samples)

            out_stems[:, :, start:end] += chunk_out[:, :, :actual_len] * w[:actual_len]
            weight[start:end] += w[:actual_len]

    # Normalize by overlap weights
    weight = torch.clamp(weight, min=1e-5)
    out_stems = out_stems / weight.unsqueeze(0).unsqueeze(0)

    return out_stems, model_sr


def calculate_rms(t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(t ** 2)).item())


def process_separation(
    input_path: str,
    output_path: str | None = None,
    output_stems_dir: str | None = None,
    remove_music: bool = False,
    gain_vocals: float = 1.0,
    gain_other: float = 1.0,
    gain_drums: float = 1.0,
    gain_bass: float = 1.0,
    device: str = "cpu",
) -> dict:
    start_time = time.time()

    # Load audio
    wav, sr = torchaudio.load(input_path)
    duration_sec = wav.shape[-1] / sr

    # Run HDemucs stem separation
    # Stems order in HDEMUCS_HIGH_MUSDB: [0: drums, 1: bass, 2: other, 3: vocals]
    stems, out_sr = separate_audio(wav, sr, device=device)

    drums = stems[0]   # (2, T)
    bass = stems[1]    # (2, T)
    other = stems[2]   # (2, T)
    vocals = stems[3]  # (2, T)

    # Music composite stem (drums + bass + other)
    music = drums + bass + other

    # Energy calculations
    rms_vocals = calculate_rms(vocals)
    rms_music = calculate_rms(music)
    rms_drums = calculate_rms(drums)
    rms_bass = calculate_rms(bass)
    rms_other = calculate_rms(other)

    total_music_energy = rms_music + 1e-6
    music_presence = rms_music / (rms_vocals + rms_music + 1e-6)

    # If remove_music is requested, mute all musical components
    if remove_music:
        eff_vocals = gain_vocals
        eff_other = 0.0
        eff_drums = 0.0
        eff_bass = 0.0
    else:
        eff_vocals = gain_vocals
        eff_other = gain_other
        eff_drums = gain_drums
        eff_bass = gain_bass

    # Synthesize combined output
    mixed = (
        eff_vocals * vocals
        + eff_other * other
        + eff_drums * drums
        + eff_bass * bass
    )

    # Downmix to mono if input was mono, or keep high-fidelity stereo
    if wav.shape[0] == 1:
        mixed_out = torch.mean(mixed, dim=0, keepdim=True)
    else:
        mixed_out = mixed

    # Save main mixed output if path provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), mixed_out.cpu().numpy().T, out_sr, subtype="PCM_24")

    # Export individual stems if directory provided
    if output_stems_dir:
        stem_dir = Path(output_stems_dir)
        stem_dir.mkdir(parents=True, exist_ok=True)

        sf.write(str(stem_dir / "vocals.wav"), vocals.cpu().numpy().T, out_sr, subtype="PCM_24")
        sf.write(str(stem_dir / "music.wav"), music.cpu().numpy().T, out_sr, subtype="PCM_24")
        sf.write(str(stem_dir / "other.wav"), other.cpu().numpy().T, out_sr, subtype="PCM_24")
        sf.write(str(stem_dir / "drums.wav"), drums.cpu().numpy().T, out_sr, subtype="PCM_24")
        sf.write(str(stem_dir / "bass.wav"), bass.cpu().numpy().T, out_sr, subtype="PCM_24")

    elapsed = time.time() - start_time
    rtf = elapsed / max(duration_sec, 0.001)

    result = {
        "status": "success",
        "input": input_path,
        "output": output_path,
        "output_stems_dir": output_stems_dir,
        "input_sample_rate": sr,
        "output_sample_rate": out_sr,
        "duration_sec": round(duration_sec, 2),
        "processing_time_sec": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "speed_multiplier": round(1.0 / max(rtf, 0.0001), 1),
        "music_presence_percent": round(music_presence * 100.0, 1),
        "music_detected": music_presence > 0.15,
        "layers": {
            "vocals": {"rms": round(rms_vocals, 5), "gain": eff_vocals, "kept": eff_vocals > 0.0},
            "music": {"rms": round(rms_music, 5), "gain": max(eff_other, eff_drums, eff_bass), "kept": (eff_other + eff_drums + eff_bass) > 0.0},
            "instruments": {"rms": round(rms_other, 5), "gain": eff_other, "kept": eff_other > 0.0},
            "drums": {"rms": round(rms_drums, 5), "gain": eff_drums, "kept": eff_drums > 0.0},
            "bass": {"rms": round(rms_bass, 5), "gain": eff_bass, "kept": eff_bass > 0.0},
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="VoicePolish AI Layer Separation & Music Remover")
    parser.add_argument("input", help="Path to input audio/video file")
    parser.add_argument("output", nargs="?", default=None, help="Path to output mixed WAV file")
    parser.add_argument("--stems-dir", default=None, help="Directory to export all separated stem WAVs")
    parser.add_argument("--remove-music", action="store_true", help="Mute background music and isolate vocals")
    parser.add_argument("--gain-vocals", type=float, default=1.0, help="Vocals/Speech layer gain (default: 1.0)")
    parser.add_argument("--gain-music", type=float, default=None, help="Composite music gain (overrides drums/bass/other)")
    parser.add_argument("--gain-other", type=float, default=1.0, help="Melody/Instruments gain (default: 1.0)")
    parser.add_argument("--gain-drums", type=float, default=1.0, help="Drums/Beats gain (default: 1.0)")
    parser.add_argument("--gain-bass", type=float, default=1.0, help="Bass/Sub gain (default: 1.0)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")

    args = parser.parse_args()

    g_other = args.gain_other
    g_drums = args.gain_drums
    g_bass = args.gain_bass

    if args.gain_music is not None:
        g_other = args.gain_music
        g_drums = args.gain_music
        g_bass = args.gain_music

    res = process_separation(
        input_path=args.input,
        output_path=args.output,
        output_stems_dir=args.stems_dir,
        remove_music=args.remove_music,
        gain_vocals=args.gain_vocals,
        gain_other=g_other,
        gain_drums=g_drums,
        gain_bass=g_bass,
        device=args.device,
    )

    if args.json:
        print(json.dumps(res))
    else:
        print("\n" + "=" * 60)
        print("       VOICEPOLISH AI SOUND LAYER SEPARATION")
        print("=" * 60)
        print(f" Input File:             {res['input']}")
        print(f" Output Master:          {res['output']}")
        print(f" Duration:               {res['duration_sec']}s")
        print(f" Processing Time:        {res['processing_time_sec']}s ({res['speed_multiplier']}x realtime)")
        print(f" Music Detected:         {'YES' if res['music_detected'] else 'NO'} ({res['music_presence_percent']}% energy)")
        print(" Layer Weights Applied:")
        for name, info in res['layers'].items():
            status = f"KEPT (gain: {info['gain']})" if info['kept'] else "MUTED"
            print(f"   - {name:<14} : {status:<18} (RMS: {info['rms']})")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
