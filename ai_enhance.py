#!/usr/bin/env python3
"""
AI Studio Voice Enhancer (Offline Master).
Powered by Resemble Enhance (Conditional Flow Matching + Neural Vocoder).
Restores muffled recordings, removes heavy reverb, and synthesizes crisp studio air harmonics.
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

from resemble_enhance.enhancer.inference import denoise as re_denoise, enhance as re_enhance


def process_audio(
    input_path: str,
    output_path: str,
    mode: str = "enhance",  # "enhance" or "denoise"
    solver: str = "midpoint",
    nfe: int = 32,
    lambd: float = 0.5,
    tau: float = 0.5,
    device: str = "cpu",
):
    start_time = time.time()

    # Load audio
    dwav, sr = torchaudio.load(input_path)
    # Downmix to mono if multi-channel
    if dwav.shape[0] > 1:
        dwav = torch.mean(dwav, dim=0, keepdim=True)
    dwav = dwav.squeeze(0)  # [T]

    duration_sec = len(dwav) / sr

    # Select compute device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Run AI inference
    if mode == "denoise":
        out_wav, out_sr = re_denoise(dwav, sr, device=device)
    else:
        out_wav, out_sr = re_enhance(
            dwav,
            sr,
            device=device,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
        )

    # Convert to numpy for saving
    out_np = out_wav.detach().cpu().numpy()

    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as 24-bit studio WAV
    sf.write(str(out_path), out_np, out_sr, subtype="PCM_24")

    elapsed = time.time() - start_time
    rtf = elapsed / max(duration_sec, 0.001)

    result = {
        "status": "success",
        "input": input_path,
        "output": output_path,
        "input_sample_rate": sr,
        "output_sample_rate": out_sr,
        "duration_sec": round(duration_sec, 2),
        "processing_time_sec": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "speed_multiplier": round(1.0 / max(rtf, 0.0001), 1),
        "mode": mode,
        "device": device,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="VoicePolish AI Studio Enhancer (Resemble Enhance)")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("output", help="Path to output WAV file")
    parser.add_argument("--mode", choices=["enhance", "denoise"], default="enhance", help="enhance (super-res + denoise) or denoise only")
    parser.add_argument("--nfe", type=int, default=32, help="Number of function evaluations (default: 32, max quality: 64)")
    parser.add_argument("--solver", default="midpoint", choices=["midpoint", "euler", "rk4"], help="ODE solver (midpoint is recommended)")
    parser.add_argument("--lambd", type=float, default=0.5, help="Denoising strength (0.0 = subtle, 1.0 = aggressive, default: 0.5)")
    parser.add_argument("--tau", type=float, default=0.5, help="CFM prior temperature / harmonic air richness (default: 0.5)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")

    args = parser.parse_args()

    result = process_audio(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        solver=args.solver,
        nfe=args.nfe,
        lambd=args.lambd,
        tau=args.tau,
        device=args.device,
    )

    if args.json:
        print(json.dumps(result))
    else:
        print("\n" + "=" * 60)
        print("          VOICEPOLISH AI STUDIO ENHANCEMENT")
        print("=" * 60)
        print(f" Input File:        {result['input']}")
        print(f" Output Master:     {result['output']}")
        print(f" Output Rate:       {result['output_sample_rate']} Hz (High-Fidelity Studio)")
        print(f" Audio Duration:    {result['duration_sec']}s")
        print(f" Processing Time:   {result['processing_time_sec']}s ({result['speed_multiplier']}x realtime)")
        print(f" Mode:              {result['mode'].upper()} (Super-Resolution + CFM)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
