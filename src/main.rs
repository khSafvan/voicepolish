//! VoicePolish CLI Entry Point.
//!
//! High-performance, studio-grade voice processing utility for Linux.

use std::fs;
use std::process::Command;
use std::time::Instant;
use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use voicepolish::cli::{AnalyzeArgs, BatchArgs, Cli, Commands, ProcessArgs, SeparateArgs, ServeArgs};
use voicepolish::config::PipelineConfig;
use voicepolish::dsp::LoudnessMeter;
use voicepolish::enhance::{export_stems, is_separator_available, separate_layers, LayerGains};
use voicepolish::io::{decode_audio_file, write_wav_file};
use voicepolish::pipeline::{PipelineStats, VoicePipeline};
use voicepolish::run_server;

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Process(args) => handle_process(args),
        Commands::Batch(args) => handle_batch(args),
        Commands::Analyze(args) => handle_analyze(args),
        Commands::Separate(args) => handle_separate(args),
        Commands::Serve(args) => handle_serve(args),
    }
}

fn handle_serve(args: ServeArgs) -> Result<()> {
    let url = format!("http://{}:{}", args.host, args.port);

    if args.open {
        // Attempt to open browser in background without blocking
        let _ = Command::new("xdg-open").arg(&url).spawn();
    }

    run_server(&args.host, args.port)
}

fn handle_process(args: ProcessArgs) -> Result<()> {
    let start_time = Instant::now();
    println!("🎙️  VoicePolish: Reading input from {:?}", args.input);

    let raw_bytes = if let Some(ext) = args.input.extension().and_then(|e| e.to_str()) {
        if voicepolish::io::video::is_video_extension(ext) {
            println!("🎬 Detected video container: extracting audio stream...");
            let temp_dir = std::env::temp_dir();
            let temp_wav = temp_dir.join(format!("vp_demux_{}.wav", std::process::id()));
            voicepolish::io::video::extract_audio_from_video(&args.input, &temp_wav)?;
            let b = fs::read(&temp_wav)?;
            let _ = fs::remove_file(&temp_wav);
            b
        } else {
            fs::read(&args.input)
                .with_context(|| format!("Failed to read input file {:?}", args.input))?
        }
    } else {
        fs::read(&args.input)
            .with_context(|| format!("Failed to read input file {:?}", args.input))?
    };

    // Layer separation & background music removal stage
    let layer_gains = LayerGains {
        remove_music: args.remove_music,
        vocals: args.gain_vocals.unwrap_or(1.0),
        music: args.gain_music,
        instruments: args.gain_instruments.unwrap_or(1.0),
        drums: args.gain_drums.unwrap_or(1.0),
        bass: args.gain_bass.unwrap_or(1.0),
    };

    let needs_separation = layer_gains.is_active() || args.stems_dir.is_some();

    let audio_bytes_after_separation = if needs_separation && is_separator_available() {
        println!("🎵 Invoking AI Sound Layer Separation (Hybrid Demucs)...");
        let sep_res = separate_layers(&raw_bytes, &layer_gains)
            .with_context(|| "Sound layer separation / music removal failed")?;

        let status = if sep_res.music_detected {
            format!("YES ({:.1}% energy)", sep_res.music_presence_percent)
        } else {
            format!("NO ({:.1}% energy)", sep_res.music_presence_percent)
        };
        println!("   ⚡ Background Music Detected: {}", status);
        println!("   ⚡ Processing time: {:.2}s ({:.1}x realtime)", sep_res.processing_time_sec, sep_res.speed_multiplier);

        if let Some(ref stems_dir) = args.stems_dir {
            println!("   📂 Exporting separated stems to {:?}", stems_dir);
            fs::create_dir_all(stems_dir)?;
            let (_, stem_files) = export_stems(&raw_bytes)?;
            for (name, bytes) in stem_files {
                let p = stems_dir.join(format!("{}.wav", name));
                fs::write(&p, &bytes)?;
                println!("      - Exported stem: {}", p.display());
            }
        }

        sep_res.audio_bytes
    } else {
        if needs_separation {
            eprintln!("⚠️  Warning: Sound layer separation engine not available, continuing with original audio.");
        }
        raw_bytes
    };

    let decoded = if args.ai && voicepolish::enhance::is_ai_enhancer_available() {
        println!("🧠 Invoking Resemble Enhance Studio AI (Flow Matching + Super-Resolution)...");
        let ai_wav_bytes = voicepolish::enhance::enhance_with_ai(&audio_bytes_after_separation, args.nfe, 0.5, 0.5)
            .with_context(|| "AI Studio enhancement failed")?;

        voicepolish::io::decode_audio_bytes(&ai_wav_bytes)
            .with_context(|| "Failed to decode AI-enhanced audio")?
    } else {
        voicepolish::io::decode_audio_bytes(&audio_bytes_after_separation)
            .with_context(|| "Failed to decode audio")?
    };

    // Build configuration starting from chosen preset
    let mut config: PipelineConfig = args.preset.config();

    // Apply CLI overrides if specified
    if let Some(lufs) = args.target_lufs {
        config.target_lufs = Some(lufs);
    }
    if let Some(dn) = args.denoise_db {
        config.denoise_amount_db = dn;
    }
    if let Some(dr) = args.dereverb {
        config.dereverb_strength = dr;
    }
    if let Some(hpf) = args.hpf {
        config.hpf_freq = hpf;
    }
    config.dry_wet_mix = args.mix;

    // Apply bypass flags
    if args.no_denoise {
        config.enable_denoise = false;
    }
    if args.no_dereverb {
        config.enable_dereverb = false;
    }
    if args.no_deess {
        config.enable_deesser = false;
    }
    if args.no_eq {
        config.enable_eq = false;
    }
    if args.no_compress {
        config.enable_compressor = false;
    }
    if args.no_limit {
        config.enable_limiter = false;
    }

    println!("⚡ Applying mastering chain (Preset: {:?})...", args.preset);
    let pipeline = VoicePipeline::new(config);
    let (polished, stats) = pipeline.process(&decoded.samples, decoded.sample_rate)?;

    // Ensure output directory exists
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let out_ext = args.output.extension().and_then(|e| e.to_str()).unwrap_or("");
    let in_ext = args.input.extension().and_then(|e| e.to_str()).unwrap_or("");

    if voicepolish::io::video::is_video_extension(out_ext)
        || (voicepolish::io::video::is_video_extension(in_ext) && out_ext != "wav")
    {
        println!("🎬 Re-muxing polished audio into video container (-c:v copy lossless video passthrough)...");
        let temp_dir = std::env::temp_dir();
        let temp_wav = temp_dir.join(format!("vp_temp_audio_{}.wav", std::process::id()));
        write_wav_file(&temp_wav, &polished, 48000, 24)?;
        voicepolish::io::video::remux_video_with_audio(&args.input, &temp_wav, &args.output)?;
        let _ = fs::remove_file(&temp_wav);
        println!("✅ Enhanced video created at {:?}", args.output);
    } else {
        println!("💾 Exporting polished audio to {:?}", args.output);
        write_wav_file(&args.output, &polished, 48000, args.bit_depth)?;
    }

    let elapsed = start_time.elapsed();
    let rtf = elapsed.as_secs_f32() / stats.input_duration_sec.max(0.001);

    print_metrics_box(&stats, elapsed.as_secs_f32(), rtf);

    Ok(())
}

fn handle_separate(args: SeparateArgs) -> Result<()> {
    println!("🎵 VoicePolish: Separating sound layers from {:?}", args.input);

    let raw_bytes = if let Some(ext) = args.input.extension().and_then(|e| e.to_str()) {
        if voicepolish::io::video::is_video_extension(ext) {
            println!("🎬 Extracting audio from video container...");
            let temp_dir = std::env::temp_dir();
            let temp_wav = temp_dir.join(format!("vp_demux_{}.wav", std::process::id()));
            voicepolish::io::video::extract_audio_from_video(&args.input, &temp_wav)?;
            let b = fs::read(&temp_wav)?;
            let _ = fs::remove_file(&temp_wav);
            b
        } else {
            fs::read(&args.input)?
        }
    } else {
        fs::read(&args.input)?
    };

    fs::create_dir_all(&args.output_dir)?;

    let (res, stem_files) = export_stems(&raw_bytes)?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              SOUND LAYER SEPARATION SUMMARY                ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Duration:            {:<38}║", format!("{:.2} seconds", res.duration_sec));
    println!("║ Processing Time:     {:<38}║", format!("{:.3} seconds", res.processing_time_sec));
    println!("║ Real-time Speed:     {:<38}║", format!("{:.1}x realtime", res.speed_multiplier));
    println!("║ Music Detected:      {:<38}║", format!("{} ({:.1}% energy)", if res.music_detected { "YES" } else { "NO" }, res.music_presence_percent));
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Exported Stems to:   {:<38}║", args.output_dir.display());

    for (name, bytes) in &stem_files {
        let p = args.output_dir.join(format!("{}.wav", name));
        fs::write(&p, bytes)?;
        println!("║   - {:<17}{:<38}║", format!("{}.wav", name), format!("({:.1} MB)", bytes.len() as f32 / (1024.0 * 1024.0)));
    }
    println!("╚════════════════════════════════════════════════════════════╝\n");

    if let Some(out_v) = args.output_vocals {
        if let Some(vocals_bytes) = stem_files.get("vocals") {
            fs::write(&out_v, vocals_bytes)?;
            println!("✅ Voice-only master saved to {:?}", out_v);
        }
    }

    Ok(())
}

fn handle_batch(args: BatchArgs) -> Result<()> {
    let start_time = Instant::now();

    if !args.input_dir.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", args.input_dir);
    }
    fs::create_dir_all(&args.output_dir)?;

    // Collect supported audio files
    let mut files = Vec::new();
    let valid_extensions = ["wav", "mp3", "flac", "ogg", "aiff", "pcm"];

    for entry in fs::read_dir(&args.input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if valid_extensions.contains(&ext.to_lowercase().as_str()) {
                    files.push(path);
                }
            }
        }
    }

    if files.is_empty() {
        println!("⚠️  No audio files found in {:?}", args.input_dir);
        return Ok(());
    }

    let num_files = files.len();
    println!(
        "🚀 Batch processing {} files with Preset: {:?} (Remove Music: {})...",
        num_files, args.preset, args.remove_music
    );

    // Configure Rayon thread pool if specified
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let progress = ProgressBar::new(num_files as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    let config = args.preset.config();
    let output_dir = args.output_dir.clone();
    let bit_depth = args.bit_depth;
    let remove_music = args.remove_music;

    let results: Vec<Result<(String, f32)>> = files
        .par_iter()
        .map(|in_path| {
            let file_stem = in_path.file_stem().unwrap().to_string_lossy();
            let out_file_name = format!("{}_polished.wav", file_stem);
            let out_path = output_dir.join(out_file_name);

            let decoded = if remove_music && is_separator_available() {
                let raw_bytes = fs::read(in_path)?;
                let sep_res = separate_layers(&raw_bytes, &LayerGains::vocals_only())?;
                voicepolish::io::decode_audio_bytes(&sep_res.audio_bytes)?
            } else {
                decode_audio_file(in_path)?
            };

            let pipeline = VoicePipeline::new(config.clone());
            let (polished, stats) = pipeline.process(&decoded.samples, decoded.sample_rate)?;
            write_wav_file(&out_path, &polished, 48000, bit_depth)?;

            progress.inc(1);
            Ok((file_stem.to_string(), stats.input_duration_sec))
        })
        .collect();

    progress.finish_with_message("Batch processing complete!");

    let mut total_duration = 0.0f32;
    let mut success_count = 0;
    for res in results {
        match res {
            Ok((_, dur)) => {
                total_duration += dur;
                success_count += 1;
            }
            Err(e) => eprintln!("❌ Error processing file: {e}"),
        }
    }

    let elapsed = start_time.elapsed().as_secs_f32();
    let total_rtf = elapsed / total_duration.max(0.001);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                 BATCH SUMMARY COMPLETE                     ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Files Processed:     {:<38}║", format!("{}/{}", success_count, num_files));
    println!("║ Total Audio Time:    {:<38}║", format!("{:.2} seconds ({:.1} min)", total_duration, total_duration / 60.0));
    println!("║ Processing Clock:    {:<38}║", format!("{:.2} seconds", elapsed));
    println!("║ Overall RTF:         {:<38}║", format!("{:.3}x ({:.1}x faster than real-time)", total_rtf, 1.0 / total_rtf.max(0.0001)));
    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn handle_analyze(args: AnalyzeArgs) -> Result<()> {
    println!("🔍 Analyzing audio recording at {:?}...", args.input);

    let decoded = decode_audio_file(&args.input)?;
    let duration_sec = decoded.samples.len() as f32 / decoded.sample_rate as f32;

    let mut meter = LoudnessMeter::new(decoded.sample_rate as f32);
    let lufs = meter.measure_integrated_loudness(&decoded.samples);
    let peak_db = LoudnessMeter::measure_true_peak(&decoded.samples);

    // Calculate approximate RMS and noise floor
    let rms: f32 = (decoded.samples.iter().map(|&x| x * x).sum::<f32>() / decoded.samples.len() as f32).sqrt();
    let rms_db = if rms > 1e-9 { 20.0 * rms.log10() } else { -100.0 };

    // Quietest 5% window for noise floor estimate
    let chunk_size = (decoded.sample_rate as usize) / 4; // 250ms chunks
    let mut chunk_energies = Vec::new();
    let mut idx = 0;
    while idx + chunk_size <= decoded.samples.len() {
        let chunk = &decoded.samples[idx..idx + chunk_size];
        let e: f32 = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk_size as f32).sqrt();
        chunk_energies.push(e);
        idx += chunk_size;
    }
    chunk_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_rms = chunk_energies.get(chunk_energies.len() / 20).copied().unwrap_or(1e-5);
    let noise_db = 20.0 * noise_rms.max(1e-6).log10();

    let dynamic_range_db = peak_db - noise_db;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                 VOICE RECORDING ANALYSIS                   ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ File:                {:<38}║", truncate_str(&args.input.to_string_lossy(), 38));
    println!("║ Duration:            {:<38}║", format!("{:.2}s ({:.1} min)", duration_sec, duration_sec / 60.0));
    println!("║ Sample Rate:         {:<38}║", format!("{} Hz", decoded.sample_rate));
    println!("║ Channels:            {:<38}║", format!("{} (mono mixdown)", decoded.original_channels));
    println!("║ Integrated Loudness: {:<38}║", format!("{:.2} LUFS", lufs));
    println!("║ True Peak:           {:<38}║", format!("{:.2} dBTP", peak_db));
    println!("║ RMS Level:           {:<38}║", format!("{:.2} dBFS", rms_db));
    println!("║ Estimated Noise:     {:<38}║", format!("{:.2} dBFS", noise_db));
    println!("║ Dynamic Range:       {:<38}║", format!("{:.2} dB", dynamic_range_db));
    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn print_metrics_box(stats: &PipelineStats, wall_clock: f32, rtf: f32) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║              VOICEPOLISH MASTERING REPORT                  ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Audio Duration:      {:<38}║", format!("{:.2}s ({:.1} min, {} Hz)", stats.input_duration_sec, stats.input_duration_sec / 60.0, stats.original_sample_rate));
    println!("║ Input Loudness:      {:<38}║", format!("{:.2} LUFS", stats.input_lufs));
    println!("║ Polished Loudness:   {:<38}║", format!("{:.2} LUFS (gain: {:+.1} dB)", stats.output_lufs, stats.loudness_gain_applied_db));
    println!("║ Input Peak:          {:<38}║", format!("{:.2} dBTP", stats.input_peak_db));
    println!("║ Polished Peak:       {:<38}║", format!("{:.2} dBTP", stats.output_peak_db));
    println!("║ Processing Time:     {:<38}║", format!("{:.3} seconds", wall_clock));
    println!("║ Real-Time Factor:    {:<38}║", format!("{:.3}x ({:.1}x real-time speed)", rtf, 1.0 / rtf.max(0.0001)));
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("...{}", &s[s.len() - (max_len - 3)..])
    } else {
        s.to_string()
    }
}
