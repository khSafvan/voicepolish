use std::f32::consts::PI;
use voicepolish::config::PipelineConfig;
use voicepolish::dsp::{Biquad, Compressor, Limiter, LoudnessMeter};
use voicepolish::pipeline::VoicePipeline;
use voicepolish::presets::Preset;

#[test]
fn test_biquad_highpass_attenuation() {
    let sr = 48000.0;
    let mut hpf = Biquad::highpass(80.0, 0.707, sr);

    // 1. Test 20Hz tone (subsonic rumble) -> should be attenuated heavily (>20 dB)
    let n = 4800; // 100ms
    let mut signal_20hz: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 20.0 * (i as f32) / sr).sin())
        .collect();
    hpf.process_block(&mut signal_20hz);

    let late_samples = &signal_20hz[2400..];
    let peak_20hz = late_samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    // At 20Hz, 80Hz 2nd order Butterworth (12dB/oct) should attenuate by ~24dB (peak < 0.1)
    assert!(
        peak_20hz < 0.15,
        "20Hz peak was {}, expected < 0.15 (>16dB attenuation)",
        peak_20hz
    );

    // 2. Test 1000Hz tone (vocal midrange) -> should pass completely unattenuated (~1.0 peak)
    hpf.reset();
    let mut signal_1khz: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 1000.0 * (i as f32) / sr).sin())
        .collect();
    hpf.process_block(&mut signal_1khz);
    let late_1khz = &signal_1khz[2400..];
    let peak_1khz = late_1khz.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    assert!(
        (peak_1khz - 1.0).abs() < 0.05,
        "1kHz peak was {}, expected ~1.0",
        peak_1khz
    );
}

#[test]
fn test_compressor_dynamics() {
    let sr = 48000.0;
    // Threshold -20 dB, ratio 4:1, 0 makeup
    let mut comp = Compressor::new(-20.0, 4.0, 5.0, 100.0, 0.0, sr);

    // 1. Quiet signal at -30 dBFS (amplitude ~0.0316) -> should NOT be compressed
    let n = 4800;
    let mut quiet: Vec<f32> = (0..n)
        .map(|i| 0.0316 * (2.0 * PI * 400.0 * (i as f32) / sr).sin())
        .collect();
    comp.process_block(&mut quiet);
    let peak_quiet = quiet[2400..].iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    assert!(
        (peak_quiet - 0.0316).abs() < 0.005,
        "Quiet signal was compressed unexpectedly: peak = {}",
        peak_quiet
    );

    // 2. Loud signal at 0 dBFS (amplitude 1.0, 20 dB over threshold) -> gain reduced
    let mut loud: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 400.0 * (i as f32) / sr).sin())
        .collect();
    comp.process_block(&mut loud);
    let peak_loud = loud[2400..].iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    // With 20 dB over threshold and 4:1 ratio, gain reduction is ~15 dB => target peak ~ -15 dBFS (~0.18)
    assert!(
        peak_loud < 0.35,
        "Loud signal was not compressed adequately: peak = {}",
        peak_loud
    );
}

#[test]
fn test_limiter_brickwall_ceiling() {
    let sr = 48000.0;
    // Ceiling at -1.0 dBTP => ~0.89125 linear
    let mut limiter = Limiter::new(-1.0, 4.0, 50.0, sr);

    // Feed heavily clipping signal (+6 dBFS, amplitude 2.0)
    let n = 9600; // 200ms
    let mut overload: Vec<f32> = (0..n)
        .map(|i| 2.0 * (2.0 * PI * 500.0 * (i as f32) / sr).sin())
        .collect();

    limiter.process_block(&mut overload);

    let ceiling_linear = 10.0_f32.powf(-1.0 / 20.0); // ~0.89125
    let max_peak = overload.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

    assert!(
        max_peak <= ceiling_linear + 0.001,
        "Limiter ceiling violated! Max peak: {}, ceiling: {}",
        max_peak,
        ceiling_linear
    );
}

#[test]
fn test_loudness_meter_and_normalization() {
    let sr = 48000.0;
    let mut meter = LoudnessMeter::new(sr);

    // 2 seconds of 1kHz sine at full scale
    let n = 96000;
    let mut audio: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 1000.0 * (i as f32) / sr).sin())
        .collect();

    let initial_lufs = meter.measure_integrated_loudness(&audio);
    assert!(
        initial_lufs > -10.0 && initial_lufs < 0.0,
        "Expected initial LUFS ~ -3 to -4 for 0 dBFS sine, got {}",
        initial_lufs
    );

    // Normalize to -16 LUFS (podcast target)
    let gain_applied = meter.normalize_to_target(&mut audio, -16.0);
    assert!(gain_applied < 0.0, "Should have attenuated 0dBFS tone");

    let final_lufs = meter.measure_integrated_loudness(&audio);
    assert!(
        (final_lufs - (-16.0)).abs() < 0.5,
        "Final LUFS was {}, expected -16.0 ± 0.5",
        final_lufs
    );
}

#[test]
fn test_end_to_end_pipeline() {
    let sr = 48000;
    let duration_sec = 1.5;
    let n = (sr as f32 * duration_sec) as usize;

    // Build synthetic recording with voice harmonics, hum, and rumble
    let mut raw_audio = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let voice = 0.4 * (2.0 * PI * 220.0 * t).sin()
            + 0.2 * (2.0 * PI * 440.0 * t).sin()
            + 0.1 * (2.0 * PI * 880.0 * t).sin();
        let hum = 0.06 * (2.0 * PI * 60.0 * t).sin();
        let rumble = 0.08 * (2.0 * PI * 25.0 * t).sin();
        raw_audio.push(voice + hum + rumble);
    }

    let config: PipelineConfig = Preset::Podcast.config();
    let pipeline = VoicePipeline::new(config);

    let (polished, stats) = pipeline.process(&raw_audio, sr).expect("Pipeline failed");

    println!("stats: {:?}", stats);
    assert_eq!(polished.len(), n);
    // When peaks hit the -1 dBTP ceiling, limiter safely prevents clipping
    assert!((stats.output_lufs - (-16.0)).abs() < 1.5);
    assert!(stats.output_peak_db <= -1.0 + 0.05);
}
