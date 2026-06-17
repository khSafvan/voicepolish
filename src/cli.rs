//! Command Line Arguments and Subcommands using Clap v4.

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use crate::presets::Preset;

#[derive(Parser, Debug)]
#[command(
    name = "voicepolish",
    author = "VoicePolish Systems Architect",
    version,
    about = "Studio-grade voice enhancement & mastering utility for Linux",
    long_about = "VoicePolish takes raw, reverberant, or noisy spoken-word recordings and automatically transforms them into clean, punchy, broadcast-standard masters."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Polish a single audio recording
    Process(ProcessArgs),

    /// Batch process an entire directory of recordings in parallel
    Batch(BatchArgs),

    /// Inspect and analyze audio characteristics without modifying the file
    Analyze(AnalyzeArgs),

    /// Separate audio into isolated sound layers (Vocals, Music, Drums, Bass, Instruments)
    Separate(SeparateArgs),

    /// Launch the interactive WebUI studio in your browser
    Serve(ServeArgs),
}

#[derive(Args, Debug)]
pub struct ProcessArgs {
    /// Path to input audio recording (WAV, MP3, FLAC, PCM)
    pub input: PathBuf,

    /// Path to save polished output WAV
    pub output: PathBuf,

    /// Broadcast mastering preset
    #[arg(short, long, default_value = "podcast")]
    pub preset: Preset,

    /// Target loudness in LUFS (overrides preset default)
    #[arg(long)]
    pub target_lufs: Option<f32>,

    /// Noise reduction amount in dB (default from preset, typically 12-20)
    #[arg(long)]
    pub denoise_db: Option<f32>,

    /// Dereverberation strength (0.0 = off, 1.0 = maximum dryness)
    #[arg(long)]
    pub dereverb: Option<f32>,

    /// Subsonic rumble filter cutoff in Hz (default 80Hz)
    #[arg(long)]
    pub hpf: Option<f32>,

    /// Dry / Wet mix (0.0 = completely dry, 1.0 = 100% processed)
    #[arg(long, default_value = "1.0")]
    pub mix: f32,

    /// Output bit depth (16 or 24)
    #[arg(long, default_value = "24")]
    pub bit_depth: u16,

    /// Automatically remove background music / isolate vocals using HDemucs neural network
    #[arg(long, alias = "isolate-voice")]
    pub remove_music: bool,

    /// Gain multiplier for vocals/speech layer (default: 1.0)
    #[arg(long)]
    pub gain_vocals: Option<f32>,

    /// Gain multiplier for composite background music layer (0.0 = muted)
    #[arg(long)]
    pub gain_music: Option<f32>,

    /// Gain multiplier for melodic instruments layer (0.0 = muted)
    #[arg(long)]
    pub gain_instruments: Option<f32>,

    /// Gain multiplier for drums/percussion layer (0.0 = muted)
    #[arg(long)]
    pub gain_drums: Option<f32>,

    /// Gain multiplier for bass layer (0.0 = muted)
    #[arg(long)]
    pub gain_bass: Option<f32>,

    /// Optional directory to export individual separated stem WAV files
    #[arg(long)]
    pub stems_dir: Option<PathBuf>,

    /// Bypass noise reduction & hum suppression stage
    #[arg(long)]
    pub no_denoise: bool,

    /// Bypass room reverberation suppression stage
    #[arg(long)]
    pub no_dereverb: bool,

    /// Bypass de-esser
    #[arg(long)]
    pub no_deess: bool,

    /// Bypass 3-band parametric EQ
    #[arg(long)]
    pub no_eq: bool,

    /// Bypass dynamics compressor
    #[arg(long)]
    pub no_compress: bool,

    /// Bypass peak limiter
    #[arg(long)]
    pub no_limit: bool,

    /// Use Resemble Enhance AI Studio engine (flow matching & super-resolution)
    #[arg(long)]
    pub ai: bool,

    /// AI solver steps / number of function evaluations (default: 32)
    #[arg(long, default_value = "32")]
    pub nfe: u32,
}

#[derive(Args, Debug)]
pub struct SeparateArgs {
    /// Path to input audio or video file
    pub input: PathBuf,

    /// Directory where separated stem WAV files will be saved (vocals, music, drums, bass, other)
    #[arg(short, long, default_value = "stems")]
    pub output_dir: PathBuf,

    /// Also export a voice-only mix with music completely removed to this path
    #[arg(short, long)]
    pub output_vocals: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct BatchArgs {
    /// Input directory containing audio files
    pub input_dir: PathBuf,

    /// Output directory where polished files will be saved
    pub output_dir: PathBuf,

    /// Mastering preset to apply to all files
    #[arg(short, long, default_value = "podcast")]
    pub preset: Preset,

    /// Automatically remove background music from all batch files
    #[arg(long)]
    pub remove_music: bool,

    /// Number of parallel threads (defaults to logical CPU cores)
    #[arg(short = 'j', long)]
    pub threads: Option<usize>,

    /// Output bit depth (16 or 24)
    #[arg(long, default_value = "24")]
    pub bit_depth: u16,
}

#[derive(Args, Debug)]
pub struct AnalyzeArgs {
    /// Path to audio recording to analyze
    pub input: PathBuf,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Host interface to bind to
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port number to listen on
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Automatically launch default browser
    #[arg(short, long)]
    pub open: bool,
}
