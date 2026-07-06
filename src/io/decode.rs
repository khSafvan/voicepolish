//! Multi-Format Audio Decoder using Symphonia.
//!
//! Supports decoding WAV, MP3, FLAC, and PCM into 32-bit floating point audio,
//! downmixing multi-channel microphone recordings to clean mono.

use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use anyhow::{anyhow, Result};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub original_channels: usize,
}

pub fn decode_audio_file<P: AsRef<Path>>(path: P) -> Result<DecodedAudio> {
    let p = path.as_ref();
    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
        if crate::io::video::is_video_extension(ext) {
            let temp_dir = std::env::temp_dir();
            let temp_wav = temp_dir.join(format!("vp_demux_{}.wav", std::process::id()));
            crate::io::video::extract_audio_from_video(p, &temp_wav)?;
            let res = decode_audio_file(&temp_wav);
            let _ = std::fs::remove_file(&temp_wav);
            return res;
        }
    }

    let file = File::open(p)?;
    let mut hint = Hint::new();
    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    decode_from_source(Box::new(file), hint)
}

pub fn decode_audio_bytes(bytes: &[u8]) -> Result<DecodedAudio> {
    // Check if input is a video stream
    if crate::io::video::is_video_bytes(bytes) {
        let wav_bytes = crate::io::video::extract_audio_from_video_bytes(bytes)?;
        return decode_audio_bytes(&wav_bytes);
    }

    let cursor = Cursor::new(bytes.to_vec());
    let mut hint = Hint::new();
    // Guess common formats if magic bytes present
    if bytes.starts_with(b"RIFF") {
        hint.with_extension("wav");
    } else if bytes.starts_with(b"fLaC") {
        hint.with_extension("flac");
    } else if bytes.starts_with(b"ID3") || (bytes.len() > 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0) {
        hint.with_extension("mp3");
    }
    decode_from_source(Box::new(cursor), hint)
}

fn decode_from_source(source: Box<dyn MediaSource>, hint: Hint) -> Result<DecodedAudio> {
    let mss = MediaSourceStream::new(source, Default::default());

    let format_opts = FormatOptions::default();
    let meta_opts = MetadataOptions::default();
    let decode_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &meta_opts)
        .map_err(|e| anyhow!("Failed to probe audio format: {e}"))?;

    let mut format = probed.format;

    // Find the primary audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No supported audio track found in file"))?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("Audio track has unknown sample rate"))?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decode_opts)
        .map_err(|e| anyhow!("Failed to initialize audio decoder: {e}"))?;

    let mut mono_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(anyhow!("Error reading audio packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                append_samples_as_mono(&audio_buf, &mut mono_samples);
            }
            Err(SymphoniaError::DecodeError(e)) => {
                eprintln!("Warning: decode frame error: {e}");
                continue;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(anyhow!("Critical decoder error: {e}")),
        }
    }

    if mono_samples.is_empty() {
        return Err(anyhow!("Audio source contained no decodable audio samples"));
    }

    Ok(DecodedAudio {
        samples: mono_samples,
        sample_rate,
        original_channels: channels,
    })
}

fn append_samples_as_mono(buf: &AudioBufferRef, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(buf) => {
            let n_ch = buf.spec().channels.count();
            let n_frames = buf.frames();
            for f in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_ch {
                    sum += buf.chan(c)[f];
                }
                out.push(sum / (n_ch as f32));
            }
        }
        AudioBufferRef::S16(buf) => {
            let n_ch = buf.spec().channels.count();
            let n_frames = buf.frames();
            for f in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_ch {
                    sum += (buf.chan(c)[f] as f32) / 32768.0;
                }
                out.push(sum / (n_ch as f32));
            }
        }
        AudioBufferRef::S24(buf) => {
            let n_ch = buf.spec().channels.count();
            let n_frames = buf.frames();
            for f in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_ch {
                    sum += (buf.chan(c)[f].0 as f32) / 8388608.0;
                }
                out.push(sum / (n_ch as f32));
            }
        }
        AudioBufferRef::S32(buf) => {
            let n_ch = buf.spec().channels.count();
            let n_frames = buf.frames();
            for f in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_ch {
                    sum += (buf.chan(c)[f] as f32) / 2147483648.0;
                }
                out.push(sum / (n_ch as f32));
            }
        }
        AudioBufferRef::U8(buf) => {
            let n_ch = buf.spec().channels.count();
            let n_frames = buf.frames();
            for f in 0..n_frames {
                let mut sum = 0.0f32;
                for c in 0..n_ch {
                    sum += ((buf.chan(c)[f] as f32) - 128.0) / 128.0;
                }
                out.push(sum / (n_ch as f32));
            }
        }
        _ => {
            eprintln!("Warning: unsupported native buffer ref variant encountered");
        }
    }
}
