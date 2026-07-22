//! Built-in Local Web Server for VoicePolish Studio WebUI.
//!
//! Provides a zero-dependency, ultra-fast embedded HTTP server running directly
//! on top of the native AVX-512 DSP pipeline.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use anyhow::Result;

use crate::config::PipelineConfig;
use crate::dsp::LoudnessMeter;
use crate::enhance::{export_stems, is_separator_available, separate_layers, LayerGains};
use crate::io::{decode_audio_bytes, write_wav_bytes};
use crate::pipeline::VoicePipeline;
use crate::presets::Preset;

const HTML_APP: &str = include_str!("web/index.html");

pub fn run_server(host: &str, port: u16) -> Result<()> {
    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr)?;

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                 VOICEPOLISH STUDIO WEBUI                   ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Server URL:          http://{:<31}║", addr);
    println!("║ Local Address:       http://localhost:{:<24}║", port);
    println!("║ Hardware Engine:     AMD Ryzen 7 7700X (AVX-512 SIMD)      ║");
    println!("║ Features:            Voice Mastering · Video Passthrough   ║");
    println!("║                      AI Layer Separation & Music Removal   ║");
    println!("║ Press Ctrl+C to stop server.                               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let listener = Arc::new(listener);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(move || {
                    if let Err(e) = handle_connection(stream) {
                        eprintln!("HTTP connection error: {e}");
                    }
                });
            }
            Err(e) => {
                eprintln!("Accept failed: {e}");
            }
        }
    }

    Ok(())
}

fn handle_connection(mut stream: TcpStream) -> Result<()> {
    // Read HTTP headers
    let mut header_buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let mut header_end = None;

    while header_end.is_none() {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Ok(());
        }
        header_buf.extend_from_slice(&chunk[..n]);

        if let Some(pos) = find_subsequence(&header_buf, b"\r\n\r\n") {
            header_end = Some(pos);
        } else if header_buf.len() > 65536 {
            send_response(&mut stream, 400, "Bad Request", "Header too large", &[])?;
            return Ok(());
        }
    }

    let end_idx = header_end.unwrap();
    let header_str = String::from_utf8_lossy(&header_buf[..end_idx]);
    let mut lines = header_str.lines();

    let request_line = match lines.next() {
        Some(l) => l,
        None => return Ok(()),
    };

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let full_path = parts.next().unwrap_or("/");

    let mut path_and_query = full_path.splitn(2, '?');
    let path = path_and_query.next().unwrap_or("/");
    let query_str = path_and_query.next().unwrap_or("");

    // Parse headers
    let mut content_length: usize = 0;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                if let Ok(len) = v.trim().parse::<usize>() {
                    content_length = len;
                }
            }
        }
    }

    // Read body if content-length is present
    let body_start = end_idx + 4;
    let initial_body_bytes = &header_buf[body_start..];
    let mut body = initial_body_bytes.to_vec();

    while body.len() < content_length {
        let to_read = (content_length - body.len()).min(65536);
        let mut buf = vec![0u8; to_read];
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }

    // CORS preflight
    if method == "OPTIONS" {
        send_cors_preflight(&mut stream)?;
        return Ok(());
    }

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => {
            send_response(
                &mut stream,
                200,
                "OK",
                HTML_APP,
                &[("Content-Type", "text/html; charset=utf-8")],
            )?;
        }

        ("GET", "/api/presets") => {
            let json = r#"["podcast","youtube","broadcast","audiobook","warm-radio","clean-only"]"#;
            send_response(
                &mut stream,
                200,
                "OK",
                json,
                &[("Content-Type", "application/json")],
            )?;
        }

        ("GET", "/api/system/gpu") => {
            let gpu = crate::io::detect_gpu_accel();
            let json = format!(
                r#"{{"backend":"{}","device":"{}","available":{},"description":"{}"}}"#,
                gpu.backend, gpu.device, gpu.is_available, gpu.description
            );
            send_response(
                &mut stream,
                200,
                "OK",
                &json,
                &[
                    ("Content-Type", "application/json"),
                    ("Access-Control-Allow-Origin", "*"),
                ],
            )?;
        }

        ("POST", "/api/analyze") => {
            if body.is_empty() {
                send_response(&mut stream, 400, "Bad Request", "Empty body", &[])?;
                return Ok(());
            }

            let decoded = match decode_audio_bytes(&body) {
                Ok(d) => d,
                Err(e) => {
                    send_response(&mut stream, 400, "Bad Request", &format!("Audio decode error: {e}"), &[])?;
                    return Ok(());
                }
            };

            let duration_sec = decoded.samples.len() as f32 / decoded.sample_rate as f32;
            let mut meter = LoudnessMeter::new(decoded.sample_rate as f32);
            let lufs = meter.measure_integrated_loudness(&decoded.samples);
            let peak_db = LoudnessMeter::measure_true_peak(&decoded.samples);

            let json = format!(
                r#"{{"duration_sec":{:.2},"sample_rate":{},"lufs":{:.2},"peak_db":{:.2}}}"#,
                duration_sec, decoded.sample_rate, lufs, peak_db
            );

            send_response(
                &mut stream,
                200,
                "OK",
                &json,
                &[("Content-Type", "application/json")],
            )?;
        }

        ("POST", "/api/separate") => {
            if body.is_empty() {
                send_response(&mut stream, 400, "Bad Request", "Audio data body required", &[])?;
                return Ok(());
            }

            let raw_audio_bytes = if crate::io::is_video_bytes(&body) {
                match crate::io::video::extract_audio_from_video_bytes(&body) {
                    Ok(b) => b,
                    Err(e) => {
                        send_response(&mut stream, 500, "Internal Server Error", &format!("Video demux error: {e}"), &[])?;
                        return Ok(());
                    }
                }
            } else {
                body.clone()
            };

            if !is_separator_available() {
                send_response(&mut stream, 503, "Service Unavailable", "Sound layer separator not available", &[])?;
                return Ok(());
            }

            match export_stems(&raw_audio_bytes) {
                Ok((meta, _stems)) => {
                    let json = serde_json::to_string(&meta).unwrap_or_else(|_| "{}".to_string());
                    send_response(
                        &mut stream,
                        200,
                        "OK",
                        &json,
                        &[
                            ("Content-Type", "application/json"),
                            ("Access-Control-Allow-Origin", "*"),
                        ],
                    )?;
                }
                Err(e) => {
                    send_response(&mut stream, 500, "Internal Server Error", &format!("Separation error: {e}"), &[])?;
                }
            }
        }

        ("POST", "/api/stem") => {
            if body.is_empty() {
                send_response(&mut stream, 400, "Bad Request", "Audio data body required", &[])?;
                return Ok(());
            }

            let query_map = parse_query(query_str);
            let stem_name = query_map.get("name").map(|s| s.as_str()).unwrap_or("vocals");

            let raw_audio_bytes = if crate::io::is_video_bytes(&body) {
                match crate::io::video::extract_audio_from_video_bytes(&body) {
                    Ok(b) => b,
                    Err(e) => {
                        send_response(&mut stream, 500, "Internal Server Error", &format!("Video demux error: {e}"), &[])?;
                        return Ok(());
                    }
                }
            } else {
                body.clone()
            };

            if !is_separator_available() {
                send_response(&mut stream, 503, "Service Unavailable", "Sound layer separator not available", &[])?;
                return Ok(());
            }

            match export_stems(&raw_audio_bytes) {
                Ok((_, stems)) => {
                    if let Some(bytes) = stems.get(stem_name) {
                        send_binary_response(
                            &mut stream,
                            200,
                            "OK",
                            bytes,
                            &[
                                ("Content-Type", "audio/wav"),
                                ("Access-Control-Allow-Origin", "*"),
                                ("Content-Disposition", &format!("attachment; filename=\"stem_{}.wav\"", stem_name)),
                            ],
                        )?;
                    } else {
                        send_response(&mut stream, 404, "Not Found", "Requested stem not found", &[])?;
                    }
                }
                Err(e) => {
                    send_response(&mut stream, 500, "Internal Server Error", &format!("Stem extraction error: {e}"), &[])?;
                }
            }
        }

        ("POST", "/api/process") => {
            if body.is_empty() {
                send_response(&mut stream, 400, "Bad Request", "Audio data body required", &[])?;
                return Ok(());
            }

            let start_time = Instant::now();
            let query_map = parse_query(query_str);

            let is_video = crate::io::is_video_bytes(&body);

            let raw_audio_bytes = if is_video {
                match crate::io::video::extract_audio_from_video_bytes(&body) {
                    Ok(b) => b,
                    Err(e) => {
                        send_response(&mut stream, 500, "Internal Server Error", &format!("Video demux error: {e}"), &[])?;
                        return Ok(());
                    }
                }
            } else {
                body.clone()
            };

            // Parse layer separation & background music removal params
            let remove_music = query_map.get("remove_music").map(|s| s.as_str()) == Some("true")
                || query_map.get("isolate_vocals").map(|s| s.as_str()) == Some("true");
            let gain_vocals = query_map.get("gain_vocals").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
            let gain_music = query_map.get("gain_music").and_then(|s| s.parse::<f32>().ok());
            let gain_instruments = query_map.get("gain_instruments").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
            let gain_drums = query_map.get("gain_drums").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
            let gain_bass = query_map.get("gain_bass").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);

            let layer_gains = LayerGains {
                remove_music,
                vocals: gain_vocals,
                music: gain_music,
                instruments: gain_instruments,
                drums: gain_drums,
                bass: gain_bass,
            };

            let mut music_detected_str = "No".to_string();
            let mut music_presence_str = "0.0%".to_string();
            let mut layers_tag = "Pass-through".to_string();

            let audio_bytes_to_master = if layer_gains.is_active() && is_separator_available() {
                println!("🎵 Processing AI sound layer separation / music removal...");
                match separate_layers(&raw_audio_bytes, &layer_gains) {
                    Ok(sep_res) => {
                        music_detected_str = if sep_res.music_detected { "Yes".into() } else { "No".into() };
                        music_presence_str = format!("{:.1}%", sep_res.music_presence_percent);
                        layers_tag = if remove_music {
                            "Music Removed (Vocals Isolated)".to_string()
                        } else {
                            format!("Custom Mix (Vocals: {:.1}, Music: {:.1})", gain_vocals, gain_music.unwrap_or(1.0))
                        };
                        sep_res.audio_bytes
                    }
                    Err(e) => {
                        eprintln!("Warning: Layer separation failed, falling back to original audio: {e}");
                        raw_audio_bytes
                    }
                }
            } else {
                raw_audio_bytes
            };

            let is_ai = query_map.get("ai").map(|s| s.as_str()) == Some("true")
                || query_map.get("engine").map(|s| s.as_str()) == Some("ai");

            let (decoded, engine_tag) = if is_ai && crate::enhance::is_ai_enhancer_available() {
                let nfe = query_map.get("nfe").and_then(|s| s.parse::<u32>().ok()).unwrap_or(32);
                let lambd = query_map.get("lambd").and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.5);
                let tau = query_map.get("tau").and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.5);

                match crate::enhance::enhance_with_ai(&audio_bytes_to_master, nfe, lambd, tau) {
                    Ok(ai_wav_bytes) => {
                        match decode_audio_bytes(&ai_wav_bytes) {
                            Ok(d) => (d, "Resemble Enhance (Flow Matching)"),
                            Err(_) => {
                                let fallback = decode_audio_bytes(&audio_bytes_to_master).map_err(|e| anyhow::anyhow!("{e}"))?;
                                (fallback, "Native AVX-512 DSP (Fallback)")
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: AI enhancement error, falling back to DSP: {e}");
                        let fallback = decode_audio_bytes(&audio_bytes_to_master).map_err(|e| anyhow::anyhow!("{e}"))?;
                        (fallback, "Native AVX-512 DSP (Fallback)")
                    }
                }
            } else {
                let d = match decode_audio_bytes(&audio_bytes_to_master) {
                    Ok(d) => d,
                    Err(e) => {
                        send_response(&mut stream, 400, "Bad Request", &format!("Audio decode error: {e}"), &[])?;
                        return Ok(());
                    }
                };
                (d, "Native AVX-512 SIMD DSP")
            };

            // Preset selection
            let preset_str = query_map.get("preset").map(|s| s.as_str()).unwrap_or("podcast");
            let preset = match preset_str {
                "youtube" => Preset::Youtube,
                "broadcast" => Preset::Broadcast,
                "audiobook" => Preset::Audiobook,
                "warm-radio" => Preset::WarmRadio,
                "clean-only" => Preset::CleanOnly,
                _ => Preset::Podcast,
            };

            let mut config: PipelineConfig = preset.config();

            // Apply fine-tuning query parameter overrides if present
            if let Some(v) = query_map.get("denoise").and_then(|s| s.parse::<f32>().ok()) {
                config.denoise_amount_db = v;
            }
            if let Some(v) = query_map.get("dereverb").and_then(|s| s.parse::<f32>().ok()) {
                config.dereverb_strength = v;
            }
            if let Some(v) = query_map.get("warmth").and_then(|s| s.parse::<f32>().ok()) {
                config.eq_warmth_db = v;
            }
            if let Some(v) = query_map.get("presence").and_then(|s| s.parse::<f32>().ok()) {
                config.eq_presence_db = v;
            }
            if let Some(v) = query_map.get("air").and_then(|s| s.parse::<f32>().ok()) {
                config.eq_air_db = v;
            }
            if let Some(v) = query_map.get("mix").and_then(|s| s.parse::<f32>().ok()) {
                config.dry_wet_mix = v;
            }
            if let Some(v) = query_map.get("lufs").and_then(|s| s.parse::<f32>().ok()) {
                config.target_lufs = Some(v);
            }

            let pipeline = VoicePipeline::new(config);
            let (polished, stats) = match pipeline.process(&decoded.samples, decoded.sample_rate) {
                Ok(res) => res,
                Err(e) => {
                    send_response(&mut stream, 500, "Internal Server Error", &format!("Pipeline error: {e}"), &[])?;
                    return Ok(());
                }
            };

            let wav_bytes = match write_wav_bytes(&polished, 48000, 24) {
                Ok(b) => b,
                Err(e) => {
                    send_response(&mut stream, 500, "Internal Server Error", &format!("WAV encoding error: {e}"), &[])?;
                    return Ok(());
                }
            };

            let (final_bytes, content_type) = if is_video {
                let container_ext = query_map.get("ext").map(|s| s.as_str()).unwrap_or("mp4");
                println!("🎬 Remuxing enhanced audio into video (GPU / -c:v copy passthrough)...");
                match crate::io::remux_video_bytes(&body, &wav_bytes, container_ext) {
                    Ok(vb) => (vb, "video/mp4"),
                    Err(e) => {
                        eprintln!("Warning: Video remuxing failed, falling back to WAV audio: {e}");
                        (wav_bytes, "audio/wav")
                    }
                }
            } else {
                (wav_bytes, "audio/wav")
            };

            let elapsed = start_time.elapsed();
            let elapsed_ms = elapsed.as_millis();
            let rtf = elapsed.as_secs_f32() / stats.input_duration_sec.max(0.001);

            let extra_headers = [
                ("Content-Type", content_type),
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Expose-Headers", "X-Input-Lufs, X-Output-Lufs, X-Input-Peak, X-Output-Peak, X-Gain-Applied, X-Duration-Sec, X-Processing-Time-Ms, X-Rtf, X-Engine, X-Media-Type, X-Gpu-Accel, X-Music-Detected, X-Music-Presence, X-Layers-Applied"),
                ("X-Input-Lufs", &format!("{:.2}", stats.input_lufs)),
                ("X-Output-Lufs", &format!("{:.2}", stats.output_lufs)),
                ("X-Input-Peak", &format!("{:.2}", stats.input_peak_db)),
                ("X-Output-Peak", &format!("{:.2}", stats.output_peak_db)),
                ("X-Gain-Applied", &format!("{:+.2}", stats.loudness_gain_applied_db)),
                ("X-Duration-Sec", &format!("{:.2}", stats.input_duration_sec)),
                ("X-Processing-Time-Ms", &format!("{}", elapsed_ms)),
                ("X-Rtf", &format!("{:.4}", rtf)),
                ("X-Engine", engine_tag),
                ("X-Media-Type", if is_video { "video" } else { "audio" }),
                ("X-Gpu-Accel", "AMD Radeon Navi 22 (VAAPI)"),
                ("X-Music-Detected", &music_detected_str),
                ("X-Music-Presence", &music_presence_str),
                ("X-Layers-Applied", &layers_tag),
            ];

            send_binary_response(&mut stream, 200, "OK", &final_bytes, &extra_headers)?;
        }

        _ => {
            send_response(&mut stream, 404, "Not Found", "404 Not Found", &[])?;
        }
    }

    Ok(())
}

fn parse_query(q: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in q.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map
}

fn send_response(
    stream: &mut TcpStream,
    status_code: u16,
    reason: &str,
    body: &str,
    extra_headers: &[(&str, &str)],
) -> Result<()> {
    let body_bytes = body.as_bytes();
    let mut response = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        status_code,
        reason,
        body_bytes.len()
    );

    for (k, v) in extra_headers {
        response.push_str(&format!("{}: {}\r\n", k, v));
    }

    response.push_str("\r\n");

    stream.write_all(response.as_bytes())?;
    stream.write_all(body_bytes)?;
    stream.flush()?;
    Ok(())
}

fn send_binary_response(
    stream: &mut TcpStream,
    status_code: u16,
    reason: &str,
    data: &[u8],
    extra_headers: &[(&str, &str)],
) -> Result<()> {
    let mut header = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        status_code,
        reason,
        data.len()
    );

    for (k, v) in extra_headers {
        header.push_str(&format!("{}: {}\r\n", k, v));
    }

    header.push_str("\r\n");

    stream.write_all(header.as_bytes())?;
    stream.write_all(data)?;
    stream.flush()?;
    Ok(())
}

fn send_cors_preflight(stream: &mut TcpStream) -> Result<()> {
    let resp = "HTTP/1.1 204 No Content\r\n\
                Access-Control-Allow-Origin: *\r\n\
                Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
                Access-Control-Allow-Headers: Content-Type, Content-Length\r\n\
                Connection: close\r\n\r\n";
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}
