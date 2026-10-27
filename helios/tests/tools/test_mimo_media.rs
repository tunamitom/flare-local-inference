//! Preprocess MiMo audio clips and video frames exactly as serving does, for parity tests.
//!
//! The case JSON names the prepared audio and video settings, raw float32 mono PCM clips with
//! their sample rates, and raw RGB8 frame files with the clip's total frame count. Outputs
//! are little-endian tensors plus a metadata JSON describing segments, sampled frames, and
//! frame-pair geometry.

use flare_helios::frontend::{
    audio::{AudioProfile, AudioSettings},
    images::{
        ImageLimits, RgbImage,
        video::{VideoFrames, VideoOptions, VideoProfile, VideoSettings, sample_indices},
    },
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: test_mimo_media CASE_JSON OUTPUT_DIRECTORY".into());
    }
    let case: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let output = Path::new(&args[2]);
    let limits = ImageLimits { max_patches_per_image: 32_768, max_total_patches: 1 << 20, ..ImageLimits::default() };
    let mut metadata = json!({});
    if let Some(audio) = case.get("audio") {
        let settings: AudioSettings = serde_json::from_value(case["audio_settings"].clone())?;
        let profile = AudioProfile::load(&settings, 1 << 30)?;
        let pcm = fs::read(audio["pcm"].as_str().unwrap())?;
        let samples: Vec<f32> = pcm.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let rate = audio["sample_rate"].as_u64().unwrap() as u32;
        let mel = profile.log_mel(&samples, rate)?;
        let description = profile.describe(&mel, rate, samples.len())?;
        fs::write(output.join("mel.f32"), mel.values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        let mut segments = Vec::new();
        for segment in &description.segments {
            let processed = profile.segment(&mel, segment)?;
            segments.push(json!({"offset": segment.offset, "frames": processed.frames, "tokens": segment.tokens}));
        }
        metadata["audio"] = json!({"frames": mel.frames, "mels": mel.mels, "segments": segments,
            "tokens": description.tokens(), "tokens_per_second": profile.tokens_per_second()});
    }
    if let Some(video) = case.get("video") {
        let settings: VideoSettings = serde_json::from_value(case["video_settings"].clone())?;
        let profile = VideoProfile::load(&settings, limits)?;
        let (width, height) = (video["width"].as_u64().unwrap() as usize, video["height"].as_u64().unwrap() as usize);
        let total_frames = video["total_frames"].as_u64().unwrap() as usize;
        let source_fps = video["fps"].as_f64().unwrap_or(0.0);
        let options = VideoOptions {
            num_frames: video["num_frames"].as_u64().map(|n| n as usize),
            fps: video["sample_fps"].as_f64(),
            max_pixels: video["max_pixels"].as_u64().map(|n| n as usize),
        };
        let frames: Vec<RgbImage> = video["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|path| RgbImage::new(width, height, fs::read(path.as_str().unwrap()).unwrap()))
            .collect::<Result<_, _>>()?;
        let count = profile.sampled_count(total_frames, source_fps, &options)?;
        let indices = sample_indices(total_frames, count);
        let frames = VideoFrames { frames };
        let description = profile.describe(&frames, total_frames, &options)?;
        let mut pixels = Vec::new();
        let mut pairs = Vec::new();
        for pair in &description.pairs {
            let processed = profile.process_pair(&frames.frames[pair.frames[0]], &frames.frames[pair.frames[1]], &pair.geometry)?;
            pixels.extend(processed.pixel_values.iter().flat_map(|v| v.to_le_bytes()));
            pairs.push(json!({"frames": pair.frames, "grid_thw": pair.geometry.grid_thw, "patches": pair.geometry.patches,
                "tokens": pair.geometry.tokens, "timestamp": pair.timestamp, "width": pair.geometry.width,
                "height": pair.geometry.height, "positions": processed.position_ids}));
        }
        fs::write(output.join("pixels.f32"), pixels)?;
        metadata["video"] = json!({"sampled_indices": indices, "pairs": pairs, "tokens": description.tokens()});
    }
    fs::write(output.join("metadata.json"), serde_json::to_vec(&metadata)?)?;
    Ok(())
}
