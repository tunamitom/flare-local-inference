//! Audio and video profile settings as one JSON shape.
//!
//! A package spells these settings twice: once as the payload the frontend deserializes into
//! its own settings types, and once as the description an operator inspects. Both go through
//! this module so a new setting cannot appear in one and be missing from the other.

use super::format::*;
use super::wire;
use crate::Result;

fn object(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(object) => object,
        _ => unreachable!("a JSON object literal is an object"),
    }
}

/// The audio profile's name and its settings.
pub(super) fn audio(settings: &wire::AudioSettings) -> Result<(&'static str, serde_json::Map<String, serde_json::Value>)> {
    use wire::audio_settings::Profile;
    let profile = settings.profile.as_ref().ok_or_else(|| invalid("missing protobuf audio profile"))?;
    Ok(match profile {
        Profile::Mimo(config) => (
            "mimo",
            object(serde_json::json!({
                "sampling_rate": config.sampling_rate,
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "window_length": config.window_length,
                "n_mels": config.n_mels,
                "f_min": config.f_min,
                "f_max": config.f_max,
                "segment_frames": config.segment_frames,
                "kernel_size": config.kernel_size,
                "stride": config.stride,
                "pooling": config.pooling,
                "group_size": config.group_size,
                "max_seconds": config.max_seconds,
            })),
        ),
    })
}

/// The video profile's name and its settings.
pub(super) fn video(settings: &wire::VideoSettings) -> Result<(&'static str, serde_json::Map<String, serde_json::Value>)> {
    use wire::video_settings::Profile;
    let profile = settings.profile.as_ref().ok_or_else(|| invalid("missing protobuf video profile"))?;
    Ok(match profile {
        Profile::Mimo(config) => (
            "mimo",
            object(serde_json::json!({
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
                "total_max_pixels": config.total_max_pixels,
                "num_frames": config.num_frames,
                "max_frames": config.max_frames,
                "pixel_mean": config.pixel_mean,
                "pixel_std": config.pixel_std,
            })),
        ),
    })
}
