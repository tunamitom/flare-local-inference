//! Audio graph inputs, named for the frontend's audio preparation.
//!
//! Every audio graph input is a named output of the audio preparation rather than a row of some
//! shared table: the prepared contract binds each declared field as an `AUDIO_TENSOR`, so the
//! buffers below are the whole interface between a described segment and its encoder graph. The
//! publication contract reads the arena slot, request generation, and destination feature offset
//! from `audio_slot`, `audio_generation`, and `audio_feature_offset`, which is how audio features
//! land in the same arena the vision graphs publish into.

use crate::{
    Result,
    execution::Dtype,
    frontend::audio::ProcessedAudio,
    model::{ModelConfig, inputs::UploadBatch},
    scheduler::RequestHandle,
};

/// Named tensors one audio graph run uploads, in the order the preparation produces them.
pub const TENSORS: [&str; 5] = [
    "audio_features",
    "audio_frames",
    "audio_slot",
    "audio_generation",
    "audio_feature_offset",
];

/// The dtype a named audio tensor carries; `None` when the name is not an audio preparation output.
pub(crate) fn tensor_dtype(name: &str) -> Option<Dtype> {
    match name {
        "audio_features" => Some(Dtype::Float32),
        "audio_slot" | "audio_frames" | "audio_feature_offset" => Some(Dtype::Int32),
        "audio_generation" => Some(Dtype::Int64),
        _ => None,
    }
}

pub(crate) fn prepare(
    config: &ModelConfig,
    handle: RequestHandle,
    slot: usize,
    audio: &ProcessedAudio,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    if slot >= config.contract.vision.feature_slots
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
        || audio.frames == 0
        || audio.mels == 0
        || audio.values.len() != audio.frames * audio.mels
    {
        return Err("invalid audio graph input extent".into());
    }
    i32::try_from(destination_feature_offset)
        .map_err(|_| "audio feature offset exceeds int32")?;
    let frames = i32::try_from(audio.frames).map_err(|_| "audio segment is too long")?;
    let mut staging = UploadBatch::default();
    staging.put(
        "audio_features",
        0,
        audio.values.iter().copied().flat_map(f32::to_le_bytes),
    );
    staging.i32("audio_frames", 0, [frames]);
    staging.i32("audio_slot", 0, [slot as i32]);
    staging.i64("audio_generation", 0, [handle.generation as i64]);
    staging.i32(
        "audio_feature_offset",
        0,
        [destination_feature_offset as i32],
    );
    Ok(staging)
}
