//! Request-time decoding of audio and video content parts into described conditioned inputs.
//!
//! Both media kinds are decoded while the prompt is prepared, because their placeholder counts
//! follow from the decoded media: an audio block carries one feature token per encoder segment's
//! log-mel frames, and a video block carries one per grid of sampled frames. A clip's decoded
//! media therefore stays resident from preparation until its last graph run has packed it, which
//! is what the arena admission bounds: the features a request may publish are admitted against
//! `features_per_request`, so the mel frames resident for one request are bounded by the arena
//! the features land in. Decoding itself runs through the frontend profiles' bounded helpers, so a
//! hostile container cannot exceed the accepted extent.

use std::sync::Arc;

use base64::Engine;

use crate::{
    AdmissionError, Error, Result,
    frontend::{
        audio::{AudioDescription, AudioProfile, AudioSource, Mel},
        images::{
            ImageProcessor,
            video::{VideoDescription, VideoProfile, VideoSource},
        },
    },
    model::ModelConfig,
};

/// One decoded audio clip with the features its segments will publish.
pub struct DescribedAudio {
    pub description: AudioDescription,
    pub mel: Arc<Mel>,
    /// Graph capacity per segment: the smallest captured frame capacity that covers its frames.
    pub capacities: Vec<usize>,
}

impl DescribedAudio {
    pub fn tokens(&self) -> usize {
        self.description.tokens()
    }
}

/// One decoded video clip: its described grids and the sampled frames they were described from.
///
/// The frames are shared by reference with the prepared inputs, so the clip's decoded pixels are
/// allocated once and released as each grid is packed.
pub struct DescribedVideo {
    pub description: VideoDescription,
    pub frames: Vec<Arc<crate::frontend::images::RgbImage>>,
}

fn audio_error(message: impl Into<String>) -> Error {
    Error::admission(AdmissionError::Audio, message.into())
}

fn video_error(message: impl Into<String>) -> Error {
    Error::admission(AdmissionError::Video, message.into())
}

fn decode_base64(data: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|error| {
            Error::Admission(AdmissionError::Invalid(crate::Failure::caused_by(
                format!("invalid base64 media payload: {error}"),
                error,
            )))
        })
}

fn audio_profile(processor: Option<&ImageProcessor>) -> Result<&AudioProfile> {
    processor
        .and_then(ImageProcessor::audio)
        .map(Arc::as_ref)
        .ok_or_else(|| audio_error("prepared model does not support audio inputs"))
}

fn video_profile(processor: Option<&ImageProcessor>) -> Result<&VideoProfile> {
    processor
        .and_then(ImageProcessor::video)
        .ok_or_else(|| video_error("prepared model does not support video inputs"))
}

/// Decodes each clip to 24 kHz mono log-mel frames and describes its encoder segments.
pub fn describe_audio(
    config: &ModelConfig,
    processor: Option<&ImageProcessor>,
    sources: &[AudioSource],
) -> Result<Vec<DescribedAudio>> {
    if sources.is_empty() {
        return Ok(vec![]);
    }
    let profile = audio_profile(processor)?;
    let capacities = config
        .contract
        .audio
        .as_ref()
        .filter(|audio| audio.enabled())
        .map(|audio| audio.frame_capacities.clone())
        .ok_or_else(|| audio_error("prepared model declares no audio graphs"))?;
    let mut clips = Vec::with_capacity(sources.len());
    let mut features = 0usize;
    for (index, source) in sources.iter().enumerate() {
        let bytes = decode_base64(&source.data)?;
        if bytes.len() > profile.max_encoded_bytes() {
            return Err(audio_error(format!(
                "encoded audio is {} bytes; the maximum is {}",
                bytes.len(),
                profile.max_encoded_bytes()
            )));
        }
        let (samples, sample_rate) = profile
            .decode(source.media_type(), &bytes)
            .map_err(|error| error.admission_context(AdmissionError::Audio))?;
        let mel = profile
            .log_mel(&samples, sample_rate)
            .map_err(|error| error.admission_context(AdmissionError::Audio))?;
        let description = profile
            .describe(&mel, sample_rate, samples.len())
            .map_err(|error| error.admission_context(AdmissionError::Audio))?;
        let mut clip_capacities = Vec::with_capacity(description.segments.len());
        for segment in &description.segments {
            let capacity = capacities
                .iter()
                .copied()
                .find(|capacity| *capacity >= segment.frames)
                .ok_or_else(|| {
                    audio_error(format!(
                        "audio clip {} is {} mel frames in one segment; the prepared graphs encode at most {}",
                        index + 1,
                        segment.frames,
                        capacities.last().copied().unwrap_or(0)
                    ))
                })?;
            clip_capacities.push(capacity);
        }
        features = features
            .checked_add(description.tokens())
            .ok_or_else(|| audio_error("audio feature count overflow"))?;
        clips.push(DescribedAudio {
            description,
            mel: Arc::new(mel),
            capacities: clip_capacities,
        });
    }
    if features > config.contract.vision.features_per_request {
        return Err(audio_error(format!(
            "audio requests {features} features; the prepared arena holds {}",
            config.contract.vision.features_per_request
        )));
    }
    Ok(clips)
}

/// Decodes each clip to the frames the video frontend samples and describes its temporal grids.
pub fn describe_video(
    config: &ModelConfig,
    processor: Option<&ImageProcessor>,
    sources: &[VideoSource],
) -> Result<Vec<DescribedVideo>> {
    if sources.is_empty() {
        return Ok(vec![]);
    }
    let profile = video_profile(processor)?;
    let limits = profile.limits();
    let mut clips = Vec::with_capacity(sources.len());
    let mut features = 0usize;
    for source in sources {
        let (media, payload) = source
            .encoded()
            .map_err(|error| error.admission_context(AdmissionError::Video))?;
        if !media.starts_with("video/") {
            return Err(video_error(format!(
                "video_url data URL must carry a video media type, not {media}"
            )));
        }
        let bytes = decode_base64(payload)?;
        if bytes.len() > limits.max_encoded_bytes {
            return Err(video_error(format!(
                "encoded video is {} bytes; the maximum is {}",
                bytes.len(),
                limits.max_encoded_bytes
            )));
        }
        let (frames, total_frames) = profile
            .decode(&bytes, &source.options)
            .map_err(|error| error.admission_context(AdmissionError::Video))?;
        let description = profile
            .describe(&frames, total_frames, &source.options)
            .map_err(|error| error.admission_context(AdmissionError::Video))?;
        features = features
            .checked_add(description.tokens())
            .ok_or_else(|| video_error("video feature count overflow"))?;
        clips.push(DescribedVideo {
            description,
            frames: frames.frames.into_iter().map(Arc::new).collect(),
        });
    }
    if features > config.contract.vision.features_per_request {
        return Err(video_error(format!(
            "video requests {features} features; the prepared arena holds {}",
            config.contract.vision.features_per_request
        )));
    }
    Ok(clips)
}
