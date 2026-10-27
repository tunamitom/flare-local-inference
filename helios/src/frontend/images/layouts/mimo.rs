//! MiMo's conditioned media layout: one feature token per published feature for images, audio
//! segments, and video grids.
//!
//! The chat template emits one placeholder per visual input — `<|image_pad|>` for an image,
//! `<|vision_start|><|video_pad|><|vision_end|>` for a video, and `<|audio_pad|>` inside the audio
//! envelope for a clip. This layout rewrites those placeholders in place:
//!
//! * an image placeholder becomes its patch-merge feature count,
//! * a video placeholder's whole vision envelope becomes the reference processor's timestamped
//!   grid block, one vision graph run per temporal grid,
//! * an audio placeholder becomes one feature token per encoder segment.
//!
//! Every feature a request publishes lands in the same arena in prompt order, so an image, a clip,
//! and a video in one request take consecutive feature ranges. A placeholder count that disagrees
//! with the described inputs is a hard error: a silent shift would publish features under the
//! wrong prompt positions.

use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        audio::PreparedAudioInput,
        images::{ImageProcessor, ImageInput, PreparedImageInput, video::VideoPromptTokens},
    },
    model::{
        ModelConfig, PreparedInput, PromptData,
        media::{DescribedAudio, DescribedVideo},
    },
};
use std::sync::Arc;

/// The tokens this layout rewrites; a kind is absent when the package prepares no such media.
#[derive(Default)]
struct Tokens {
    image: Option<u32>,
    audio: Option<u32>,
    vision_start: Option<u32>,
    vision_end: Option<u32>,
    video: Option<VideoPromptTokens>,
}

fn single(frontend: &Frontend, text: &str) -> Result<u32> {
    match frontend.encode(text, false)?.as_slice() {
        [id] => Ok(*id),
        _ => Err(format!("{text} must encode to one token").into()),
    }
}

impl Tokens {
    fn resolve(
        frontend: &Frontend,
        config: &ModelConfig,
        images: usize,
        audio: usize,
        video: usize,
    ) -> Result<Self> {
        let syntax = &config.contract.frontend;
        let mut tokens = Self::default();
        if images != 0 {
            tokens.image = Some(
                syntax
                    .image_tokens
                    .as_ref()
                    .ok_or("prepared model declares no image feature token")?
                    .feature_token,
            );
        }
        if audio != 0 {
            tokens.audio = Some(
                syntax
                    .audio_tokens
                    .as_ref()
                    .ok_or("prepared model declares no audio feature token")?
                    .feature_token,
            );
        }
        if video != 0 {
            let layout = syntax
                .video_tokens
                .as_ref()
                .ok_or("prepared model declares no video token layout")?;
            let vision_start = single(frontend, "<|vision_start|>")?;
            let vision_end = single(frontend, "<|vision_end|>")?;
            tokens.vision_start = Some(vision_start);
            tokens.vision_end = Some(vision_end);
            tokens.video = Some(VideoPromptTokens {
                feature: layout.feature_token,
                start: layout.start_token,
                end: layout.end_token,
                vision_start,
                vision_end,
            });
        }
        Ok(tokens)
    }
}

fn audio_profile(processor: &ImageProcessor) -> Result<Arc<crate::frontend::audio::AudioProfile>> {
    // The described clips were produced by this profile, so a missing one is a wiring error.
    processor
        .audio()
        .cloned()
        .ok_or_else(|| "prepared model does not support audio inputs".into())
}

pub(crate) fn prepare_media_input(
    config: &ModelConfig,
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageInput],
    audio: &[DescribedAudio],
    video: &[DescribedVideo],
) -> Result<PreparedInput> {
    if config.contract.frontend.position_dimensions != 0 {
        return Err("MiMo media layout requires scalar position dimensions".into());
    }
    let tokens = Tokens::resolve(frontend, config, images.len(), audio.len(), video.len())?;
    if images.is_empty() && audio.is_empty() && video.is_empty() {
        // A prompt with no media still owes the contract one index per token: the validator
        // requires the index list to line up with the prompt whenever the model declares an
        // image layout, and "no feature here" is -1. Returning a default (empty) list made every
        // text-only request fail admission with a conditioned-feature mismatch.
        return Ok(PreparedInput {
            tokens: rendered.token_ids.clone(),
            token_expansions: vec![],
            data: PromptData {
                vision_token_indices: vec![-1; rendered.token_ids.len()],
                ..Default::default()
            },
            conditioning: vec![],
            conditioned_inputs: vec![],
            audio_inputs: vec![],
        });
    }
    let processor = Arc::new(
        processor
            .ok_or("prepared model does not support media inputs")?
            .clone(),
    );
    expand(
        frontend, &tokens, &processor, rendered, images, audio, video,
    )
}

#[allow(clippy::too_many_arguments)]
fn expand(
    frontend: &Frontend,
    tokens: &Tokens,
    processor: &Arc<ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageInput],
    audio: &[DescribedAudio],
    video: &[DescribedVideo],
) -> Result<PreparedInput> {
    // Every placeholder the template emitted has to name one described input.
    let count = |feature: u32| {
        rendered
            .token_ids
            .iter()
            .filter(|id| **id == feature)
            .count()
    };
    for (feature, kind, described) in [
        (tokens.image, "image", images.len()),
        (tokens.audio, "audio", audio.len()),
        (tokens.video.as_ref().map(|video| video.feature), "video", video.len()),
    ] {
        let Some(feature) = feature else { continue };
        if count(feature) != described {
            return Err(format!(
                "prompt has {} {kind} placeholders and {described} described {kind} inputs",
                count(feature)
            )
            .into());
        }
    }
    let audio_profile = if audio.is_empty() {
        None
    } else {
        Some(audio_profile(processor)?)
    };
    let mut tokens_out = Vec::with_capacity(rendered.token_ids.len());
    let mut data = PromptData::default();
    let mut conditioning = Vec::new();
    let mut conditioned_inputs: Vec<PreparedImageInput> = Vec::new();
    let mut audio_inputs: Vec<PreparedAudioInput> = Vec::new();
    let mut token_expansions = Vec::new();
    let mut feature_offset = 0usize;
    let (mut image, mut clip, mut clip_index) = (0usize, 0usize, 0usize);
    let mut index = 0usize;
    while index < rendered.token_ids.len() {
        let token = rendered.token_ids[index];
        if Some(token) == tokens.vision_start
            && tokens.video.as_ref().is_some_and(|video| {
                rendered.token_ids.get(index + 1) == Some(&video.feature)
                    && rendered.token_ids.get(index + 2) == tokens.vision_end.as_ref()
            })
        {
            let video_tokens = tokens.video.as_ref().unwrap();
            let described = video
                .get(clip_index)
                .ok_or("video placeholder has no described clip")?;
            let block = described
                .description
                .prompt_block(video_tokens, |label| frontend.encode(label, false))?;
            let runs = feature_runs(&block, video_tokens.feature);
            if runs.len() != described.description.pairs.len() {
                return Err(format!(
                    "video block carries {} feature runs for {} grids",
                    runs.len(),
                    described.description.pairs.len()
                )
                .into());
            }
            let before = tokens_out.len();
            for (pair_index, (pair, run)) in described
                .description
                .pairs
                .iter()
                .zip(&runs)
                .enumerate()
            {
                let first = described
                    .frames
                    .get(pair.frames[0])
                    .ok_or("video pair names an unsampled frame")?;
                let second = described
                    .frames
                    .get(pair.frames[1])
                    .ok_or("video pair names an unsampled frame")?;
                conditioned_inputs.push(PreparedImageInput::video(
                    pair,
                    first.clone(),
                    second.clone(),
                    before + run.start..before + run.end,
                    feature_offset..feature_offset + pair.geometry.tokens,
                    processor.clone(),
                ));
                feature_offset += pair.geometry.tokens;
                if pair_index == 0 {
                    conditioning.push(ConditioningEvent {
                        position: (before + run.start) as u64,
                        identity: pair.identity,
                    });
                }
            }
            let mut next = 0usize;
            for block_token in &block {
                if *block_token == video_tokens.feature {
                    data.vision_token_indices.push(next as i32);
                    next += 1;
                } else {
                    data.vision_token_indices.push(-1);
                }
            }
            let start = data.vision_token_indices.len() - block.len();
            let base = (feature_offset - next) as i32;
            for value in &mut data.vision_token_indices[start..] {
                if *value >= 0 {
                    *value += base;
                }
            }
            tokens_out.extend_from_slice(&block);
            token_expansions.push((index, tokens_out.len() - before - 3));
            clip_index += 1;
            index += 3;
            continue;
        }
        if Some(token) == tokens.image {
            let input = images
                .get(image)
                .ok_or("image placeholder has no described image")?;
            let features = input.description.geometry.tokens;
            let before = tokens_out.len();
            conditioning.push(ConditioningEvent {
                position: before as u64,
                identity: input.description.identity,
            });
            tokens_out.extend(std::iter::repeat_n(token, features));
            data.vision_token_indices
                .extend((feature_offset..feature_offset + features).map(|i| i as i32));
            conditioned_inputs.push(PreparedImageInput::new(
                input.clone(),
                before..before + features,
                feature_offset..feature_offset + features,
                processor.clone(),
            ));
            feature_offset += features;
            token_expansions.push((index, tokens_out.len() - before - 1));
            image += 1;
            index += 1;
            continue;
        }
        if Some(token) == tokens.audio {
            let described = audio.get(clip).ok_or("audio placeholder has no described clip")?;
            let before = tokens_out.len();
            let features = described.tokens();
            tokens_out.extend(std::iter::repeat_n(token, features));
            data.vision_token_indices
                .extend((feature_offset..feature_offset + features).map(|i| i as i32));
            let prepared = PreparedAudioInput::new(
                before..before + features,
                feature_offset..feature_offset + features,
                described.description.clone(),
                described.capacities.clone(),
                described.mel.clone(),
                audio_profile
                    .clone()
                    .ok_or("prepared model does not support audio inputs")?,
            );
            for segment in 0..prepared.segments().len() {
                conditioning.push(ConditioningEvent {
                    position: prepared.segment_span(segment)?.start as u64,
                    identity: prepared.segments()[segment].identity,
                });
            }
            audio_inputs.push(prepared);
            feature_offset += features;
            token_expansions.push((index, tokens_out.len() - before - 1));
            clip += 1;
            index += 1;
            continue;
        }
        if tokens
            .video
            .as_ref()
            .is_some_and(|video| token == video.feature)
        {
            return Err("MiMo video placeholders require their vision envelope".into());
        }
        tokens_out.push(token);
        data.vision_token_indices.push(-1);
        index += 1;
    }
    if image != images.len() || clip != audio.len() || clip_index != video.len() {
        return Err("prompt media placeholders were not all expanded".into());
    }
    Ok(PreparedInput {
        tokens: tokens_out,
        token_expansions,
        data,
        conditioning,
        conditioned_inputs,
        audio_inputs,
    })
}

/// The feature-token runs of one video block, in grid order.
fn feature_runs(block: &[u32], feature: u32) -> Vec<std::ops::Range<usize>> {
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    for (index, token) in block.iter().enumerate() {
        if *token != feature {
            continue;
        }
        match runs.last_mut() {
            Some(run) if run.end == index => run.end += 1,
            _ => runs.push(index..index + 1),
        }
    }
    runs
}
