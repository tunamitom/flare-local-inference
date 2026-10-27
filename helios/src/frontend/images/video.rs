//! MiMo video preprocessing: uniform frame sampling, the per-frame pixel budget, float bilinear
//! resizing, and the temporal frame pairs one vision graph run consumes.
//!
//! The serving reference samples at most `num_frames` frames uniformly over a clip (32 by
//! default) and hands them to the processor with one-second nominal timestamps; the container's
//! real rate and presentation times are discarded before the processor ever sees the frames.
//! Every downstream number follows from that fiction: the sampled rate is 1 fps, the last frame
//! is repeated when the count is odd, one temporal grid spans `temporal_patch_size / 1 fps = 2`
//! seconds, so grid `j` is labelled `MM:SS` of `2 * j` and a thirty-second clip and a
//! thirty-minute clip that sample the same number of frames produce the same token sequence.
//!
//! Each frame's area is capped so the whole clip stays inside `total_max_pixels`: the reference
//! divides that budget by the number of sampled frames and bounds the quotient by the profile's
//! own maximum frame area. Frames are resized with float bilinear interpolation at a 32-pixel
//! factor (`patch_size * merge_size`), standardized with ImageNet statistics on the 0..255
//! scale, and packed as `temporal_patch_size`-deep patch stacks: two consecutive frames form one
//! temporal grid, and each 32x32-pixel block of a grid is one vision feature.
//!
//! A video's prompt block replaces the template's whole vision envelope
//! (`<|vision_start|><|video_pad|><|vision_end|>`), which is what the reference processor does
//! when it matches and rewrites that span. vLLM's serving path instead replaces only the single
//! pad token and leaves the template's brackets in place, contributing two more non-feature
//! positions per video; this module follows the processor, whose block `prompt_tokens`
//! reproduces exactly.
//!
//! The accounting here is the real one: `grid_t = ceil(frames/2)` grids and a prompt block that
//! includes its timestamps and per-grid brackets. vLLM's scheduler instead estimates
//! `grid_t = num_frames` (twice the grids that exist, from a `tokens_per_second` resampling the
//! processor never performs) and leaves the timestamp and bracket tokens out. Copying that
//! estimate would reserve twice the features a clip produces, so Helios budgets what it will
//! actually emit.

use sha2::{Digest, Sha256};

use super::{ImageGeometry, ImageLimits, MERGE, PATCH, ProcessedImages, RgbImage, check_input_extent};
use crate::{Error, Failure, FrontendError, Result};

/// Frames one temporal patch holds, and therefore the number of frames per temporal grid.
const TEMPORAL_PATCH: usize = 2;

/// Frames per second the reference presents sampled frames at, whatever the container's rate.
const NOMINAL_FPS: f64 = 1.0;

/// Nominal seconds one temporal grid spans: `temporal_patch_size / NOMINAL_FPS`.
pub const SECOND_PER_GRID_TS: f64 = TEMPORAL_PATCH as f64 / NOMINAL_FPS;

/// Decoded frames of one request may occupy at most this many bytes before resizing.
const MAX_DECODED_BYTES: usize = 1 << 30;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum VideoSettings {
    Mimo(MimoVideoSettings),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoVideoSettings {
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// Pixel budget for every frame of one clip, divided over the frames the loader samples.
    pub total_max_pixels: usize,
    /// Frames sampled from a clip when the request names no count.
    pub num_frames: usize,
    /// Largest frame count a request may ask for.
    pub max_frames: usize,
    pub pixel_mean: [f32; 3],
    pub pixel_std: [f32; 3],
}

/// Per-request sampling controls carried on a video content part.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VideoOptions {
    pub num_frames: Option<usize>,
    /// Requested sampling rate; the loader keeps the smaller of the frame count and the clip's
    /// duration times this rate, as the reference's `fps` target does.
    pub fps: Option<f64>,
    /// Requested cap on one frame's area, in place of the profile's maximum.
    pub max_pixels: Option<usize>,
}

impl VideoOptions {
    /// Reads the sampling controls an OpenAI `video_url` part carries beside its URL.
    pub fn from_part(part: &serde_json::Value) -> Result<Self> {
        let video = part
            .get("video_url")
            .filter(|value| value.is_object())
            .ok_or_else(|| input("video_url must be an object"))?;
        let number = |key: &str| -> Result<Option<usize>> {
            match video.get(key).filter(|value| !value.is_null()) {
                None => Ok(None),
                Some(value) => value
                    .as_u64()
                    .filter(|count| *count > 0)
                    .and_then(|count| usize::try_from(count).ok())
                    .map(Some)
                    .ok_or_else(|| input(format!("{key} must be a positive integer"))),
            }
        };
        let fps = match video.get("fps").filter(|value| !value.is_null()) {
            None => None,
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|fps| fps.is_finite() && *fps > 0.0)
                    .ok_or_else(|| input("fps must be a positive number"))?,
            ),
        };
        Ok(Self {
            num_frames: number("num_frames")?,
            fps,
            max_pixels: number("max_pixels")?,
        })
    }
}

/// One `video_url` content part: a base64 data URL and its sampling controls.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoSource {
    pub url: String,
    pub options: VideoOptions,
}

impl VideoSource {
    /// Reads an OpenAI `video_url` part. Remote URLs are never fetched: the clip has to arrive as
    /// a `data:video/<type>;base64,` URL.
    pub fn from_part(part: &serde_json::Value) -> Result<Self> {
        let url = part
            .get("video_url")
            .and_then(|value| value.get("url"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| input("video_url.url must be a string"))?;
        let Some(rest) = url.strip_prefix("data:") else {
            return Err(input(
                "video_url requires a base64 data URL; remote video URLs are not fetched",
            ));
        };
        let (media, payload) = rest
            .split_once(";base64,")
            .ok_or_else(|| input("video_url data URL must be base64 encoded"))?;
        if !media.starts_with("video/") {
            return Err(input(format!(
                "video_url data URL must carry a video media type, not {media}"
            )));
        }
        if payload.is_empty() {
            return Err(input("video_url data URL carries no payload"));
        }
        Ok(Self {
            url: url.into(),
            options: VideoOptions::from_part(part)?,
        })
    }

    /// The encoded bytes of the clip, and the media type the URL names.
    pub fn encoded(&self) -> Result<(&str, &str)> {
        let rest = self
            .url
            .strip_prefix("data:")
            .ok_or_else(|| input("video_url requires a base64 data URL"))?;
        let (media, payload) = rest
            .split_once(";base64,")
            .ok_or_else(|| input("video_url data URL must be base64 encoded"))?;
        Ok((media, payload))
    }
}

/// One frame pair of a described clip: one temporal grid of the vision graph's input.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VideoPairDescription {
    pub identity: [u8; 32],
    /// Sampled frame indices of the pair; a padded pair repeats the last frame.
    pub frames: [usize; 2],
    pub geometry: ImageGeometry,
    pub timestamp: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VideoDescription {
    pub width: usize,
    pub height: usize,
    /// Frames the container holds.
    pub total_frames: usize,
    /// Frames the loader sampled; the last one is repeated when the count is odd.
    pub sampled_frames: usize,
    /// The clip's 3D patch grid `[ceil(sampled/2), height/16, width/16]`, as one vision input.
    pub grid_thw: [i64; 3],
    pub pairs: Vec<VideoPairDescription>,
}

impl VideoDescription {
    /// Nominal seconds one temporal grid spans, which every clip reports regardless of its rate.
    pub fn second_per_grid_ts(&self) -> f64 {
        SECOND_PER_GRID_TS
    }

    /// Vision features the clip produces: one `video_pad` per 32x32-pixel block per grid.
    pub fn tokens(&self) -> usize {
        self.pairs.iter().map(|pair| pair.geometry.tokens).sum()
    }

    /// Prompt tokens one video contributes, as the reference processor emits them:
    ///
    /// ```text
    /// video_start
    /// for each temporal grid j:
    ///     MM:SS of 2 * j           timestamp_tokens
    ///     vision_start             1
    ///     video_pad                grid_h * grid_w / merge^2
    ///     vision_end               1
    /// video_end                    1
    /// ```
    ///
    /// i.e. `2 + grid_t * (timestamp_tokens + 2) + features`, where `timestamp_tokens` is the
    /// token count of one `MM:SS` label (5 with the checkpoint tokenizer).
    pub fn prompt_tokens(&self, timestamp_tokens: usize) -> usize {
        let mut total = 2 + timestamp_tokens * self.pairs.len();
        for pair in &self.pairs {
            total += 2 + pair.geometry.tokens;
        }
        total
    }

    /// The token ids of this clip's prompt block, in the order the reference processor emits
    /// them. `timestamp` encodes one `MM:SS` label; labels can differ in length, so the result's
    /// length is only `prompt_tokens(timestamp_tokens)` when every label encodes to that many
    /// tokens, which the checkpoint tokenizer does.
    pub fn prompt_block(
        &self,
        tokens: &VideoPromptTokens,
        mut timestamp: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<Vec<u32>> {
        let features = self
            .pairs
            .first()
            .map_or(0, |pair| pair.geometry.tokens);
        let mut block = Vec::with_capacity(self.prompt_tokens(5));
        block.push(tokens.start);
        for pair in &self.pairs {
            if pair.geometry.tokens != features {
                return Err(input("video frame pairs must share one patch grid"));
            }
            let label = timestamp(&pair.timestamp)?;
            if label.is_empty() {
                return Err(input("video timestamp encodes to no tokens"));
            }
            block.extend(label);
            block.push(tokens.vision_start);
            block.extend(std::iter::repeat_n(tokens.feature, features));
            block.push(tokens.vision_end);
        }
        block.push(tokens.end);
        Ok(block)
    }
}

/// The token ids a video block is built from: the prepared contract's feature, start, and end
/// tokens plus the vision envelope the chat template uses around visual inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoPromptTokens {
    pub feature: u32,
    pub start: u32,
    pub end: u32,
    pub vision_start: u32,
    pub vision_end: u32,
}

/// Sampled RGB frames at the clip's native extent.
#[derive(Debug)]
pub struct VideoFrames {
    pub frames: Vec<RgbImage>,
}

fn input(message: impl Into<String>) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message.into())))
}

/// `MM:SS` of a nominal timestamp, as the processor formats it.
pub fn format_timestamp(seconds: f64) -> String {
    let minutes = (seconds / 60.0).floor() as i64;
    let rest = (seconds % 60.0).floor() as i64;
    format!("{minutes:02}:{rest:02}")
}

/// The stream's display rotation in whole counter-clockwise degrees within `0..360`, from the
/// display matrix ffprobe reports as side data; `0` when the stream carries none.
fn display_rotation(stream: &serde_json::Value) -> u32 {
    stream["side_data_list"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|side| super::super::ffmpeg::float(&side["rotation"]))
        .map_or(0, |degrees| degrees.round().rem_euclid(360.0) as u32)
}

/// The reference loader's uniform frame selection: `numpy.linspace(0, total - 1, count, dtype=int)`.
pub fn sample_indices(total_frames: usize, count: usize) -> Vec<usize> {
    if count >= total_frames {
        return (0..total_frames).collect();
    }
    if count <= 1 {
        return vec![0];
    }
    let step = (total_frames - 1) as f64 / (count - 1) as f64;
    let mut indices: Vec<usize> = (0..count).map(|i| (i as f64 * step).trunc() as usize).collect();
    indices[count - 1] = total_frames - 1;
    indices
}

/// The reference `_smart_resize`: aspect-preserving extent aligned to `factor` within an area.
fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    let (mut h, mut w) = (height as f64, width as f64);
    if h.min(w) < factor as f64 {
        if h < w {
            w = (w * factor as f64 / h).trunc();
            h = factor as f64;
        } else {
            h = (h * factor as f64 / w).trunc();
            w = factor as f64;
        }
    } else if h.max(w) / h.min(w) > 200.0 {
        return Err(input(format!("video frame aspect ratio exceeds 200: {height}x{width}")));
    }
    let f = factor as f64;
    // Python round() breaks ties towards the even multiple, as round_ties_even does.
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = (h / beta / f).floor() * f;
        w_bar = (w / beta / f).floor() * f;
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    if h_bar < f || w_bar < f {
        return Err(input("video frame extent is too small for one patch"));
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)` on float frames.
fn bilinear_axis(input: usize, output: usize) -> Vec<(usize, usize, f32, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|o| {
            let source = (scale * (o as f32 + 0.5) - 0.5).max(0.0);
            let index = (source as usize).min(input - 1);
            let next = (index + 1).min(input - 1);
            let lambda1 = source - index as f32;
            (index, next, 1.0 - lambda1, lambda1)
        })
        .collect()
}

pub struct VideoProfile {
    settings: MimoVideoSettings,
    limits: ImageLimits,
    identity_domain: [u8; 32],
}

impl VideoProfile {
    pub fn load(settings: &VideoSettings, limits: ImageLimits) -> Result<Self> {
        let VideoSettings::Mimo(settings) = settings;
        if settings.min_pixels == 0 || settings.max_pixels < settings.min_pixels
            || settings.total_max_pixels < settings.max_pixels || settings.num_frames == 0
            || settings.max_frames < settings.num_frames
            || settings.pixel_mean.iter().any(|v| !v.is_finite())
            || settings.pixel_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(input("invalid prepared video settings"));
        }
        let identity_domain = Sha256::digest(serde_json::to_vec(settings).map_err(|e| input(e.to_string()))?).into();
        Ok(Self { settings: settings.clone(), limits, identity_domain })
    }

    pub fn settings(&self) -> &MimoVideoSettings {
        &self.settings
    }

    pub fn limits(&self) -> ImageLimits {
        self.limits
    }

    /// The resize factor every frame is aligned to: one merged patch, `patch_size * merge_size`.
    pub fn resize_factor(&self) -> usize {
        PATCH * MERGE
    }

    /// Frames the reference loader samples from a clip of `total_frames` at `source_fps`: the
    /// request's count or the packaged default, whichever is smaller than the clip, further
    /// bounded by a requested sampling rate, and never fewer than one.
    pub fn sampled_count(&self, total_frames: usize, source_fps: f64, options: &VideoOptions) -> Result<usize> {
        if total_frames == 0 {
            return Err(input("video contains no frames"));
        }
        let requested = options.num_frames.unwrap_or(self.settings.num_frames);
        if requested == 0 || requested > self.settings.max_frames {
            return Err(input(format!("num_frames must be from 1 through {}", self.settings.max_frames)));
        }
        if let Some(fps) = options.fps {
            if !fps.is_finite() || fps <= 0.0 {
                return Err(input("fps must be a positive number"));
            }
        }
        let mut count = requested.min(total_frames);
        // The loader derives duration from the frame count and the container's rate; a requested
        // rate then caps the count by how many frames fit in that duration.
        if let Some(fps) = options.fps.filter(|_| source_fps > 0.0) {
            count = count.min((total_frames as f64 / source_fps * fps).floor() as usize);
        }
        Ok(count.max(1))
    }

    /// Resized extent and patch grid of one frame of a clip with `sampled` frames.
    pub fn frame_geometry(&self, width: usize, height: usize, sampled: usize, options: &VideoOptions) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        let s = &self.settings;
        let mut max_pixels = s.max_pixels;
        if let Some(requested) = options.max_pixels {
            if requested < s.min_pixels || requested > s.max_pixels {
                return Err(input(format!(
                    "max_pixels must be from {} through {}",
                    s.min_pixels, s.max_pixels
                )));
            }
            max_pixels = requested;
        }
        // The clip's total budget is divided over the frames it keeps, so one frame's area
        // shrinks as the clip samples more of them, down to the profile's minimum area.
        let per_frame = (s.total_max_pixels * TEMPORAL_PATCH / sampled.max(1))
            .min(max_pixels)
            .max(s.min_pixels);
        let (resized_height, resized_width) = smart_resize(height, width, self.resize_factor(), s.min_pixels, per_frame)?;
        let patches = resized_height / PATCH * (resized_width / PATCH);
        super::check_image_patches(self.limits, patches)?;
        Ok(ImageGeometry {
            height: resized_height,
            width: resized_width,
            grid_thw: [1, (resized_height / PATCH) as i64, (resized_width / PATCH) as i64],
            patches,
            tokens: patches / (MERGE * MERGE),
        })
    }

    pub(crate) fn pair_identity(&self, geometry: &ImageGeometry, first: &RgbImage, second: &RgbImage) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"flare-conditioned-video-v1\0");
        hash.update(self.identity_domain);
        for value in [first.width(), first.height(), geometry.width, geometry.height] {
            hash.update((value as u64).to_le_bytes());
        }
        hash.update(first.pixels());
        hash.update(second.pixels());
        hash.finalize().into()
    }

    /// Describes sampled frames as padded, timestamped frame pairs; the pair at index `j` is
    /// temporal grid `j` and carries the nominal time `2 * j` seconds.
    pub fn describe(&self, frames: &VideoFrames, total_frames: usize, options: &VideoOptions) -> Result<VideoDescription> {
        let sampled = frames.frames.len();
        let first = frames.frames.first().ok_or_else(|| input("video contains no frames"))?;
        if frames.frames.iter().any(|frame| frame.width() != first.width() || frame.height() != first.height()) {
            return Err(input("video frames must share one extent"));
        }
        let geometry = self.frame_geometry(first.width(), first.height(), sampled, options)?;
        let grid_t = sampled.div_ceil(TEMPORAL_PATCH);
        let pairs = (0..grid_t)
            .map(|pair| {
                let a = (TEMPORAL_PATCH * pair).min(sampled - 1);
                let b = (TEMPORAL_PATCH * pair + 1).min(sampled - 1);
                VideoPairDescription {
                    identity: self.pair_identity(&geometry, &frames.frames[a], &frames.frames[b]),
                    frames: [a, b],
                    geometry,
                    timestamp: format_timestamp(SECOND_PER_GRID_TS * pair as f64),
                }
            })
            .collect();
        Ok(VideoDescription {
            width: first.width(),
            height: first.height(),
            total_frames,
            sampled_frames: sampled,
            grid_thw: [grid_t as i64, geometry.grid_thw[1], geometry.grid_thw[2]],
            pairs,
        })
    }

    /// Decodes a container to the frames the reference loader would sample, through ffmpeg.
    ///
    /// The frame count comes from the container: its own count when it states one, otherwise
    /// its duration at the stream's average rate. A clip whose stated timing disagrees with the
    /// frames it holds fails the decoded-length check rather than sampling the wrong ones.
    /// Frames are turned to their display orientation the way the reference's OpenCV loader
    /// turns them: quarter and half turns from the stream's display matrix are applied, so a
    /// portrait phone clip yields portrait frames, and any other angle is left alone. Colour
    /// conversion is ffmpeg's rather than OpenCV's, so YUV sources can differ from the reference
    /// loader by codec and matrix rounding; the bilinear resize and standardization damp but do
    /// not remove that difference.
    pub fn decode(&self, bytes: &[u8], options: &VideoOptions) -> Result<(VideoFrames, usize)> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(input("encoded video exceeds the configured byte limit"));
        }
        let staged = super::super::ffmpeg::temp_file(bytes)?;
        let (stream, format) = super::super::ffmpeg::probe_stream_and_format(
            &staged.path,
            "v",
            "nb_frames,avg_frame_rate,r_frame_rate,duration,width,height",
            "rotation",
            "duration",
        )?;
        let coded_width = super::super::ffmpeg::integer(&stream["width"])
            .filter(|width| *width > 0)
            .ok_or_else(|| input("video stream has no extent"))? as usize;
        let coded_height = super::super::ffmpeg::integer(&stream["height"])
            .filter(|height| *height > 0)
            .ok_or_else(|| input("video stream has no extent"))? as usize;
        // ffmpeg's transpose filters name the turn they make; a quarter turn swaps the extent.
        let (turn, width, height) = match display_rotation(&stream) {
            90 => ("transpose=cclock,", coded_height, coded_width),
            180 => ("hflip,vflip,", coded_width, coded_height),
            270 => ("transpose=clock,", coded_height, coded_width),
            _ => ("", coded_width, coded_height),
        };
        check_input_extent(self.limits, width, height)?;
        let fps = super::super::ffmpeg::rational(&stream["avg_frame_rate"])
            .filter(|fps| *fps > 0.0)
            .or_else(|| super::super::ffmpeg::rational(&stream["r_frame_rate"]).filter(|fps| *fps > 0.0))
            .unwrap_or(0.0);
        // Matroska keeps the duration out of the stream, so the container's own field is the
        // fallback that turns a frame rate into a frame count.
        let duration = super::super::ffmpeg::float(&stream["duration"])
            .or_else(|| super::super::ffmpeg::float(&format["duration"]));
        let total_frames = match super::super::ffmpeg::integer(&stream["nb_frames"]).filter(|count| *count > 0) {
            Some(count) => count as usize,
            None => match (duration, fps > 0.0) {
                (Some(duration), true) => (duration * fps).round() as usize,
                _ => return Err(input("video stream reports neither a frame count nor a duration")),
            },
        };
        if total_frames == 0 {
            return Err(input("video stream reports no frames"));
        }
        let count = self.sampled_count(total_frames, fps, options)?;
        let indices = sample_indices(total_frames, count);
        let frame_bytes = width * height * 3;
        let limit = indices.len().checked_mul(frame_bytes).filter(|bytes| *bytes <= MAX_DECODED_BYTES)
            .ok_or_else(|| input(format!(
                "video decodes {count} frames of {width}x{height} px, more than the {} MiB frame budget",
                MAX_DECODED_BYTES >> 20
            )))?;
        let select = indices.iter().map(|index| format!("eq(n\\,{index})")).collect::<Vec<_>>().join("+");
        let mut command = std::process::Command::new("ffmpeg");
        command
            // The explicit turn above replaces ffmpeg's own autorotation, which would also
            // rotate by angles the reference loader ignores.
            .args(["-nostdin", "-v", "error", "-threads", "4", "-noautorotate", "-i"])
            .arg(&staged.path)
            .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf"])
            .arg(format!("{turn}select='{select}'"))
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]);
        let rgb = super::super::ffmpeg::run_bounded(command, "ffmpeg", limit)?;
        drop(staged);
        if rgb.len() != limit {
            return Err(input(format!(
                "video decoded {} of {} sampled frames",
                rgb.len() / frame_bytes.max(1),
                indices.len()
            )));
        }
        let frames = rgb
            .chunks_exact(frame_bytes)
            .map(|pixels| RgbImage::new(width, height, pixels.to_vec()))
            .collect::<Result<Vec<_>>>()?;
        Ok((VideoFrames { frames }, total_frames))
    }

    /// Resizes, standardizes, and packs one frame pair into vision patches: for every merged
    /// 32x32-pixel block, the two frames' patches follow in temporal order.
    pub fn process_pair(&self, first: &RgbImage, second: &RgbImage, geometry: &ImageGeometry) -> Result<ProcessedImages> {
        let (width, height) = (geometry.width, geometry.height);
        let resized: Vec<Vec<f32>> = [first, second]
            .into_iter()
            .map(|frame| self.standardized_bilinear(frame, width, height))
            .collect::<Result<_>>()?;
        let grid_h = height / PATCH;
        let grid_w = width / PATCH;
        let patches = grid_h * grid_w;
        if patches != geometry.patches {
            return Err(input("video pair geometry differs from its frames"));
        }
        let mut result = ProcessedImages {
            pixel_values: Vec::with_capacity(patches * 3 * TEMPORAL_PATCH * PATCH * PATCH),
            geometry: vec![*geometry],
            position_ids: Vec::with_capacity(patches),
            interpolation_indices: Vec::new(),
            interpolation_weights: Vec::new(),
            cu_seqlens: vec![0, patches as i32],
        };
        for by in 0..grid_h / MERGE {
            for bx in 0..grid_w / MERGE {
                for dy in 0..MERGE {
                    for dx in 0..MERGE {
                        let (row, col) = (by * MERGE + dy, bx * MERGE + dx);
                        result.position_ids.push([row as i64, col as i64]);
                        for channel in 0..3 {
                            for frame in &resized {
                                for y in 0..PATCH {
                                    for x in 0..PATCH {
                                        let pixel = (row * PATCH + y) * width + col * PATCH + x;
                                        result.pixel_values.push(frame[pixel * 3 + channel]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    /// Float bilinear resize of `frame` to `width` x `height`, then (x - mean) / std per channel.
    fn standardized_bilinear(&self, frame: &RgbImage, width: usize, height: usize) -> Result<Vec<f32>> {
        if width == 0 || height == 0 {
            return Err(input("video resize target must be positive"));
        }
        let source = frame.pixels();
        let (in_w, in_h) = (frame.width(), frame.height());
        let rows = bilinear_axis(in_h, height);
        let columns = bilinear_axis(in_w, width);
        let mut output = vec![0.0f32; width * height * 3];
        for (y, &(y0, y1, h0, h1)) in rows.iter().enumerate() {
            for (x, &(x0, x1, w0, w1)) in columns.iter().enumerate() {
                for channel in 0..3 {
                    let at = |row: usize, col: usize| f32::from(source[(row * in_w + col) * 3 + channel]);
                    let value = h0 * (w0 * at(y0, x0) + w1 * at(y0, x1)) + h1 * (w0 * at(y1, x0) + w1 * at(y1, x1));
                    output[(y * width + x) * 3 + channel] =
                        (value - self.settings.pixel_mean[channel]) / self.settings.pixel_std[channel];
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
pub(crate) fn test_settings() -> VideoSettings {
    VideoSettings::Mimo(MimoVideoSettings {
        min_pixels: 8192, max_pixels: 8_388_608, total_max_pixels: 268_435_456, num_frames: 32, max_frames: 3600,
        pixel_mean: [123.675, 116.28, 103.53], pixel_std: [58.395, 57.12, 57.375],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(limits: ImageLimits) -> VideoProfile {
        VideoProfile::load(&test_settings(), limits).unwrap()
    }

    fn solid(width: usize, height: usize, value: u8) -> RgbImage {
        RgbImage::new(width, height, vec![value; width * height * 3]).unwrap()
    }

    #[test]
    fn frame_sampling_follows_the_reference_loader() {
        // numpy.linspace(0, 89, 8, dtype=int), the all-frames case, and one frame.
        assert_eq!(sample_indices(90, 8), vec![0, 12, 25, 38, 50, 63, 76, 89]);
        assert_eq!(sample_indices(5, 32), vec![0, 1, 2, 3, 4]);
        assert_eq!(sample_indices(100, 1), vec![0]);
        assert_eq!(sample_indices(1000, 32)[1], 32);
        let profile = profile(ImageLimits::default());
        let default = VideoOptions::default();
        assert_eq!(profile.sampled_count(90, 30.0, &default).unwrap(), 32);
        assert_eq!(profile.sampled_count(20, 30.0, &default).unwrap(), 20);
        // A requested rate caps the count by the clip's duration, but the packaged frame count
        // still caps first: a minute at 30 fps would fill 60 frames at 1 fps, and only 32 exist.
        let fps = VideoOptions { fps: Some(1.0), ..default };
        assert_eq!(profile.sampled_count(90, 30.0, &fps).unwrap(), 3);
        assert_eq!(profile.sampled_count(1800, 30.0, &fps).unwrap(), 32);
        // A rate finer than the source keeps the packaged count as well.
        assert_eq!(profile.sampled_count(90, 30.0, &VideoOptions { fps: Some(60.0), ..default }).unwrap(), 32);
        let both = VideoOptions { num_frames: Some(2), fps: Some(1.0), max_pixels: None };
        assert_eq!(profile.sampled_count(90, 30.0, &both).unwrap(), 2);
        assert!(profile.sampled_count(90, 30.0, &VideoOptions { num_frames: Some(0), ..default }).is_err());
        assert!(profile.sampled_count(90, 30.0, &VideoOptions { num_frames: Some(5000), ..default }).is_err());
        assert!(profile.sampled_count(90, 30.0, &VideoOptions { fps: Some(0.0), ..default }).is_err());
        assert!(profile.sampled_count(0, 30.0, &default).is_err());
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(62.0), "01:02");
        assert_eq!(format_timestamp(3599.9), "59:59");
    }

    #[test]
    fn frame_geometry_divides_the_clip_budget_and_aligns_to_32() {
        let profile = profile(ImageLimits { max_patches_per_image: 65_536, ..ImageLimits::default() });
        let default = VideoOptions::default();
        // 1080p keeps its area at 32 frames: 1920x1088 after alignment.
        let geometry = profile.frame_geometry(1920, 1080, 32, &default).unwrap();
        assert_eq!((geometry.width, geometry.height, geometry.patches, geometry.tokens), (1920, 1088, 8160, 2040));
        // 512 frames cap each frame at 2 * 268,435,456 / 512 = 1,048,576 pixels.
        let geometry = profile.frame_geometry(1920, 1080, 512, &default).unwrap();
        assert!(geometry.width * geometry.height <= 1_048_576);
        assert_eq!((geometry.width, geometry.height), (1344, 768));
        // 720 rows break the tie towards the even 32-pixel multiple: 720 / 32 = 22.5 -> 22.
        let geometry = profile.frame_geometry(1280, 720, 32, &default).unwrap();
        assert_eq!((geometry.width, geometry.height, geometry.tokens), (1280, 704, 880));
        let capped = profile.frame_geometry(1920, 1080, 32, &VideoOptions { max_pixels: Some(921_600), ..default }).unwrap();
        assert!(capped.width * capped.height <= 921_600);
        assert!(profile.frame_geometry(1920, 1080, 32, &VideoOptions { max_pixels: Some(100), ..default }).is_err());
        // The aspect guard only runs once both sides clear the 32-pixel factor.
        assert!(profile.frame_geometry(100, 40_000, 32, &default).is_err());
    }

    #[test]
    fn golden_video_token_counts_match_the_reference_formula() {
        let profile = profile(ImageLimits::default());
        let default = VideoOptions::default();
        // 32 frames of 640x480: 16 temporal grids of 30x40 patches, 300 video features each.
        let frames = VideoFrames { frames: (0..32).map(|index| solid(640, 480, index as u8 * 8)).collect() };
        let description = profile.describe(&frames, 300, &default).unwrap();
        assert_eq!(description.grid_thw, [16, 30, 40]);
        assert_eq!(description.second_per_grid_ts(), 2.0);
        assert_eq!(description.tokens(), 4_800);
        assert_eq!(description.prompt_tokens(5), 4_914);
        assert_eq!(description.pairs[3].timestamp, "00:06");
        assert_eq!(description.pairs[15].timestamp, "00:30");
        // One sampled frame is repeated into a single grid.
        let single = VideoFrames { frames: vec![solid(640, 480, 9)] };
        let description = profile.describe(&single, 1, &default).unwrap();
        assert_eq!((description.sampled_frames, description.grid_thw), (1, [1, 30, 40]));
        assert_eq!(description.pairs[0].frames, [0, 0]);
        assert_eq!(description.tokens(), 300);
        assert_eq!(description.prompt_tokens(5), 309);
        // A 30 s 720p screen recording: 900 frames at 30 fps sample 32, and 720 rows align to
        // 704, so each of the 16 grids carries 44 * 80 / 4 = 880 features and the block is
        // 2 + 16 * (5 + 2) + 14,080 = 14,194 tokens.
        let geometry = profile.frame_geometry(1280, 720, 32, &default).unwrap();
        assert_eq!(16 * geometry.tokens, 14_080);
        assert_eq!(2 + 16 * (5 + 2) + 16 * geometry.tokens, 14_194);
    }

    #[test]
    fn prompt_block_emits_the_reference_groups() {
        let profile = profile(ImageLimits::default());
        let frames = VideoFrames { frames: (0..5).map(|index| solid(64, 48, index as u8 * 30)).collect() };
        let description = profile.describe(&frames, 5, &VideoOptions::default()).unwrap();
        // Five frames pad to six: three grids of 6x8 patches, 12 features each.
        assert_eq!(description.grid_thw, [3, 6, 8]);
        let tokens = VideoPromptTokens { feature: 7, start: 8, end: 9, vision_start: 10, vision_end: 11 };
        // ASCII labels stand in for the tokenizer, which encodes MM:SS to five tokens.
        let encode = |label: &str| Ok(label.bytes().map(u32::from).collect::<Vec<_>>());
        let block = description.prompt_block(&tokens, encode).unwrap();
        assert_eq!(block.len(), description.prompt_tokens(5));
        assert_eq!(block.len(), 2 + 3 * (5 + 2 + 12));
        assert_eq!(block[0], 8);
        assert_eq!(block[1..6], "00:00".bytes().map(u32::from).collect::<Vec<_>>()[..]);
        assert_eq!(block[6], 10);
        assert_eq!(block[7..19], [7; 12]);
        assert_eq!(block[19], 11);
        assert_eq!(block[20..25], "00:02".bytes().map(u32::from).collect::<Vec<_>>()[..]);
        assert_eq!(block[block.len() - 1], 9);
        assert!(description.prompt_block(&tokens, |_| Ok(vec![])).is_err());
    }

    #[test]
    fn video_preparation_requires_the_merge_patch_layout() {
        use crate::frontend::images::{
            ImageProcessor, ImageSettings, MimoImageSettings, PatchMergeImageSettings, QwenImageSettings,
        };
        let limits = ImageLimits::default();
        let mimo = ImageSettings::Mimo(MimoImageSettings {
            min_pixels: 8_192, max_pixels: 8_388_608,
            image_mean: [123.675, 116.28, 103.53], image_std: [58.395, 57.12, 57.375],
        });
        assert!(ImageProcessor::load(mimo, limits).unwrap().with_video(&test_settings()).is_ok());
        // A layout that does not pack 16x16 patches in 2x2 merge groups cannot host a video: the
        // pair packing and the 32-pixel resize factor both assume that geometry.
        let qwen = ImageSettings::Qwen(QwenImageSettings {
            min_pixels: 65_536, max_pixels: 16_777_216, position_side: 48,
            image_token_id: 248_056, video_token_id: 248_057,
        });
        assert!(ImageProcessor::load(qwen, limits).unwrap().with_video(&test_settings()).is_err());
        let patch_merge = ImageSettings::PatchMerge(PatchMergeImageSettings {
            patch_size: 14, min_pixels: 4_096, max_pixels: 4_194_304,
            image_mean: [0.5; 3], image_std: [0.5; 3],
            image_token: "<image>".into(), start_token: "<s>".into(), end_token: "</s>".into(),
        });
        assert!(ImageProcessor::load(patch_merge, limits).unwrap().with_video(&test_settings()).is_err());
    }

    #[test]
    fn bilinear_pairs_pack_temporal_slots_from_distinct_frames() {
        let profile = profile(ImageLimits::default());
        // 2x2 source upscaled to 32x32: torch bilinear with align_corners=False keeps corner
        // values and interpolates linearly between pixel centers.
        let first = RgbImage::new(2, 2, vec![0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0]).unwrap();
        let second = RgbImage::new(2, 2, vec![100; 12]).unwrap();
        let geometry = ImageGeometry { height: 32, width: 32, grid_thw: [1, 2, 2], patches: 4, tokens: 1 };
        let processed = profile.process_pair(&first, &second, &geometry).unwrap();
        assert_eq!(processed.pixel_values.len(), 4 * 1536);
        assert_eq!(processed.cu_seqlens, [0, 4]);
        let mean = profile.settings.pixel_mean[0];
        let std = profile.settings.pixel_std[0];
        // Patch 0 (top-left), channel 0, temporal slot 0, pixel (0,0) is the first source pixel.
        assert!((processed.pixel_values[0] - (0.0 - mean) / std).abs() < 1e-5);
        // Temporal slot 1 of the same patch holds the second frame.
        assert!((processed.pixel_values[256] - (100.0 - mean) / std).abs() < 1e-5);
        // Output pixel 8 of 32 has source coordinate 2/32*(8.5)-0.5 = 0.03125: 3.125% of the way.
        let expected = (255.0 * 0.03125 - mean) / std;
        assert!((processed.pixel_values[8] - expected).abs() < 1e-4, "{}", processed.pixel_values[8]);
    }

    /// Clips written on the fly: `ffmpeg` encodes lossless RGB, so decoded pixels are the
    /// frames this file generated and the loader's frame choice is visible in them.
    mod clip {
        use std::path::PathBuf;
        use std::process::Command;

        pub(super) struct Clip {
            pub(super) path: PathBuf,
            dir: PathBuf,
        }

        impl Drop for Clip {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.dir);
            }
        }

        fn scratch(name: &str) -> PathBuf {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_nanos());
            let dir = std::env::temp_dir().join(format!("flare-video-{}-{name}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create the clip directory");
            dir
        }

        fn encode(
            name: &str,
            width: usize,
            height: usize,
            fps: u32,
            frames: usize,
            codec: &[&str],
            extension: &str,
            pixel: impl Fn(usize, usize, usize) -> [u8; 3],
        ) -> Clip {
            let dir = scratch(name);
            let mut raw = Vec::with_capacity(width * height * 3 * frames);
            for frame in 0..frames {
                for y in 0..height {
                    for x in 0..width {
                        raw.extend_from_slice(&pixel(frame, x, y));
                    }
                }
            }
            let raw_path = dir.join("frames.rgb");
            std::fs::write(&raw_path, &raw).expect("write raw frames");
            let path = dir.join(format!("clip.{extension}"));
            let output = Command::new("ffmpeg")
                .args(["-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"])
                .args(["-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i"])
                .arg(&raw_path)
                .args(codec)
                .args(["-pix_fmt", "rgb24", "-y"])
                .arg(&path)
                .output()
                .unwrap_or_else(|error| panic!("cannot run ffmpeg to encode a test clip: {error}"));
            assert!(output.status.success(), "ffmpeg could not encode a test clip: {}", String::from_utf8_lossy(&output.stderr));
            let _ = std::fs::remove_file(&raw_path);
            Clip { path, dir }
        }

        /// An MP4 whose container states the frame count, encoded losslessly.
        pub(super) fn mp4(
            name: &str,
            width: usize,
            height: usize,
            fps: u32,
            frames: usize,
            pixel: impl Fn(usize, usize, usize) -> [u8; 3],
        ) -> Clip {
            encode(name, width, height, fps, frames, &["-c:v", "libx264rgb", "-qp", "0"], "mp4", pixel)
        }

        /// A Matroska clip, which keeps the frame count out of the stream: the container
        /// duration and the rate have to supply it.
        pub(super) fn mkv(
            name: &str,
            width: usize,
            height: usize,
            fps: u32,
            frames: usize,
            pixel: impl Fn(usize, usize, usize) -> [u8; 3],
        ) -> Clip {
            encode(name, width, height, fps, frames, &["-c:v", "ffv1"], "mkv", pixel)
        }

        /// A clip drawn by an ffmpeg filter source, for shapes too large to hand over as raw
        /// frames.
        pub(super) fn lavfi(name: &str, source: &str, seconds: u32, codec: &[&str], extension: &str) -> Clip {
            let dir = scratch(name);
            let path = dir.join(format!("clip.{extension}"));
            let output = Command::new("ffmpeg")
                .args(["-nostdin", "-v", "error", "-f", "lavfi", "-i"])
                .arg(format!("{source}:d={seconds}"))
                .args(codec)
                .arg("-y")
                .arg(&path)
                .output()
                .unwrap_or_else(|error| panic!("cannot run ffmpeg to generate a test clip: {error}"));
            assert!(output.status.success(), "ffmpeg could not generate a test clip: {}", String::from_utf8_lossy(&output.stderr));
            Clip { path, dir }
        }

        /// Copies `clip` with a display rotation of `degrees` (counter-clockwise, as ffmpeg
        /// states it), or nothing when this ffmpeg cannot write one.
        pub(super) fn rotate(clip: &Clip, degrees: &str) -> Option<Clip> {
            let dir = scratch(&format!("rotated{}", degrees.replace('-', "m")));
            let path = dir.join("rotated.mp4");
            let status = Command::new("ffmpeg")
                .args(["-nostdin", "-v", "error", "-display_rotation", degrees, "-i"])
                .arg(&clip.path)
                .args(["-c", "copy", "-y"])
                .arg(&path)
                .status()
                .ok()?;
            if !status.success() {
                return None;
            }
            let probe = Command::new("ffprobe")
                .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream_side_data=rotation", "-of", "json"])
                .arg(&path)
                .output()
                .ok()?;
            let side_data = String::from_utf8_lossy(&probe.stdout);
            side_data.contains("side_data_list").then_some(Clip { path, dir })
        }
    }

    #[test]
    fn decode_samples_uniform_frames_from_a_generated_clip() {
        let profile = profile(ImageLimits::default());
        // 90 frames at 30 fps: the loader keeps 32, uniformly spaced with both endpoints, and
        // each frame's colour names the frame it came from.
        let clip = clip::mkv("sampled", 320, 240, 30, 90, |frame, _, _| [frame as u8 * 2, 0, 0]);
        let bytes = std::fs::read(&clip.path).unwrap();
        let (frames, total) = profile.decode(&bytes, &VideoOptions::default()).unwrap();
        assert_eq!((total, frames.frames.len()), (90, 32));
        let expected = sample_indices(90, 32);
        for (frame, index) in frames.frames.iter().zip(&expected) {
            assert_eq!((frame.width(), frame.height()), (320, 240));
            assert_eq!(&frame.pixels()[..3], &[*index as u8 * 2, 0, 0], "frame {index}");
        }
        let description = profile.describe(&frames, total, &VideoOptions::default()).unwrap();
        // 240 rows break the tie towards 8 blocks, so frames resize to 256x320.
        assert_eq!(description.grid_thw, [16, 16, 20]);
        assert_eq!(description.tokens(), 16 * 80);
    }

    #[test]
    fn decode_reads_a_container_frame_count_and_pads_short_clips() {
        let profile = profile(ImageLimits::default());
        let clip = clip::mp4("short", 64, 48, 10, 5, |frame, _, _| [frame as u8 * 40, 5, 6]);
        let bytes = std::fs::read(&clip.path).unwrap();
        let (frames, total) = profile.decode(&bytes, &VideoOptions::default()).unwrap();
        assert_eq!((total, frames.frames.len()), (5, 5));
        for (index, frame) in frames.frames.iter().enumerate() {
            assert_eq!(&frame.pixels()[..3], &[index as u8 * 40, 5, 6]);
        }
        let description = profile.describe(&frames, total, &VideoOptions::default()).unwrap();
        // The odd count repeats the last frame, so five frames fill three temporal grids.
        assert_eq!(description.pairs[2].frames, [4, 4]);
        assert_eq!(description.grid_thw, [3, 6, 8]);
        assert_eq!(description.tokens(), 36);
    }

    #[test]
    fn decode_bounds_sampling_by_a_requested_rate() {
        let profile = profile(ImageLimits::default());
        let clip = clip::mp4("rate", 64, 48, 30, 90, |frame, _, _| [frame as u8 * 2, 0, 0]);
        let bytes = std::fs::read(&clip.path).unwrap();
        let options = VideoOptions { fps: Some(10.0), ..VideoOptions::default() };
        let (frames, _) = profile.decode(&bytes, &options).unwrap();
        // Three seconds at 10 fps is thirty frames, below the packaged count of 32.
        assert_eq!(frames.frames.len(), 30);
        assert_eq!(&frames.frames[1].pixels()[..3], &[sample_indices(90, 30)[1] as u8 * 2, 0, 0]);
    }

    #[test]
    fn decode_rotates_frames_to_their_display_orientation() {
        let profile = profile(ImageLimits::default());
        // A red top-left pixel names the coded layout. The reference loader, OpenCV's FFmpeg
        // backend, rotates frames by their display matrix: on this 64x32 clip it yields a
        // 32x64 frame with the red pixel at the bottom left for a 90-degree rotation, at the
        // top right for -90, and a 64x32 frame with it at the bottom right for 180 (observed
        // with opencv-python-headless 5.0.0 on these clips).
        let clip = clip::mp4("rotated", 64, 32, 10, 3, |_, x, y| if (x, y) == (0, 0) { [255, 0, 0] } else { [0, 0, 255] });
        for (degrees, extent, red, grid) in [
            ("90", (32, 64), (0, 63), [8, 4]),
            ("-90", (32, 64), (31, 0), [8, 4]),
            ("180", (64, 32), (63, 31), [4, 8]),
        ] {
            let Some(rotated) = clip::rotate(&clip, degrees) else {
                eprintln!("skipping rotation check: this ffmpeg cannot write display rotation");
                return;
            };
            let (frames, total) = profile.decode(&std::fs::read(&rotated.path).unwrap(), &VideoOptions::default()).unwrap();
            assert_eq!((total, frames.frames.len()), (3, 3));
            for frame in &frames.frames {
                assert_eq!((frame.width(), frame.height()), extent, "{degrees} degrees");
                let offset = (red.1 * frame.width() + red.0) * 3;
                assert_eq!(&frame.pixels()[offset..offset + 3], &[255, 0, 0], "{degrees} degrees");
                assert_eq!(frame.pixels().chunks_exact(3).filter(|pixel| *pixel == [255, 0, 0]).count(), 1, "{degrees} degrees");
            }
            // The described grid follows the rotated extent, scaled up to the minimum pixels.
            let description = profile.describe(&frames, total, &VideoOptions::default()).unwrap();
            assert_eq!(&description.grid_thw[1..], &grid, "{degrees} degrees");
        }
    }

    #[test]
    fn decode_rejects_media_without_a_video_stream() {
        let profile = profile(ImageLimits::default());
        assert!(profile.decode(b"not a container", &VideoOptions::default()).is_err());
    }

    #[test]
    fn decode_describes_a_thirty_second_720p_screen_recording() {
        let profile = profile(ImageLimits { max_patches_per_image: 65_536, ..ImageLimits::default() });
        // A 30 s 720p screen recording drawn by ffmpeg: 900 frames sample down to 32, 720 rows
        // align to the even 704, and each of the 16 grids carries 44 * 80 / 4 = 880 features.
        let clip = clip::lavfi(
            "screen",
            "testsrc2=s=1280x720:r=30",
            30,
            &["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p"],
            "mp4",
        );
        let (frames, total) = profile.decode(&std::fs::read(&clip.path).unwrap(), &VideoOptions::default()).unwrap();
        assert_eq!((total, frames.frames.len()), (900, 32));
        assert_eq!((frames.frames[0].width(), frames.frames[0].height()), (1280, 720));
        let description = profile.describe(&frames, total, &VideoOptions::default()).unwrap();
        assert_eq!(description.grid_thw, [16, 44, 80]);
        assert_eq!(description.tokens(), 14_080);
        assert_eq!(description.prompt_tokens(5), 14_194);
        // The sixteenth grid is labelled with the nominal time of its pair's first frame.
        assert_eq!(description.pairs[15].timestamp, "00:30");
    }
}
