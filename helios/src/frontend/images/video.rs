//! MiMo video preprocessing: sampled frames become timestamped frame pairs for the vision graph.
//!
//! The reference serving path (the vLLM MiMo Omni processor behind its OpenCV loader) samples
//! `num_frames` frames uniformly over the clip, hands them to the processor with one-second
//! nominal timestamps, caps each frame's area so the whole clip stays within
//! `total_max_pixels`, pads the frame count to the temporal patch size by repeating the last
//! frame, resizes every frame with float bilinear interpolation, standardizes with ImageNet
//! statistics on the 0..255 scale, and packs consecutive frame pairs into the vision encoder's
//! temporal patches. Each pair is one vision graph run whose features follow a `MM:SS`
//! timestamp inside the video envelope.

use sha2::{Digest, Sha256};

use super::{ImageGeometry, ImageLimits, MERGE, PATCH, ProcessedImages, RgbImage, check_input_extent};
use crate::{Error, Failure, FrontendError, Result};

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
    pub total_max_pixels: usize,
    /// Frames sampled from a clip when the request names no count.
    pub num_frames: usize,
    pub max_frames: usize,
    pub pixel_mean: [f32; 3],
    pub pixel_std: [f32; 3],
}

/// Per-request sampling controls carried on a video content part.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VideoOptions {
    pub num_frames: Option<usize>,
    pub fps: Option<f64>,
    pub max_pixels: Option<usize>,
}

/// One frame pair of a described clip.
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
    pub total_frames: usize,
    pub sampled_frames: usize,
    pub pairs: Vec<VideoPairDescription>,
}

impl VideoDescription {
    pub fn tokens(&self) -> usize {
        self.pairs.iter().map(|pair| pair.geometry.tokens).sum()
    }
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

/// The vLLM loader's uniform frame selection: `numpy.linspace(0, total - 1, count, dtype=int)`.
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

/// Decoded frames of a request's video may occupy at most this many bytes.
const MAX_DECODED_BYTES: usize = 1 << 30;

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

    /// Frames sampled from a clip of `total_frames` at `source_fps`, following the reference
    /// loader: the request's count or the packaged default, further bounded by a requested
    /// sampling rate, and never more than the packaged maximum.
    pub fn sampled_count(&self, total_frames: usize, source_fps: f64, options: &VideoOptions) -> Result<usize> {
        if total_frames == 0 {
            return Err(input("video contains no frames"));
        }
        if let Some(count) = options.num_frames {
            if count == 0 || count > self.settings.max_frames {
                return Err(input(format!("num_frames must be from 1 through {}", self.settings.max_frames)));
            }
        }
        if let Some(fps) = options.fps {
            if !fps.is_finite() || fps <= 0.0 {
                return Err(input("fps must be a positive number"));
            }
        }
        let mut count = total_frames;
        match (options.num_frames, options.fps) {
            (Some(frames), _) => count = count.min(frames),
            (None, None) => count = count.min(self.settings.num_frames),
            (None, Some(_)) => {}
        }
        if let Some(fps) = options.fps {
            if source_fps > 0.0 {
                let duration = total_frames as f64 / source_fps;
                count = count.min((duration * fps).floor() as usize);
            }
        }
        let count = count.max(1);
        if count > self.settings.max_frames {
            return Err(input(format!(
                "video sampling selects {count} frames; the maximum is {} (lower fps or set num_frames)",
                self.settings.max_frames
            )));
        }
        Ok(count)
    }

    /// Resized extent of every frame of a `sampled` frame clip.
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
        // Per-pair area shrinks so the whole clip stays within total_max_pixels.
        let per_frame = (s.total_max_pixels * 2 / sampled.max(1)).min(max_pixels).max(s.min_pixels);
        let (resized_height, resized_width) = smart_resize(height, width, PATCH * MERGE, s.min_pixels, per_frame)?;
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

    fn pair_identity(&self, geometry: &ImageGeometry, first: &RgbImage, second: &RgbImage) -> [u8; 32] {
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

    /// Describes sampled frames as padded, timestamped pairs; the pair at index `j` carries the
    /// nominal time `2 * j` seconds, the processor's one-second frame spacing.
    pub fn describe(&self, frames: &VideoFrames, total_frames: usize, options: &VideoOptions) -> Result<VideoDescription> {
        let sampled = frames.frames.len();
        let first = frames.frames.first().ok_or_else(|| input("video contains no frames"))?;
        if frames.frames.iter().any(|frame| frame.width() != first.width() || frame.height() != first.height()) {
            return Err(input("video frames must share one extent"));
        }
        let geometry = self.frame_geometry(first.width(), first.height(), sampled, options)?;
        let padded = sampled.div_ceil(2) * 2;
        let pairs = (0..padded / 2)
            .map(|pair| {
                let a = (2 * pair).min(sampled - 1);
                let b = (2 * pair + 1).min(sampled - 1);
                VideoPairDescription {
                    identity: self.pair_identity(&geometry, &frames.frames[a], &frames.frames[b]),
                    frames: [a, b],
                    geometry,
                    timestamp: format_timestamp(2.0 * pair as f64),
                }
            })
            .collect();
        Ok(VideoDescription { width: first.width(), height: first.height(), total_frames, sampled_frames: sampled, pairs })
    }

    /// Decodes a container to the frames the reference loader would sample, through ffmpeg.
    pub fn decode(&self, bytes: &[u8], options: &VideoOptions) -> Result<(VideoFrames, usize)> {
        if bytes.len() > self.limits.max_encoded_bytes {
            return Err(input("encoded video exceeds the configured byte limit"));
        }
        let staged = super::super::ffmpeg::temp_file(bytes)?;
        let stream = super::super::ffmpeg::probe_stream(
            &staged.path, "v", "nb_frames,avg_frame_rate,r_frame_rate,duration,width,height",
        )?;
        let width = super::super::ffmpeg::integer(&stream["width"]).ok_or_else(|| input("video stream has no extent"))? as usize;
        let height = super::super::ffmpeg::integer(&stream["height"]).ok_or_else(|| input("video stream has no extent"))? as usize;
        check_input_extent(self.limits, width, height)?;
        let fps = super::super::ffmpeg::rational(&stream["avg_frame_rate"])
            .filter(|fps| *fps > 0.0)
            .or_else(|| super::super::ffmpeg::rational(&stream["r_frame_rate"]).filter(|fps| *fps > 0.0))
            .unwrap_or(0.0);
        let total_frames = match super::super::ffmpeg::integer(&stream["nb_frames"]).filter(|n| *n > 0) {
            Some(count) => count as usize,
            None => {
                let duration = super::super::ffmpeg::float(&stream["duration"]).unwrap_or(0.0);
                (duration * fps + 0.5).floor() as usize
            }
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
        let select = indices.iter().map(|i| format!("eq(n\\,{i})")).collect::<Vec<_>>().join("+");
        let mut command = std::process::Command::new("ffmpeg");
        command
            .args(["-nostdin", "-v", "error", "-threads", "4", "-i"])
            .arg(&staged.path)
            .args(["-an", "-sn", "-dn", "-vf"])
            .arg(format!("select='{select}'"))
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

    /// Resizes, standardizes, and packs one frame pair into vision patches.
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
            pixel_values: Vec::with_capacity(patches * 3 * 2 * PATCH * PATCH),
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

    #[test]
    fn frame_sampling_matches_the_reference_loader() {
        // numpy.linspace(0, 89, 8, dtype=int) and the all-frames and single-frame cases.
        assert_eq!(sample_indices(90, 8), vec![0, 12, 25, 38, 50, 63, 76, 89]);
        assert_eq!(sample_indices(5, 32), vec![0, 1, 2, 3, 4]);
        assert_eq!(sample_indices(100, 1), vec![0]);
        assert_eq!(sample_indices(1000, 32)[1], 32);
        let profile = VideoProfile::load(&test_settings(), ImageLimits::default()).unwrap();
        let default = VideoOptions::default();
        assert_eq!(profile.sampled_count(90, 30.0, &default).unwrap(), 32);
        assert_eq!(profile.sampled_count(20, 30.0, &default).unwrap(), 20);
        let fps = VideoOptions { fps: Some(1.0), ..default };
        assert_eq!(profile.sampled_count(90, 30.0, &fps).unwrap(), 3);
        let both = VideoOptions { num_frames: Some(2), fps: Some(1.0), max_pixels: None };
        assert_eq!(profile.sampled_count(90, 30.0, &both).unwrap(), 2);
        assert!(profile.sampled_count(90, 30.0, &VideoOptions { num_frames: Some(0), ..default }).is_err());
        assert!(profile.sampled_count(90, 30.0, &VideoOptions { num_frames: Some(5000), ..default }).is_err());
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(62.0), "01:02");
        assert_eq!(format_timestamp(3599.9), "59:59");
    }

    #[test]
    fn pair_geometry_follows_the_total_pixel_budget_and_pads_odd_counts() {
        let profile = VideoProfile::load(&test_settings(), ImageLimits { max_patches_per_image: 65_536, ..ImageLimits::default() }).unwrap();
        let default = VideoOptions::default();
        // 1080p keeps its area at 32 frames: 1920x1088 after alignment.
        let geometry = profile.frame_geometry(1920, 1080, 32, &default).unwrap();
        assert_eq!((geometry.width, geometry.height, geometry.patches, geometry.tokens), (1920, 1088, 8160, 2040));
        // 512 frames cap each pair at 2 * 268,435,456 / 512 = 1,048,576 pixels.
        let geometry = profile.frame_geometry(1920, 1080, 512, &default).unwrap();
        assert!(geometry.width * geometry.height <= 1_048_576);
        assert_eq!((geometry.width, geometry.height), (1344, 768));
        let capped = profile.frame_geometry(1920, 1080, 32, &VideoOptions { max_pixels: Some(921_600), ..default }).unwrap();
        assert!(capped.width * capped.height <= 921_600);
        assert!(profile.frame_geometry(1920, 1080, 32, &VideoOptions { max_pixels: Some(100), ..default }).is_err());
        let frames = VideoFrames {
            frames: (0..3).map(|i| RgbImage::new(64, 48, vec![i * 40; 64 * 48 * 3]).unwrap()).collect(),
        };
        let description = profile.describe(&frames, 90, &default).unwrap();
        assert_eq!(description.pairs.len(), 2);
        assert_eq!(description.pairs[1].frames, [2, 2]);
        assert_eq!(description.pairs[0].timestamp, "00:00");
        assert_eq!(description.pairs[1].timestamp, "00:02");
        // 64x48 frames are below min_pixels and scale up to 128x96: a 6x8 patch grid.
        assert_eq!(description.pairs[0].geometry.grid_thw, [1, 6, 8]);
        assert_ne!(description.pairs[0].identity, description.pairs[1].identity);
        assert_eq!(description.tokens(), 24);
    }

    #[test]
    fn bilinear_pairs_pack_temporal_slots_from_distinct_frames() {
        let profile = VideoProfile::load(&test_settings(), ImageLimits::default()).unwrap();
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
}
