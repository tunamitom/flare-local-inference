//! RIFF WAVE decoding to float mono at the file's native sample rate.
//!
//! PCM integer samples of 8, 16, 24 and 32 bits and IEEE float samples of 32 and 64 bits are
//! accepted, including WAVE_FORMAT_EXTENSIBLE headers naming those subformats. Channels are
//! averaged after conversion to float, as the reference loader averages decoded channels
//! before resampling.

use crate::{Error, Failure, FrontendError, Result};

pub(super) struct Waveform {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

fn invalid(message: &str) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(format!("invalid WAV: {message}"))))
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(bytes.get(at..at + 2).ok_or_else(|| invalid("truncated header"))?.try_into().unwrap()))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(bytes.get(at..at + 4).ok_or_else(|| invalid("truncated header"))?.try_into().unwrap()))
}

/// Decodes `bytes`, rejecting more than `max_samples` frames per channel before allocating.
pub(super) fn decode(bytes: &[u8], max_samples: usize) -> Result<Waveform> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid("missing RIFF WAVE header"));
    }
    let mut at = 12;
    let mut format: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4)? as usize;
        let body_start = at + 8;
        let body_end = body_start.checked_add(size).ok_or_else(|| invalid("chunk size overflows"))?;
        let body = bytes.get(body_start..body_end.min(bytes.len())).ok_or_else(|| invalid("truncated chunk"))?;
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(invalid("short fmt chunk"));
                }
                let mut tag = u16_at(body, 0)?;
                let channels = u16_at(body, 2)?;
                let rate = u32_at(body, 4)?;
                let bits = u16_at(body, 14)?;
                if tag == 0xFFFE {
                    // WAVE_FORMAT_EXTENSIBLE: the subformat GUID's first two bytes carry the tag.
                    if body.len() < 40 {
                        return Err(invalid("short extensible fmt chunk"));
                    }
                    tag = u16_at(body, 24)?;
                }
                format = Some((tag, channels, rate, bits));
            }
            b"data" => {
                if body_end > bytes.len() {
                    return Err(invalid("data chunk exceeds the file"));
                }
                data = Some(body);
            }
            _ => {}
        }
        at = body_end + (size & 1);
    }
    let (tag, channels, rate, bits) = format.ok_or_else(|| invalid("missing fmt chunk"))?;
    let data = data.ok_or_else(|| invalid("missing data chunk"))?;
    if channels == 0 || rate == 0 {
        return Err(invalid("zero channels or sample rate"));
    }
    let width = match (tag, bits) {
        (1, 8) | (1, 16) | (1, 24) | (1, 32) | (3, 32) | (3, 64) => bits as usize / 8,
        _ => return Err(invalid("unsupported sample format; use PCM 8, 16, 24 or 32-bit or IEEE float")),
    };
    let frame_bytes = width * channels as usize;
    let frames = data.len() / frame_bytes;
    if frames == 0 {
        return Err(invalid("no audio samples"));
    }
    if frames > max_samples {
        return Err(Error::frontend(FrontendError::Input(Failure::new(format!(
            "audio has {frames} samples at {rate} Hz; the maximum is {max_samples}"
        )))));
    }
    let mut samples = Vec::with_capacity(frames);
    let scale = 1.0 / channels as f32;
    for frame in data[..frames * frame_bytes].chunks_exact(frame_bytes) {
        let mut sum = 0.0f32;
        for sample in frame.chunks_exact(width) {
            sum += match (tag, bits) {
                (1, 8) => (sample[0] as f32 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0,
                (1, 24) => {
                    let value = i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) >> 8;
                    value as f32 / 8388608.0
                }
                (1, 32) => i32::from_le_bytes(sample.try_into().unwrap()) as f32 / 2147483648.0,
                (3, 32) => f32::from_le_bytes(sample.try_into().unwrap()),
                (3, 64) => f64::from_le_bytes(sample.try_into().unwrap()) as f32,
                _ => unreachable!(),
            };
        }
        samples.push(if channels == 1 { sum } else { sum * scale });
    }
    Ok(Waveform { sample_rate: rate, channels, samples })
}

#[cfg(test)]
pub(crate) fn encode_pcm16(sample_rate: u32, channels: &[Vec<i16>]) -> Vec<u8> {
    let frames = channels[0].len();
    let count = channels.len() as u16;
    let data_len = frames * 2 * count as usize;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2 * count as u32).to_le_bytes());
    out.extend_from_slice(&(2 * count).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for frame in 0..frames {
        for channel in channels {
            out.extend_from_slice(&channel[frame].to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_stereo_averages_channels_and_scales_to_unit() {
        let left = vec![0i16, 16384, -32768, 32767];
        let right = vec![0i16, 0, -32768, -32767];
        let bytes = encode_pcm16(48_000, &[left, right]);
        let wave = decode(&bytes, 1 << 20).unwrap();
        assert_eq!((wave.sample_rate, wave.channels), (48_000, 2));
        assert_eq!(wave.samples.len(), 4);
        assert!((wave.samples[1] - 0.25).abs() < 1e-7);
        assert!((wave.samples[2] + 1.0).abs() < 1e-7);
        assert!(wave.samples[3].abs() < 1e-4);
        assert!(decode(&bytes, 3).is_err());
        assert!(decode(&bytes[..20], 1 << 20).is_err());
        assert!(decode(b"RIFF\0\0\0\0WAVE", 1 << 20).is_err());
    }
}
