//! Audio decoding and log-mel preprocessing for prepared audio encoder graphs.
//!
//! The MiMo profile reproduces the reference processor's input pipeline: decoded channels are
//! averaged, the waveform is resampled to the encoder rate with torchaudio's sinc kernel, and a
//! centered reflect-padded STFT with a periodic Hann window feeds an HTK mel filterbank; the
//! log of the clamped magnitudes is the graph input. The encoder consumes fixed segments of
//! `segment_frames` mel frames, so a clip becomes one conditioned input per segment whose
//! feature count follows the encoder's convolution, pooling, and grouping arithmetic.

mod fft;
mod resample;
mod wav;

use sha2::{Digest, Sha256};

use crate::{Error, Failure, FrontendError, Result};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AudioSettings {
    Mimo(MimoAudioSettings),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoAudioSettings {
    pub sampling_rate: u32,
    pub n_fft: u32,
    pub hop_length: u32,
    pub window_length: u32,
    pub n_mels: u32,
    pub f_min: f32,
    pub f_max: Option<f32>,
    /// Mel frames one encoder graph consumes; longer clips are split into segments.
    pub segment_frames: u32,
    pub kernel_size: u32,
    pub stride: u32,
    pub pooling: u32,
    pub group_size: u32,
    /// Longest clip admitted, in seconds of audio.
    pub max_seconds: u32,
}

/// One encoder segment of a described clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AudioSegmentDescription {
    pub identity: [u8; 32],
    /// First mel frame of the segment within the clip.
    pub offset: usize,
    pub frames: usize,
    pub tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AudioDescription {
    pub sample_rate: u32,
    pub samples: usize,
    pub frames: usize,
    pub segments: Vec<AudioSegmentDescription>,
}

impl AudioDescription {
    pub fn tokens(&self) -> usize {
        self.segments.iter().map(|segment| segment.tokens).sum()
    }
}

/// Row-major log-mel frames of a whole clip at the encoder rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Mel {
    pub frames: usize,
    pub mels: usize,
    pub values: Vec<f32>,
}

/// One segment's graph input.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedAudio {
    pub frames: usize,
    pub mels: usize,
    pub values: Vec<f32>,
}

fn input(message: impl Into<String>) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message.into())))
}

/// `torch.linspace` in float32: the second half counts down from the end.
fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let halfway = steps / 2;
    (0..steps)
        .map(|i| if i < halfway { start + step * i as f32 } else { end - step * (steps - i - 1) as f32 })
        .collect()
}

pub struct AudioProfile {
    settings: MimoAudioSettings,
    window: Vec<f64>,
    /// `[n_freqs][n_mels]` HTK triangular filters without normalization.
    filterbank: Vec<f32>,
    plan: fft::Plan,
    identity_domain: [u8; 32],
    max_encoded_bytes: usize,
}

impl AudioProfile {
    pub fn load(settings: &AudioSettings, max_encoded_bytes: usize) -> Result<Self> {
        let AudioSettings::Mimo(settings) = settings;
        let s = settings;
        if s.sampling_rate == 0 || s.n_fft == 0 || s.hop_length == 0 || s.window_length == 0 || s.n_mels == 0
            || s.window_length != s.n_fft || s.n_fft % 2 != 0 || s.segment_frames == 0
            || s.kernel_size == 0 || s.stride == 0 || s.pooling == 0 || s.group_size == 0 || s.max_seconds == 0
            || !s.f_min.is_finite() || s.f_min < 0.0
            || s.f_max.is_some_and(|f| !f.is_finite() || f <= s.f_min)
            || max_encoded_bytes == 0
        {
            return Err(input("invalid prepared audio settings"));
        }
        let n = s.window_length as usize;
        let window = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
        let n_freqs = s.n_fft as usize / 2 + 1;
        let n_mels = s.n_mels as usize;
        let f_max = s.f_max.unwrap_or((s.sampling_rate / 2) as f32);
        let all_freqs = linspace_f32(0.0, (s.sampling_rate / 2) as f32, n_freqs);
        let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let m_pts = linspace_f32(hz_to_mel(f64::from(s.f_min)) as f32, hz_to_mel(f64::from(f_max)) as f32, n_mels + 2);
        let f_pts: Vec<f32> = m_pts.iter().map(|m| 700.0f32 * (10.0f32.powf(m / 2595.0) - 1.0)).collect();
        let f_diff: Vec<f32> = f_pts.windows(2).map(|p| p[1] - p[0]).collect();
        let mut filterbank = vec![0.0f32; n_freqs * n_mels];
        for (bin, freq) in all_freqs.iter().enumerate() {
            for mel in 0..n_mels {
                let down = -(f_pts[mel] - freq) / f_diff[mel];
                let up = (f_pts[mel + 2] - freq) / f_diff[mel + 1];
                filterbank[bin * n_mels + mel] = down.min(up).max(0.0);
            }
        }
        let identity_domain = Sha256::digest(serde_json::to_vec(settings).map_err(|e| input(e.to_string()))?).into();
        Ok(Self {
            settings: settings.clone(),
            window,
            filterbank,
            plan: fft::Plan::new(s.n_fft as usize),
            identity_domain,
            max_encoded_bytes,
        })
    }

    pub fn settings(&self) -> &MimoAudioSettings {
        &self.settings
    }

    pub fn max_encoded_bytes(&self) -> usize {
        self.max_encoded_bytes
    }

    pub fn max_seconds(&self) -> usize {
        self.settings.max_seconds as usize
    }

    /// Features one encoder segment of `frames` mel frames produces.
    pub fn tokens(&self, frames: usize) -> usize {
        let s = &self.settings;
        let (k, stride, pooling, group) = (s.kernel_size as usize, s.stride as usize, s.pooling as usize, s.group_size as usize);
        let n = frames + 3 - k;
        let n = (n + 2 - k) / stride + 1;
        let n = n.div_ceil(pooling);
        n.div_ceil(group)
    }

    /// Codes per second of audio before grouping, as the API documents it.
    pub fn tokens_per_second(&self) -> f64 {
        let s = &self.settings;
        f64::from(s.sampling_rate) / f64::from(s.hop_length) / f64::from(s.stride) / f64::from(s.pooling) / f64::from(s.group_size)
    }

    fn max_samples(&self, sample_rate: u32) -> usize {
        self.settings.max_seconds as usize * sample_rate as usize
    }

    /// Decodes a clip to mono float samples at its native rate: WAV natively, other
    /// containers through ffmpeg, averaging channels like the reference loader.
    pub fn decode(&self, media_type: &str, bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
        if bytes.len() > self.max_encoded_bytes {
            return Err(input("encoded audio exceeds the configured byte limit"));
        }
        if matches!(media_type, "wav" | "x-wav" | "wave" | "vnd.wave") || bytes.starts_with(b"RIFF") {
            let wave = wav::decode(bytes, self.max_samples(48_000).max(1))?;
            if wave.samples.len() > self.max_samples(wave.sample_rate) {
                return Err(input(format!(
                    "audio is {:.1} s long; the maximum is {} s",
                    wave.samples.len() as f64 / f64::from(wave.sample_rate),
                    self.settings.max_seconds
                )));
            }
            return Ok((wave.samples, wave.sample_rate));
        }
        let staged = super::ffmpeg::temp_file(bytes)?;
        let stream = super::ffmpeg::probe_stream(&staged.path, "a", "sample_rate,channels,duration")?;
        let sample_rate = super::ffmpeg::integer(&stream["sample_rate"])
            .filter(|rate| (1..=1_000_000).contains(rate))
            .ok_or_else(|| input("audio stream has no usable sample rate"))? as u32;
        let channels = super::ffmpeg::integer(&stream["channels"])
            .filter(|channels| (1..=64).contains(channels))
            .ok_or_else(|| input("audio stream has no usable channel count"))? as usize;
        if let Some(duration) = super::ffmpeg::float(&stream["duration"]) {
            if duration > self.settings.max_seconds as f64 + 1.0 {
                return Err(input(format!("audio is {duration:.1} s long; the maximum is {} s", self.settings.max_seconds)));
            }
        }
        let limit = self.max_samples(sample_rate) * channels * 4;
        let mut command = std::process::Command::new("ffmpeg");
        command
            .args(["-nostdin", "-v", "error", "-threads", "2", "-i"])
            .arg(&staged.path)
            .args(["-vn", "-sn", "-dn", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]);
        let pcm = super::ffmpeg::run_bounded(command, "ffmpeg", limit)?;
        drop(staged);
        let frames = pcm.len() / (4 * channels);
        if frames == 0 {
            return Err(input("audio stream decoded to no samples"));
        }
        let scale = 1.0 / channels as f32;
        let samples = pcm[..frames * 4 * channels]
            .chunks_exact(4 * channels)
            .map(|frame| {
                let sum: f32 = frame.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).sum();
                if channels == 1 { sum } else { sum * scale }
            })
            .collect();
        Ok((samples, sample_rate))
    }

    /// Resamples mono `samples` to the encoder rate and computes the clip's log-mel frames.
    pub fn log_mel(&self, samples: &[f32], sample_rate: u32) -> Result<Mel> {
        let s = &self.settings;
        if sample_rate == 0 || samples.len() > self.max_samples(sample_rate) {
            return Err(input(format!("audio exceeds the maximum of {} s", s.max_seconds)));
        }
        let samples = resample::resample(samples, sample_rate, s.sampling_rate);
        let (n_fft, hop) = (s.n_fft as usize, s.hop_length as usize);
        let pad = n_fft / 2;
        if samples.len() <= pad {
            return Err(input(format!(
                "audio must be longer than {} samples at {} Hz",
                pad,
                s.sampling_rate
            )));
        }
        // torch.stft(center=True, pad_mode="reflect") frames the reflect-padded waveform.
        let padded: Vec<f64> = (0..samples.len() + 2 * pad)
            .map(|i| {
                let index = if i < pad {
                    pad - i
                } else if i < pad + samples.len() {
                    i - pad
                } else {
                    2 * samples.len() - 2 - (i - pad)
                };
                f64::from(samples[index])
            })
            .collect();
        let frames = 1 + (padded.len() - n_fft) / hop;
        let n_freqs = n_fft / 2 + 1;
        let n_mels = s.n_mels as usize;
        let mut values = Vec::with_capacity(frames * n_mels);
        let mut frame = vec![0.0f64; n_fft];
        let mut spectrum = vec![fft::Complex::default(); n_fft];
        let mut magnitude = vec![0.0f64; n_freqs];
        for t in 0..frames {
            for (i, value) in frame.iter_mut().enumerate() {
                *value = padded[t * hop + i] * self.window[i];
            }
            fft::magnitudes(&self.plan, &frame, &mut spectrum, &mut magnitude);
            for mel in 0..n_mels {
                let mut sum = 0.0f64;
                for bin in 0..n_freqs {
                    sum += magnitude[bin] * f64::from(self.filterbank[bin * n_mels + mel]);
                }
                values.push((sum as f32).max(1e-7).ln());
            }
        }
        Ok(Mel { frames, mels: n_mels, values })
    }

    pub fn segment_frames(&self, frames: usize) -> Vec<(usize, usize)> {
        let segment = self.settings.segment_frames as usize;
        let mut result = Vec::with_capacity(frames.div_ceil(segment));
        let mut offset = 0;
        while offset < frames {
            let count = segment.min(frames - offset);
            result.push((offset, count));
            offset += count;
        }
        result
    }

    fn segment_identity(&self, frames: usize, values: &[f32]) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"flare-conditioned-audio-v1\0");
        hash.update(self.identity_domain);
        hash.update((frames as u64).to_le_bytes());
        for value in values {
            hash.update(value.to_le_bytes());
        }
        hash.finalize().into()
    }

    pub fn describe(&self, mel: &Mel, sample_rate: u32, samples: usize) -> Result<AudioDescription> {
        if mel.mels != self.settings.n_mels as usize || mel.values.len() != mel.frames * mel.mels || mel.frames == 0 {
            return Err(input("audio features differ from the prepared mel geometry"));
        }
        let segments = self
            .segment_frames(mel.frames)
            .into_iter()
            .map(|(offset, frames)| AudioSegmentDescription {
                identity: self.segment_identity(frames, &mel.values[offset * mel.mels..(offset + frames) * mel.mels]),
                offset,
                frames,
                tokens: self.tokens(frames),
            })
            .collect();
        Ok(AudioDescription { sample_rate, samples, frames: mel.frames, segments })
    }

    /// One segment's graph input, checked against its description.
    pub fn segment(&self, mel: &Mel, segment: &AudioSegmentDescription) -> Result<ProcessedAudio> {
        let end = segment.offset.checked_add(segment.frames).filter(|end| *end <= mel.frames).ok_or_else(|| input("audio segment exceeds its clip"))?;
        let values = &mel.values[segment.offset * mel.mels..end * mel.mels];
        if self.segment_identity(segment.frames, values) != segment.identity || self.tokens(segment.frames) != segment.tokens {
            return Err(input("audio source differs from its prompt description"));
        }
        Ok(ProcessedAudio { frames: segment.frames, mels: mel.mels, values: values.to_vec() })
    }
}

#[cfg(test)]
pub(crate) fn test_settings() -> AudioSettings {
    AudioSettings::Mimo(MimoAudioSettings {
        sampling_rate: 24_000, n_fft: 960, hop_length: 240, window_length: 960, n_mels: 128, f_min: 0.0, f_max: None,
        segment_frames: 6000, kernel_size: 3, stride: 2, pooling: 2, group_size: 4, max_seconds: 3600,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_arithmetic_follows_the_encoder_convolutions() {
        let profile = AudioProfile::load(&test_settings(), 1 << 20).unwrap();
        // 2 s at 24 kHz: 201 frames -> 101 -> 51 codes -> 13 groups.
        assert_eq!(profile.tokens(201), 13);
        assert_eq!(profile.tokens(6000), 375);
        assert_eq!(profile.tokens(6001), 376);
        assert_eq!(profile.tokens(1), 1);
        assert!((profile.tokens_per_second() - 6.25).abs() < 1e-9);
        assert_eq!(profile.segment_frames(6001), vec![(0, 6000), (6000, 1)]);
        assert_eq!(profile.segment_frames(12_000), vec![(0, 6000), (6000, 6000)]);
    }

    #[test]
    fn wav_clip_becomes_log_mel_segments_with_stable_identities() {
        let profile = AudioProfile::load(&test_settings(), 1 << 20).unwrap();
        let rate = 48_000u32;
        let tone: Vec<i16> = (0..2 * rate as usize)
            .map(|i| ((2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate)).sin() * 12_000.0) as i16)
            .collect();
        let bytes = wav::encode_pcm16(rate, &[tone.clone(), tone]);
        let (samples, decoded_rate) = profile.decode("wav", &bytes).unwrap();
        assert_eq!((samples.len(), decoded_rate), (96_000, rate));
        let mel = profile.log_mel(&samples, decoded_rate).unwrap();
        assert_eq!((mel.frames, mel.mels), (201, 128));
        assert!(mel.values.iter().all(|v| v.is_finite()));
        // 440 Hz sits at 550 mel, band 21 of 128 on the HTK scale up to 12 kHz.
        let frame = &mel.values[100 * 128..101 * 128];
        let peak = frame.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!((20..=22).contains(&peak), "peak mel band {peak}");
        assert!(frame[120] < frame[peak]);
        let description = profile.describe(&mel, decoded_rate, samples.len()).unwrap();
        assert_eq!(description.segments.len(), 1);
        assert_eq!(description.segments[0].tokens, 13);
        let processed = profile.segment(&mel, &description.segments[0]).unwrap();
        assert_eq!(processed.values.len(), 201 * 128);
        let mut altered = description.segments[0];
        altered.identity[0] ^= 1;
        assert!(profile.segment(&mel, &altered).is_err());
        assert!(profile.log_mel(&samples[..400], decoded_rate).is_err());
    }
}
