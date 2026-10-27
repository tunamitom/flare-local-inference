//! Bandlimited sinc resampling with torchaudio's default kernel.
//!
//! `torchaudio.transforms.Resample(orig, new)` with `lowpass_filter_width=6`, `rolloff=0.99`
//! and a Hann window: the kernel is built in f64 and rounded to f32 as torchaudio does, then
//! applied as a strided convolution over the padded waveform. The reference loader applies the
//! same kernel (vLLM `resample_sinc`), so resampled samples agree to float32 accumulation order.

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Resamples `waveform` from `orig` to `new` Hz; equal rates return the input unchanged.
pub(super) fn resample(waveform: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new {
        return waveform.to_vec();
    }
    const LOWPASS_FILTER_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let divisor = gcd(orig as usize, new as usize);
    let orig = orig as usize / divisor;
    let new = new as usize / divisor;
    let base_freq = orig.min(new) as f64 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig as f64 / base_freq).ceil() as usize;
    let taps = 2 * width + orig;
    // kernels[phase][tap] in f32, as torchaudio stores them.
    let scale = base_freq / orig as f64;
    let kernels: Vec<Vec<f32>> = (0..new)
        .map(|phase| {
            let offset = -(phase as f64) / new as f64;
            (0..taps)
                .map(|tap| {
                    let idx = (tap as f64 - width as f64) / orig as f64;
                    let mut t = (offset + idx) * base_freq;
                    t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
                    let window = (t * std::f64::consts::PI / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
                    t *= std::f64::consts::PI;
                    let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                    (sinc * window * scale) as f32
                })
                .collect()
        })
        .collect();
    let length = waveform.len();
    let target = (new as f64 * length as f64 / orig as f64).ceil() as usize;
    // conv1d over the waveform padded with `width` zeros before and `width + orig` after, so
    // every stride-aligned window fits: floor(length / orig) + 1 of them.
    let blocks = length / orig + 1;
    let mut output = Vec::with_capacity(blocks * new);
    for block in 0..blocks {
        let start = block * orig; // index into the padded waveform
        for kernel in &kernels {
            let mut sum = 0.0f64;
            for (tap, weight) in kernel.iter().enumerate() {
                let padded = start + tap;
                if padded >= width && padded - width < length {
                    sum += f64::from(waveform[padded - width]) * f64::from(*weight);
                }
            }
            output.push(sum as f32);
        }
    }
    output.truncate(target);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_preserves_a_tone_and_the_torchaudio_output_length() {
        let orig = 48_000;
        let samples: Vec<f32> = (0..9600).map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / orig as f64).sin() as f32).collect();
        let out = resample(&samples, orig, 24_000);
        assert_eq!(out.len(), 4800);
        // Away from the edges the resampled tone equals the analytic signal at the new rate.
        for i in 100..4700 {
            let expected = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 24_000.0).sin() as f32;
            assert!((out[i] - expected).abs() < 2e-3, "sample {i}: {} vs {expected}", out[i]);
        }
        assert_eq!(resample(&samples, orig, orig).len(), samples.len());
        assert_eq!(resample(&samples[..7], 44_100, 24_000).len(), 4);
    }
}
