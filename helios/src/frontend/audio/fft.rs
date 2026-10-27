//! Mixed-radix complex FFT over f64 for one fixed analysis size.
//!
//! The plan factors its size into 4, 2, 3, 5 and any remaining primes and precomputes the
//! twiddles once. Frames are transformed in f64, more precisely than the float32 reference
//! spectrogram, so the comparison tolerance is the reference's own rounding.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }
}

pub(super) struct Plan {
    size: usize,
    factors: Vec<usize>,
    twiddles: Vec<Complex>,
}

impl Plan {
    pub(super) fn new(size: usize) -> Self {
        let size = size.max(1);
        let mut factors = Vec::new();
        let mut remaining = size;
        for radix in [4, 2, 3, 5] {
            while remaining % radix == 0 {
                factors.push(radix);
                remaining /= radix;
            }
        }
        let mut prime = 7;
        while remaining > 1 {
            while remaining % prime == 0 {
                factors.push(prime);
                remaining /= prime;
            }
            prime += 2;
        }
        let twiddles = (0..size)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / size as f64;
                Complex { re: angle.cos(), im: angle.sin() }
            })
            .collect();
        Self { size, factors, twiddles }
    }

    pub(super) fn size(&self) -> usize {
        self.size
    }

    /// Forward transform of `input` into `output`, both of the plan's size.
    pub(super) fn forward(&self, input: &[Complex], output: &mut [Complex]) {
        assert_eq!(input.len(), self.size);
        assert_eq!(output.len(), self.size);
        self.recurse(input, 1, output, 0);
    }

    /// Decimation in time: the `output.len()` samples of `input` read with `stride` are the
    /// residue classes of the factor at `level`.
    fn recurse(&self, input: &[Complex], stride: usize, output: &mut [Complex], level: usize) {
        let n = output.len();
        if n == 1 {
            output[0] = input[0];
            return;
        }
        let radix = self.factors[level];
        let m = n / radix;
        for r in 0..radix {
            self.recurse(&input[r * stride..], stride * radix, &mut output[r * m..(r + 1) * m], level + 1);
        }
        let step = self.size / n;
        let root = self.size / radix;
        let mut combined = vec![Complex::default(); n];
        let mut columns = vec![Complex::default(); radix];
        for k in 0..m {
            for r in 0..radix {
                columns[r] = output[r * m + k].mul(self.twiddles[(r * k * step) % self.size]);
            }
            for q in 0..radix {
                let mut sum = Complex::default();
                for r in 0..radix {
                    sum = sum.add(columns[r].mul(self.twiddles[(r * q * root) % self.size]));
                }
                combined[q * m + k] = sum;
            }
        }
        output.copy_from_slice(&combined);
    }
}

/// Magnitudes of the onesided spectrum of one real frame: `size / 2 + 1` values.
pub(super) fn magnitudes(plan: &Plan, frame: &[f64], spectrum: &mut [Complex], output: &mut [f64]) {
    let input: Vec<Complex> = frame.iter().map(|value| Complex { re: *value, im: 0.0 }).collect();
    plan.forward(&input, spectrum);
    for (bin, value) in output.iter_mut().enumerate() {
        let c = spectrum[bin];
        *value = (c.re * c.re + c.im * c.im).sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(input: &[Complex]) -> Vec<Complex> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let mut sum = Complex::default();
                for (j, x) in input.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                    sum = sum.add(x.mul(Complex { re: angle.cos(), im: angle.sin() }));
                }
                sum
            })
            .collect()
    }

    #[test]
    fn mixed_radix_matches_the_direct_transform() {
        for size in [1, 2, 3, 4, 5, 6, 8, 12, 15, 30, 64, 96, 120, 960, 1024, 1029] {
            let plan = Plan::new(size);
            let input: Vec<Complex> = (0..size)
                .map(|i| Complex {
                    re: ((i * 7919) % 101) as f64 / 50.0 - 1.0,
                    im: ((i * 104729) % 53) as f64 / 26.0 - 1.0,
                })
                .collect();
            let mut output = vec![Complex::default(); size];
            plan.forward(&input, &mut output);
            for (a, b) in output.iter().zip(naive(&input)) {
                assert!((a.re - b.re).abs() < 1e-8 && (a.im - b.im).abs() < 1e-8, "size {size}");
            }
        }
    }
}
