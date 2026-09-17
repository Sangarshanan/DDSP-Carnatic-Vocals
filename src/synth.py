import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import SAMPLE_RATE


def _upsample_to_audio_rate(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Linearly interpolate a (B, n_frames, C) control signal to (B, n_samples, C)."""
    x = x.transpose(1, 2)
    x = F.interpolate(x, size=n_samples, mode="linear", align_corners=True)
    return x.transpose(1, 2)


class HarmonicOscillator(nn.Module):
    """
    Additive synthesizer: Sum of harmonically-related sinusoids.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        f0_hz: torch.Tensor,  # (B, n_frames)
        harmonic_distribution: torch.Tensor,  # (B, n_frames, n_harmonics), sums to 1
        amplitude: torch.Tensor,  # (B, n_frames)
        n_samples: int,
    ) -> torch.Tensor:
        n_harmonics = harmonic_distribution.shape[-1]
        harmonics = torch.arange(
            1, n_harmonics + 1, device=f0_hz.device, dtype=f0_hz.dtype
        )

        f0_up = _upsample_to_audio_rate(f0_hz.unsqueeze(-1), n_samples).squeeze(
            -1
        )  # (B, T)
        amp_up = _upsample_to_audio_rate(amplitude.unsqueeze(-1), n_samples).squeeze(
            -1
        )  # (B, T)
        dist_up = _upsample_to_audio_rate(
            harmonic_distribution, n_samples
        )  # (B, T, n_harmonics)

        harmonic_freqs = f0_up.unsqueeze(-1) * harmonics.view(
            1, 1, -1
        )  # (B, T, n_harmonics)

        # Anti-alias: zero out partials above Nyquist.
        anti_alias_mask = (harmonic_freqs < (self.sample_rate / 2.0)).to(dist_up.dtype)
        dist_up = dist_up * anti_alias_mask
        dist_up = dist_up / dist_up.sum(dim=-1, keepdim=True).clamp(min=1e-7)

        phase = 2.0 * math.pi * torch.cumsum(harmonic_freqs / self.sample_rate, dim=1)
        signal = (torch.sin(phase) * dist_up).sum(dim=-1)  # (B, T)
        return signal * amp_up


class FilteredNoiseSynth(nn.Module):
    """
    Time-varying filtered-noise synthesizer via STFT magnitude shaping.

    A white-noise excitation is passed through an STFT; each frame's magnitude
    is replaced by the (per-frame, learned) target magnitude while keeping the
    noise's own random phase, then inverted. This realizes a cheap but
    effective time-varying FIR filter driven by the encoder and is what 
    lets the model reproduce breath/consonant noise instead of only tonal content.
    """

    def __init__(self, hop_length: int = 256):
        super().__init__()
        self.hop_length = hop_length

    def forward(self, noise_magnitudes: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Noise_magnitudes: (B, n_frames, n_mags), n_mags = n_fft // 2 + 1
        """
        B, n_frames, n_mags = noise_magnitudes.shape
        n_fft = 2 * (n_mags - 1)
        device = noise_magnitudes.device

        noise = torch.rand(B, n_samples, device=device) * 2.0 - 1.0
        window = torch.hann_window(n_fft, device=device)
        noise_stft = torch.stft(
            noise,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
        )  # (B, n_mags, n_frames_stft)

        n_frames_stft = noise_stft.shape[-1]
        mag = noise_magnitudes.transpose(1, 2)  # (B, n_mags, n_frames)
        mag = F.interpolate(mag, size=n_frames_stft, mode="linear", align_corners=True)

        unit_phase = noise_stft / noise_stft.abs().clamp(min=1e-7)
        shaped_stft = unit_phase * mag

        return torch.istft(
            shaped_stft,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            length=n_samples,
        )


class TrainableFIRReverb(nn.Module):
    """
    A single learned room impulse response, applied via FFT convolution.

    Global (not per-frame) reverb: Real recordings include room/mic coloration
    that a dry oscillator+noise model can't match.

    Without this, the spectral loss keeps a floor no matter how good the
    dry synthesis is.
    """

    def __init__(self, ir_seconds: float = 1.0, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        ir_len = int(ir_seconds * sample_rate)
        init = torch.randn(ir_len) * torch.linspace(1.0, 0.0, ir_len).pow(4.0) * 0.1
        init[0] = 1.0  # dry impulse at t=0 so training doesn't start all-wet
        self.ir = nn.Parameter(init)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        n_samples = signal.shape[-1]
        ir_len = self.ir.shape[0]
        n_fft = 1
        while n_fft < n_samples + ir_len - 1:
            n_fft *= 2

        S = torch.fft.rfft(signal, n=n_fft)
        H = torch.fft.rfft(self.ir, n=n_fft).unsqueeze(0)
        wet = torch.fft.irfft(S * H, n=n_fft)[..., :n_samples]
        return wet


class DDSPSynth(nn.Module):
    """Combines harmonic + filtered-noise synthesizers (+ optional reverb)."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = 256,
        use_reverb: bool = True,
    ):
        super().__init__()
        self.harmonic = HarmonicOscillator(sample_rate=sample_rate)
        self.noise = FilteredNoiseSynth(hop_length=hop_length)
        self.reverb = (
            TrainableFIRReverb(sample_rate=sample_rate) if use_reverb else None
        )

    def forward(self, controls: dict, n_samples: int) -> torch.Tensor:
        harmonic_signal = self.harmonic(
            controls["f0_hz"],
            controls["harmonic_distribution"],
            controls["amplitude"],
            n_samples,
        )
        noise_signal = self.noise(controls["noise_magnitudes"], n_samples)
        dry = harmonic_signal + noise_signal
        return self.reverb(dry) if self.reverb is not None else dry
