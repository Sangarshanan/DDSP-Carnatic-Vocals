import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class DDSPEncoder(nn.Module):
    """Frame-wise encoder producing DDSP control signals

    This network only has to learn timbre: for every analysis
    frame it outputs an overall loudness/amplitude, a normalized harmonic
    amplitude distribution, and a filtered-noise magnitude spectrum.
    
    Every frame gets its own output, so the resynthesis can track
    fast pitch/loudness/timbre movement for gamakas.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
        n_harmonics: int = 80,
        n_noise_bands: int = 257,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands

        in_dim = n_mels + 1  # + log-f0 channel
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.amp_head = nn.Linear(hidden_dim, 1)
        self.harmonic_head = nn.Linear(hidden_dim, n_harmonics)
        self.noise_head = nn.Linear(hidden_dim, n_noise_bands)

    def forward(self, audio: torch.Tensor, f0_hz: torch.Tensor) -> dict:
        """
        audio: input audio waveform (B, T)
        f0_hz: f0 contour at the same hop rate as the mel spec

        Returns a dict of per-frame control signals:
          amplitude
          harmonic_distribution
          noise_magnitudes
          f0_hz
        """
        mel = self.mel_spec(audio)  # (B, n_mels, n_frames_mel)
        mel = torch.log(mel.clamp(min=1e-5))
        mel = mel.transpose(1, 2)  # (B, n_frames_mel, n_mels)

        n_frames = min(mel.shape[1], f0_hz.shape[1])
        mel = mel[:, :n_frames, :]
        f0_hz = f0_hz[:, :n_frames]

        log_f0 = torch.log(f0_hz.clamp(min=1.0)).unsqueeze(-1)  # (B, n_frames, 1)
        x = torch.cat([mel, log_f0], dim=-1)

        x = self.input_proj(x)
        x, _ = self.gru(x)
        x = self.out_proj(x)

        amplitude = F.softplus(self.amp_head(x)).squeeze(-1)  # (B, n_frames)
        harmonic_distribution = torch.softmax(self.harmonic_head(x), dim=-1)
        noise_magnitudes = torch.sigmoid(self.noise_head(x))

        return {
            "amplitude": amplitude,
            "harmonic_distribution": harmonic_distribution,
            "noise_magnitudes": noise_magnitudes,
            "f0_hz": f0_hz,
        }
