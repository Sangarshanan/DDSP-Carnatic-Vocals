import os
import glob
import torch
import torchaudio
import torchcrepe
import numpy as np
import torchaudio.functional as AF
from torch.utils.data import Dataset


SAMPLE_RATE = 16000
HOP_LENGTH = 256  # 16ms frames
CHUNK_SECONDS = 4.0
FMIN, FMAX = (
    80.0,
    1000.0,
)  # f0 range


def _extract_f0(waveform: torch.Tensor, sr: int, hop_length: int, device: str = "cpu"):
    """
    Run CREPE once over a full clip; returns (f0_hz, periodicity) per analysis frame.
    """
    audio = waveform.unsqueeze(0).to(device)  # (1, T)
    f0, periodicity = torchcrepe.predict(
        audio,
        sample_rate=sr,
        hop_length=hop_length,
        fmin=FMIN,
        fmax=FMAX,
        model="full",
        batch_size=1024,
        device=device,
        return_periodicity=True,
    )
    # Smooth the f0 curves to reduce frame to frame jitter
    periodicity = torchcrepe.filter.median(periodicity, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    return f0.squeeze(0).cpu().numpy(), periodicity.squeeze(0).cpu().numpy()


class CarnaticDataset(Dataset):
    def __init__(
        self,
        data_dir="A",
        sample_rate=SAMPLE_RATE,
        duration=CHUNK_SECONDS,
        hop_length=HOP_LENGTH,
        cache_dir=None,
        device="cpu",
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.target_length = int(sample_rate * duration)
        self.device = device

        # Cached so it's only computed once and not every epoch.
        self.cache_dir = cache_dir or os.path.join(data_dir, ".f0_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.filepaths = sorted(glob.glob(os.path.join(data_dir, "*.mp3")))

    def __len__(self):
        return len(self.filepaths)

    def _load_full(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0)

    def _f0_cache_path(self, path):
        base = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(self.cache_dir, base + ".npz")

    def _get_f0(self, path, waveform):
        cache_path = self._f0_cache_path(path)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data["f0"], data["periodicity"]
        f0, periodicity = _extract_f0(
            waveform, self.sample_rate, self.hop_length, device=self.device
        )
        np.savez(cache_path, f0=f0, periodicity=periodicity)
        return f0, periodicity

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        waveform = self._load_full(path)
        f0, periodicity = self._get_f0(path, waveform)

        n_frames_per_chunk = self.target_length // self.hop_length

        # Pad short clips (audio + aligned f0) up to at least one full chunk.
        if waveform.shape[0] < self.target_length:
            pad = self.target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        if len(f0) < n_frames_per_chunk + 1:
            frame_pad = n_frames_per_chunk + 1 - len(f0)
            f0 = np.pad(f0, (0, frame_pad), mode="edge")
            periodicity = np.pad(periodicity, (0, frame_pad))

        # Random crop, aligned to a hop boundary so audio & f0 frames stay in sync.
        max_start_frame = max(0, len(f0) - n_frames_per_chunk - 1)
        start_frame = (
            np.random.randint(0, max_start_frame + 1) if max_start_frame > 0 else 0
        )
        start_sample = start_frame * self.hop_length

        audio_chunk = waveform[start_sample : start_sample + self.target_length]
        if audio_chunk.shape[0] < self.target_length:
            audio_chunk = torch.nn.functional.pad(
                audio_chunk, (0, self.target_length - audio_chunk.shape[0])
            )

        f0_chunk = f0[start_frame : start_frame + n_frames_per_chunk]
        periodicity_chunk = periodicity[start_frame : start_frame + n_frames_per_chunk]
        if len(f0_chunk) < n_frames_per_chunk:
            frame_pad = n_frames_per_chunk - len(f0_chunk)
            f0_chunk = np.pad(f0_chunk, (0, frame_pad), mode="edge")
            periodicity_chunk = np.pad(periodicity_chunk, (0, frame_pad))

        peak = audio_chunk.abs().max().clamp(min=1e-8)
        audio_chunk = audio_chunk / peak * 0.9

        return {
            "audio": audio_chunk,
            "f0": torch.from_numpy(f0_chunk.astype(np.float32)),
            "periodicity": torch.from_numpy(periodicity_chunk.astype(np.float32)),
        }
