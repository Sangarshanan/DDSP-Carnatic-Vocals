import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio

from dataset import CarnaticDataset, SAMPLE_RATE, HOP_LENGTH, CHUNK_SECONDS
from encoder import DDSPEncoder
from synth import DDSPSynth


def _spectral_loss(
    pred: torch.Tensor, target: torch.Tensor, fft_sizes: tuple = (2048, 1024, 512, 256)
) -> torch.Tensor:
    """Multi-scale spectral distance (linear + log magnitude)"""
    loss = pred.new_zeros(1)
    for n in fft_sizes:
        hop = max(n // 4, 1)
        win = torch.hann_window(n, device=pred.device)

        def S(x, n=n, hop=hop, win=win):
            # x is (B, T)
            return torch.stft(
                x,
                n_fft=n,
                hop_length=hop,
                win_length=n,
                window=win,
                return_complex=True,
            ).abs()

        Sp, St = S(pred), S(target)
        loss = loss + (Sp - St).abs().mean()
        loss = loss + (Sp.clamp(1e-7).log() - St.clamp(1e-7).log()).abs().mean()
    return loss


def train():
    """
    We Train!!
    """
    device = torch.device("cuda:1")
    print(f"Using device: {device}")
    print(f"CUDA device {device} available: {torch.cuda.is_available()}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Dataset and Dataloader. Short (4s) random crops instead of one static 20s
    print("\n=== Loading Dataset ===")
    dataset = CarnaticDataset(
        data_dir="audio",
        sample_rate=SAMPLE_RATE,
        duration=CHUNK_SECONDS,
        hop_length=HOP_LENGTH,
        device=str(device),
    )

    print(f"Dataset size: {len(dataset)} clips")
    if len(dataset) == 0:
        print("Warning: No .mp3 files found in audio/. Please add data to train.")
        return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
    print(f"Dataloader batches per epoch: {len(dataloader)}")

    # Models
    print("\n=== Initializing Models ===")
    encoder = DDSPEncoder(sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH).to(device)
    synth = DDSPSynth(sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH).to(device)

    n_encoder_params = sum(p.numel() for p in encoder.parameters())
    n_synth_params = sum(p.numel() for p in synth.parameters())
    print(f"Encoder params: {n_encoder_params:,}")
    print(f"Synth params: {n_synth_params:,}")
    print(f"Total params: {n_encoder_params + n_synth_params:,}")

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(synth.parameters()), lr=0.001
    )
    print(f"Optimizer: Adam(lr=0.001)")

    n_epochs = 100
    print(f"\n=== Starting Training ({n_epochs} epochs) ===\n")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)  # (B, T)
            f0_hz = batch["f0"].to(device)  # (B, n_frames)

            optimizer.zero_grad()

            # Predict per-frame timbre controls (pitch comes from CREPE, not the encoder)
            controls = encoder(audio, f0_hz)

            # Synthesize
            pred_audio = synth(controls, n_samples=audio.shape[-1])  # (B, T)

            # Loss
            loss = _spectral_loss(pred_audio, audio)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_steps += 1

            # Per-batch logging
            if batch_idx % max(1, len(dataloader) // 4) == 0:
                print(
                    f"  Epoch {epoch:3d}/{n_epochs} | "
                    f"Batch {batch_idx:3d}/{len(dataloader)} | "
                    f"Loss: {batch_loss:.6f}"
                )

            if batch_idx == 0 and epoch % 25 == 0:
                # Save the first sample of the batch for inspection
                sample_pred = pred_audio[0].detach().cpu().unsqueeze(0)
                sample_target = audio[0].detach().cpu().unsqueeze(0)

                torchaudio.save(
                    os.path.join(output_dir, f"epoch_{epoch}_pred.wav"),
                    sample_pred,
                    SAMPLE_RATE,
                )
                torchaudio.save(
                    os.path.join(output_dir, f"epoch_{epoch}_target.wav"),
                    sample_target,
                    SAMPLE_RATE,
                )
                print(f"  → Saved audio samples to {output_dir}/")

        if len(dataloader) > 0:
            avg_loss = epoch_loss / len(dataloader)
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Batches: {epoch_steps}\n"
            )

    if len(dataloader) > 0:
        # Save model
        torch.save(
            {"encoder": encoder.state_dict(), "synth": synth.state_dict()},
            os.path.join(output_dir, "encoder.pt"),
        )
        print("\n=== Training Complete ===")
        print(f"Model saved to {os.path.join(output_dir, 'encoder.pt')}")
        print(f"Audio samples saved to {output_dir}/")


if __name__ == "__main__":
    train()
