import logging
import traceback
import numpy as np
import torch
import torchaudio
import torchcrepe
import gradio as gr

from src.dataset import SAMPLE_RATE, HOP_LENGTH
from src.encoder import DDSPEncoder
from src.synth import DDSPSynth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ddsp-app")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "output/encoder.pt"
CREPE_MODEL = "tiny"

# Load models once at startup
log.info("Loading models onto %s...", DEVICE)
encoder = DDSPEncoder(sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH).to(DEVICE)
synth = DDSPSynth(sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH).to(DEVICE)

_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
encoder.load_state_dict(_ckpt["encoder"])
synth.load_state_dict(_ckpt["synth"])
encoder.eval()
synth.eval()

torchcrepe.load.model(device=str(DEVICE), capacity=CREPE_MODEL)
log.info("Models loaded successfully.")

_resamplers = {}

def _resample(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SAMPLE_RATE: return x
    if sr_in not in _resamplers:
        _resamplers[sr_in] = torchaudio.transforms.Resample(sr_in, SAMPLE_RATE)
    return _resamplers[sr_in](torch.from_numpy(x).float().unsqueeze(0)).squeeze(0).numpy()

def _normalize(x: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    if len(x) == 0: return x
    x = x - np.mean(x)
    peak = float(np.abs(x).max())
    return x / peak * target_peak if peak > 1e-4 else x

# Inference
@torch.inference_mode()
def convert_file(audio_input):
    if not audio_input:
        return None
        
    try:
        # 1. Preprocess: audio_input is now a filepath string
        waveform, sr_in = torchaudio.load(audio_input)
        
        # Convert to mono 1D numpy float array
        data = waveform.mean(dim=0).numpy()
        
        if len(data) == 0:
            return SAMPLE_RATE, np.zeros((0,), dtype=np.float32)
            
        data = _normalize(_resample(data, sr_in))

        # Pad to exactly match model's hop length to prevent tensor shape mismatches
        pad_len = (HOP_LENGTH - (len(data) % HOP_LENGTH)) % HOP_LENGTH
        if pad_len > 0:
            data = np.pad(data, (0, pad_len))
            
        audio = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)

        # Extract Pitch
        f0 = torchcrepe.predict(
            audio,
            sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=80.0,
            fmax=1200.0,
            model=CREPE_MODEL,
            batch_size=2048,
            device=str(DEVICE),
            return_periodicity=False,
        )
        
        # Generate Controls & Synthesize
        controls = encoder(audio, f0)
        out = synth(controls, n_samples=audio.shape[-1])
        
        # Postprocess
        result = out.squeeze(0).float().cpu().numpy()
        result = np.clip(_normalize(result), -1.0, 1.0).astype(np.float32)
        
        return SAMPLE_RATE, result

    except Exception:
        log.error("Conversion failed:\n%s", traceback.format_exc())
        raise

# UI
with gr.Blocks(title="Carnatic Vocal (DDSP)") as demo:
    gr.Markdown("# 🎵 Carnatic Vocal Converter (DDSP)")

    with gr.Tab("📁 Upload"):
        with gr.Row():
            file_input = gr.Audio(sources=["upload"], type="filepath", label="Upload audio")
            file_output = gr.Audio(label="Converted Vocal Output", autoplay=True)
        gr.Button("Convert", variant="primary").click(
            fn=convert_file, inputs=[file_input], outputs=[file_output]
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
