# DDSP Carnatic

Differentiable Digital Signal Processing (DDSP) implementation for Carnatic vocal synthesis. It has two main components: a `DDSPEncoder` that extracts frame-wise timbre controls (amplitude, harmonic distribution, and noise magnitude) along with extracted f0 contours using CREPE, and a `DDSPSynth` that reconstructs audio from those controls using differentiable harmonic and filtered-noise oscillators plus a learned reverb module.

Trained end-to-end on a [KritiSamhita: Carnatic Tonic Recognition Dataset](https://data.mendeley.com/datasets/nkdm57hvw3/2) using multi-scale spectral loss, the model learns to separate pitch (an external signal) from timbre (the learned neural content), allowing it to synthesize vocals with new pitch contours while preserving the acoustic character of a reference recording.

### Installation

```bash
pip install -r requirements.txt
```

### Training

Place `.mp3` training samples in `audio/` and run:

```bash
python train.py
```

### Running the Gradio App

```bash
python app.py
```
