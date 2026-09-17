# DDSP with Carnatic Vocals

Differentiable Digital Signal Processing (DDSP) implementation for Carnatic vocal synthesis. It has two main components: a `DDSPEncoder` that extracts frame-wise timbre controls (amplitude, harmonic distribution, and noise magnitude) along with extracted f0 contours using CREPE, and a `DDSPSynth` that reconstructs audio from those controls using differentiable harmonic and filtered-noise oscillators plus a learned reverb module.

Trained end-to-end on a [KritiSamhita: Carnatic Tonic Recognition Dataset](https://data.mendeley.com/datasets/nkdm57hvw3/2) using multi-scale spectral loss, the model learns to separate pitch (an external signal) from timbre (the learned neural content), allowing it to synthesize vocals with new pitch contours while preserving the acoustic character of a reference recording.

Our model is based out of two components, First we have the `DDSPSynth` which is made up of 3 components:

- **Harmonic oscillator:** An additive synthesizer that sums sinusoids at integer multiples of f0.
- **Filtered noise synth:** To model noisy vocal elements, we filter white noise in the STFT domain with a predicted magnitude envelope while preserving its random phase prior to inverse transformation.
- **Trainable FIR reverb:** Learned room impulse response applied via FFT convolution added on top of the dry harmonic + noise signal cause real recordings always have room.

Together, these help model expressive carnatic vocals. Next is the `DDSPEncoder` and since f0 is extracted up front by CREPE, our encoder only learns timbre. We use the log-mel spectrogram of the input audio and create a timbre representation with three output heads:

- **Amplitude:** Overall loudness per frame. *it's softplus, non-negative*
- **Harmonic distribution:**  Softmax over every bin (80 bins) of harmonic amplitudes
- **Noise magnitudes:** per-frame filtered-noise spectral envelope for non tonal content. *it's a sigmoid*

These 3 control signals map into the Synthesizer we defined earlier and we slowly train the network to tune the synth and make it sound closer to the training audio. After every epoch we use a **multi-scale spectral loss** which is just the STFT magnitude differences (both linear and log-scale) computed at multiple FFT sizes and then backpropagate through our entirely differentiable synthesizer, slowly reducing the loss and tuning our synth to the training dataset.

### Installation

Pretty straightforward, just create a virtual environment and install all the requirement necessary

```bash
uv venv --python python3.13 # or whatever floats your env boat
pip install -r requirements.txt
```

### Running the Gradio App

There is a gradio app that runs inference using the model, you can test it by uploading any audio, more harmonic content will yield better results because (DDSP) and also the training is on a vocal datasets.

```bash
python app.py
```

A bit of how the interface looks like, Inference should be pretty fast.

![alt text](https://github.com/Sangarshanan/DDSP-Carnatic-Vocals/blob/main/output/gradio.png "Gradio")


### Training the model

To train the model yourself, either with more data or with different parameters, more epochs and whatever random stuff you wanna do, Just put all the audio `.mp3` files in `audio/` directory for training and run the script

```bash
python src/train.py
```
