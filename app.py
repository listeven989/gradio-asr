import gradio as gr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import numpy as np
import librosa

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("openai/whisper-base.en")
model = Wav2Vec2ForCTC.from_pretrained("openai/whisper-base.en")

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Resample the audio to 16kHz
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sr = 16000

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    input_values = processor(stream, sampling_rate=sr, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return stream, transcription

demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch()