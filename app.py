from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.io.wavfile import read
from transformers import pipeline
from scipy.signal import resample

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

@socketio.on('audio')
def handle_audio(payload):
    # Decode the MULAW-encoded data
    audio_data = np.frombuffer(payload, dtype=np.uint8)
    audio_data = np.sign(audio_data - 127.5) * (1/256.) * ((audio_data - 127.5) ** 2)
    # Normalize the audio data
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    # Transcribe the audio data using the Whisper ASR model
    transcription = transcriber({"sampling_rate": 8000, "raw": audio_data})
    emit('transcription', transcription)

if __name__ == '__main__':
    socketio.run(app)