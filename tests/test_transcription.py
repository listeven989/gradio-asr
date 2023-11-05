import requests
from gtts import gTTS
import io
import subprocess
import socketio

def convert_to_mulaw(input_file, output_file):
    command = ['ffmpeg', '-i', input_file, '-f', 'mulaw', '-ar', '8000', '-ac', '1', output_file]
    subprocess.run(command, check=True)

# Use the function in your test
def test_transcribe():
    # Create an audio file with the phrase "hello"
    print("Creating audio file...")
    tts = gTTS('hello')
    tts.save('hello.mp3')
    
    # Convert the audio file to MULAW format
    print("Converting audio to MULAW...")
    convert_to_mulaw('hello.mp3', 'hello.mulaw')

    # Load the MULAW audio file and convert it to bytes
    print("Converting audio to bytes...")
    with open('hello.mulaw', 'rb') as f:
        mulaw_bytes = f.read()

    # Connect to the server
    print("Connecting to server...")
    sio = socketio.Client()

    # Define a handler for the 'transcription' event
    @sio.on('transcription')
    def on_transcription(transcription):
        print("Received transcription:", transcription)
        assert transcription["text"].lower().replace(" ", "") == 'hello'
        print("Test passed!")

    # Connect to the server
    sio.connect('http://localhost:5000')

    # Send the audio data
    print("Transcribing...")
    sio.emit('audio', mulaw_bytes)

    # Wait for the transcription to be received
    sio.wait()

test_transcribe()