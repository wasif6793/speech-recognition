import streamlit as st
import torch
import torchaudio
from transformers import pipeline
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import tempfile
import os

class SpeechRecognizer:
    def __init__(self, model_name="openai/whisper-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"🔊 Loading model `{model_name}` on {self.device}...")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        )
        self.resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
        st.success("✅ Model loaded successfully!")

    def transcribe_file(self, file):
        # Save uploaded file to a temp location
        temp_path = tempfile.mktemp()
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Check file type
        if file.name.lower().endswith(".mp3"):
            audio = AudioSegment.from_mp3(temp_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            os.remove(temp_path)
            return self.pipe({"raw": samples, "sampling_rate": 16000})["text"]
        else:
            waveform, sample_rate = torchaudio.load(temp_path)
            os.remove(temp_path)
            if sample_rate != 16000:
                waveform = self.resampler(waveform)
            return self.pipe(waveform.numpy())["text"]

    def transcribe_microphone(self, duration):
        st.info(f"🎤 Recording from microphone for {duration} seconds...")
        recording = sd.rec(
            int(duration * 16000),
            samplerate=16000,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio_data = np.squeeze(recording)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        return self.pipe({"raw": audio_data, "sampling_rate": 16000})["text"]

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="🎙️ Whisper Speech-to-Text", layout="centered")
st.title("🎙️ Speech Recognition using Whisper")

recognizer = SpeechRecognizer()

option = st.radio("Choose input type:", ("Upload Audio File", "Record from Microphone"))

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"])
    if uploaded_file is not None:
        with st.spinner("Transcribing..."):
            result = recognizer.transcribe_file(uploaded_file)
            st.text_area("📝 Transcription", result, height=200)

elif option == "Record from Microphone":
    duration = st.slider("Recording duration (seconds)", 1, 20, 5)
    if st.button("🎤 Record and Transcribe"):
        with st.spinner("Recording and transcribing..."):
            result = recognizer.transcribe_microphone(duration)
            st.text_area("📝 Transcription", result, height=200)
