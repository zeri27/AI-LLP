import whisper
import faiss
import torch
import sounddevice as sd
import numpy as np
import wave
import librosa
import noisereduce as nr
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import soundfile as sf
from ASR.AER_module import predict_emotion

# Load Whisper model (using 'small' for efficiency)
asr_model = whisper.load_model("small")

# Load embedding model for memory storage
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index for vector storage
embedding_dim = 384  # Must match MiniLM embedding size
index = faiss.IndexFlatL2(embedding_dim)

def record_audio(filename, duration=10, samplerate=16000):
    """
    Records audio from the microphone and saves it as a WAV file.
    Args:
        filename (str): Path to save the recorded audio.
        duration (int): Duration of recording in seconds.
        samplerate (int): Sampling rate for audio recording.
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording finished.")
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

def reduce_noise(input_audio, output_audio):
    """
    Applies noise reduction to an audio file and saves the cleaned version.
    Args:
        input_audio (str): Path to the input noisy audio file.
        output_audio (str): Path to save the cleaned audio file.
    """
    y, sr = librosa.load(input_audio, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_audio, reduced_noise, sr)
    return output_audio

def transcribe_audio(audio_path):
    """
    Transcribes speech from an audio file using Whisper ASR after noise reduction.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """
    emotion = predict_emotion(audio_path)

    cleaned_audio = reduce_noise(audio_path, "cleaned_audio.wav")
    result = asr_model.transcribe(cleaned_audio)

    result = "<PREDICTED EMOTION = " + emotion + "> " + result["text"]
    return result

def store_transcription_in_memory(text):
    """
    Converts transcribed text into an embedding and stores it in FAISS.
    Args:
        text (str): Transcribed speech text.
    """
    embedding = embedding_model.encode([text])
    embedding = np.array(embedding).astype('float32')
    index.add(embedding)  # Store in FAISS
    print("Stored in Memory Module:", text)

# # Test
# audio_file = "recorded_audio.wav"
# record_audio(audio_file)
# transcription = transcribe_audio(audio_file)
# if transcription:
#     store_transcription_in_memory(transcription)
