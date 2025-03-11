import whisper
import faiss
import torch
import sounddevice as sd
import numpy as np
import wave
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from ASR.AER_module import predict_emotion

# Load Whisper model (using 'small' for efficiency)
asr_model = whisper.load_model("small")

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

def transcribe_audio(audio_path):

    #
    emotion = predict_emotion(audio_path)

    
    """
    Transcribes speech from an audio file using Whisper ASR.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """
    
    result = asr_model.transcribe(audio_path)

    # Append predicted emotion to the transcribed text such that the NLP module can process it
    result = "<PREDICTED EMOTION = " + emotion + "> " + result["text"]

    return result

