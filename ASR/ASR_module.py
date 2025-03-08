# import whisper
# import faiss
# import torch
# import sounddevice as sd
# import numpy as np
# import wave
# from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer
# import os
# import time


# # Load Whisper model (using 'small' for efficiency)
# asr_model = whisper.load_model("small")

# # Load embedding model for memory storage
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize FAISS index for vector storage
# embedding_dim = 384  # Must match MiniLM embedding size
# index = faiss.IndexFlatL2(embedding_dim)

# def record_audio(filename, duration=10, samplerate=16000):
#     """
#     Records audio from the microphone and saves it as a WAV file.
#     Args:
#         filename (str): Path to save the recorded audio.
#         duration (int): Duration of recording in seconds.
#         samplerate (int): Sampling rate for audio recording.
#     """
#     print("Recording...")
#     audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
#     #audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
#     #print(type(audio_data))
#     sd.wait()
#     print("Recording finished.")
    
#     # with wave.open(filename, "wb") as wf:
#     #     wf.setnchannels(1)
#     #     wf.setsampwidth(2)
#     #     wf.setframerate(samplerate)
#     #     wf.writeframes(audio_data.tobytes())

#     return audio_data

# def transcribe_audio(audio):
#     """
#     Transcribes speech from an audio file using Whisper ASR.
#     Args:
#         audio_path (str): Path to the audio file.
#     Returns:
#         str: Transcribed text.
#     """
    
#     # #print current working directory
#     # if not os.path.exists(audio_path):
#     #     print("Audio file does not exist: ", audio_path)
#     # else:
#     #     print("Audio file exists: ", audio_path)
    
#     #only have the last part of the file path

#     # root = os.getcwd()
#     # audio_path = root + "/" + audio_path

#     result = asr_model.transcribe(audio)
#     return result["text"]

# def store_transcription_in_memory(text):
#     """
#     Converts transcribed text into an embedding and stores it in FAISS.
#     Args:
#         text (str): Transcribed speech text.
#     """
#     embedding = embedding_model.encode([text])
#     embedding = np.array(embedding).astype('float32')
#     index.add(embedding)  # Store in FAISS
#     print("Stored in Memory Module:", text)

# # Test


# path = "ASR/audio.wav"
    

    

# audio = record_audio(path)

# print(len(audio))
# time.sleep(1)
# if os.path.exists(path):
#     print("Audio file exists")
# else:
#     print("Audio file does not exist")
# transcription = transcribe_audio(audio)

# if transcription:
#     store_transcription_in_memory(transcription)


import whisper
import faiss
import torch
import sounddevice as sd
import numpy as np
import wave
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

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

def transcribe_audio(audio_path):
    """
    Transcribes speech from an audio file using Whisper ASR.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """
    result = asr_model.transcribe(audio_path)
    return result["text"]

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

# Test
audio_file = "recorded_audio.wav"
record_audio(audio_file)
transcription = transcribe_audio(audio_file)
if transcription:
    store_transcription_in_memory(transcription)
