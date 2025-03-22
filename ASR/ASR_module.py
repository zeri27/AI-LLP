import numpy as np
import wave
import whisper
import librosa
import webrtcvad
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
from ASR.AER_module import predict_emotion

# Load Whisper model (using 'small' for efficiency)
asr_model = whisper.load_model("small")

def listen_for_speech(sample_rate=16000, frame_duration_ms=30, silence_duration_ms=1500):
    """
    Continuously listens for speech and returns recorded audio when detected.
    Stops recording after 'silence_duration_ms' of silence.
    """
    vad = webrtcvad.Vad(0)  # Moderate aggressiveness
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    silence_frames = int(silence_duration_ms / frame_duration_ms)
    audio = []
    silence_counter = 0
    recording = False

    print("Listening for speech... (Speak anytime)")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
        while True:
            frame = stream.read(frame_size)[0]
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)

            if is_speech:
                audio.append(frame)
                silence_counter = 0
                recording = True
            elif recording:
                silence_counter += 1
                audio.append(frame)
                if silence_counter > silence_frames:
                    print("Stopped listening due to silence.")
                    break

    if recording:
        return b''.join([f.tobytes() for f in audio])  # Return raw audio bytes
    return None

def save_audio(audio_bytes, filename, sample_rate=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

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

    cleaned_audio = reduce_noise(audio_path, "audio_cleaned.wav")
    result = asr_model.transcribe(cleaned_audio)

    result = "<Predicted Emotion = " + emotion + "> " + result["text"]
    return result

# # Test
# audio_bytes = listen_for_speech()

# if audio_bytes:
#     audio_file = "audio_recorded.wav"
#     save_audio(audio_bytes, audio_file)
    
#     transcription = transcribe_audio(audio_file)
#     print("\nTranscription Output:")
#     print(transcription)

# else:
#     print("No speech detected. Please try again.")
