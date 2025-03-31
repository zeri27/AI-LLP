import re
import os
import threading
import queue
from gtts import gTTS
import pygame


def split_sentences(text):
    """
    Splits text into sentences based on punctuation.
    It considers both English punctuation (. ! ?) and Chinese punctuation (。！？).
    """
    # Pattern: split right after any of these punctuation marks.
    pattern = r'(?<=[。.!?])'
    sentences = re.split(pattern, text)
    # Remove empty sentences and strip whitespace.
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def is_chinese(text):
    """
    Determines if the sentence is Chinese.
    If any Chinese character is present, it returns True.
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def synthesize_sentence(sentence, lang, filename):
    """
    Synthesizes the given sentence to speech using gTTS and saves it as an MP3 file.
    """
    tts = gTTS(text=sentence, lang=lang)
    tts.save(filename)


def tts_pipeline(transcript):
    """
    Processes the transcript:
      - Splits it into sentences using punctuation (for both English and Chinese).
      - Clears the output folder before starting multithreading.
      - Synthesizes each sentence into an MP3 file (saved in the 'output' folder).
      - Plays each audio file as soon as it's ready using Pygame.

    Synthesis and playback run concurrently.
    Additionally, a message is printed whenever a new sentence's language differs from the previous one.
    """
    sentences = split_sentences(transcript)
    audio_queue = queue.Queue()

    # Create the output folder if it doesn't exist.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Clear the output folder BEFORE starting any threads.
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except PermissionError as e:
                print(f"Could not remove {file_path}: {e}")

    # Quit then reinitialize the Pygame mixer to ensure no files are locked.
    pygame.mixer.quit()
    pygame.mixer.init()

    def synthesis_worker():
        prev_lang = None
        for i, sentence in enumerate(sentences):
            current_lang = "zh-cn" if is_chinese(sentence) else "en"
            if prev_lang is None or current_lang != prev_lang:
                lang_str = "Chinese" if current_lang == "zh-cn" else "English"
                print(f"\n--- Starting new sentence in {lang_str} ---")
            prev_lang = current_lang

            filename = os.path.join(output_dir, f"sentence_{i}.mp3")
            print(f"Synthesizing (lang={current_lang}): {sentence}")
            synthesize_sentence(sentence, current_lang, filename)
            audio_queue.put(filename)
        # Signal that synthesis is complete.
        audio_queue.put(None)

    def playback_worker():
        while True:
            audio_file = audio_queue.get()
            if audio_file is None:
                break
            abs_audio_file = os.path.abspath(audio_file)
            print(f"Playing: {abs_audio_file}")
            pygame.mixer.music.load(abs_audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            # File is kept for debugging; to remove it after playback, uncomment:
            # os.remove(audio_file)

    # Run synthesis and playback concurrently.
    synth_thread = threading.Thread(target=synthesis_worker)
    play_thread = threading.Thread(target=playback_worker)
    synth_thread.start()
    play_thread.start()
    synth_thread.join()
    play_thread.join()


# if __name__ == "__main__":
#     text_input = "Hello, this is a test. 你好，这是一个测试。I love pancakes! 再见？"
#     print("Running TTS pipeline...")
#     tts_pipeline(text_input)