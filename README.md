# AI-LLP
AI Language Learning Application for Conversational Agents Course TU Delft

When having any trouble with running the code, please contact one of us for assistance:
- Yizhen Zang tmedcz@outlook.com
- Liwia Padowska liwia.padowska@gmail.com
- Jordy del Castilho jordydelcastilho@gmail.com
- Zeryab Alam zeryabalam272@icloud.com

## Introduction

This repository contains the code for a project developing a conversational agent that can help with learning languages. 
The project comprises of several modules that work together:

- ASR module for speech recognition
- TTS module for text-to-speech
- NLP module for natural language processing
- MEM module for memory




# tools required

- FFMPEG  (https://www.gyan.dev/ffmpeg/builds/ for windows, https://www.ffmpeg.org/download.html for mac or linux)
    make sure to add ffmpeg to environment variables after installing
- Ollama (https://ollama.com/) 
    make sure to pull the models you want to use. check your gpu's max memory to find the models that can be loaded onto your gpu
- python 3.12 or higher (https://www.python.org/downloads/)


# python packages


- langchain
- transformers
- pytorch
- gtts
- pygame
- faiss-cpu (or faiss-gpu if you have a linux distribution that supports it)
- langchain_ollama
- openai-whisper
- torchaudio
- sounddevice
- sentence_transformers
- librosa
- noisereduce
- webrtcvad


```bash
pip install langchain transformers pytorch gtts pygame faiss-cpu langchain_ollama openai-whisper torchaudio sounddevice sentence_transformers librosa noisereduce webrtcvad
```


