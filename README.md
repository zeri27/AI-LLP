# AI-LLP
AI Language Learning Application for Conversational Agents Course at TU Delft. 

When having any trouble with running the code, please contact one of us for assistance:
- Jordy del Castilho jordydelcastilho@gmail.com
- Liwia Padowska liwia.padowska@gmail.com
- Yizhen Zang yizhenzang@tudelft.nl
- Zeryab Alam zeryabalam272@icloud.com


# Introduction

This repository contains the code for a project developing a conversational agent that can help with language learning. 
The project comprises several modules that work together:

- ASR module for speech recognition
- TTS module for text-to-speech
- NLP module for natural language processing
- MEM module for memory


# Tools required

- FFMPEG  (https://www.gyan.dev/ffmpeg/builds/ for windows, https://www.ffmpeg.org/download.html for mac or linux)
    Make sure to add ffmpeg to the environment variables after installing
- Ollama (https://ollama.com/) 
    Make sure to pull the models you want to use. Check your GPU's max memory to find the models that can be loaded onto your GPU
- Python 3.12 or higher (https://www.python.org/downloads/)


# Python packages

- langchain
- transformers==4.36.2
- pytorch
- gtts
- pygame
- faiss-cpu (or faiss-gpu if you have a Linux distribution that supports it)
- langchain_ollama
- openai-whisper
- torchaudio
- sounddevice
- sentence_transformers
- librosa
- noisereduce
- webrtcvad
- edge_tts

```bash
pip install transformers==4.36.2 langchain  torch gtts pygame faiss-cpu langchain_ollama openai-whisper torchaudio sounddevice sentence_transformers librosa noisereduce webrtcvad edge-tts
```


# How to run the code

1. Clone the repository
2. Install the required packages
3. Move into the project directory
4. Move to src
5. Run agent_run.py

