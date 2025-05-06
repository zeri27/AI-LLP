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
- transformers
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
pip install transformers==4.50.0 langchain==0.3.21  torch==2.6.0 gTTS==2.5.4 pygame faiss-cpu==1.10.0 langchain-ollama==0.3.0 openai-whisper==20240930 torchaudio==2.6.0 sounddevice==0.5.1 sentence-transformers==3.4.1 librosa==0.11.0 noisereduce==3.0.3 webrtcvad==2.0.10 edge-tts==7.0.0
```


# How to run the code

1. Clone the repository
2. Install the required packages
3. Move into the project directory
4. Move to src
5. Run agent_run.py

