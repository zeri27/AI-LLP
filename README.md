# AI-LLP

**AI Language Learning Application**  
*Developed for the DSAIT4065 Conversational Agents course at TU Delft.*

## 👥 Authors

- **Jordy del Castilho** – jordydelcastilho@gmail.com  
- **Liwia Padowska** – liwia.padowska@gmail.com  
- **Yizhen Zang** – yizhenzang@tudelft.nl  
- **Zeryab Alam** – zeryabalam272@icloud.com

## 🧠 Project Overview

AI-LLP is a conversational agent designed to assist with language learning. The system integrates multiple modules to simulate a realistic and helpful dialogue experience for learners.

### 📦 Modules

- **ASR (Automatic Speech Recognition)** – Converts spoken input to text.
- **TTS (Text-to-Speech)** – Converts text responses into speech.
- **NLP (Natural Language Processing)** – Understands and generates natural language responses.
- **MEM (Memory)** – Keeps track of user interactions and context.

The complete report detailing the design, methodology, and evaluation of this project is included in this repository.

## 🛠️ Installation & Setup

### ✅ Prerequisites

- **Python 3.10+**  
  [Download Python](https://www.python.org/downloads/) – tested primarily on Python 3.10.

- **FFmpeg**  
  - [Windows builds](https://www.gyan.dev/ffmpeg/builds/)  
  - [macOS/Linux](https://www.ffmpeg.org/download.html)  
  ⚠️ *After installation, make sure `ffmpeg` is added to your system's environment variables.*

- **Ollama**  
  [Install Ollama](https://ollama.com/) – used to run local language models.  
  📌 *Ensure you pull a model that fits your GPU memory constraints.*

---

### 📥 Setup Instructions

1. **Clone the repository**:
```git clone git@github.com:zeri27/AI-LLP.git```
2. **Install dependencies**:
```pip install -r requirements.txt```

---

### 🚀 Usage

1. Ensure a **stable internet connection** (especially for the TTS module — use version v1 if offline).
2. Run the agent: ```python agent.py```