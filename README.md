# GateGuide UI

GateGuide-UI is the auditory interface for the GateGuide project — a wearable assistant designed to help visually impaired travelers navigate airports. This repository focuses on the UI subsystem, which provides an entirely audio-driven model that tests the directional ability through a video test. 
---

# Features

Voice-activated: program runs when hearing the correct trigger word ("Gate"), using VOSK (https://alphacephei.com/vosk/) for offline speech recognition.
Text-detection: programmed with the purpose of syncing with GateGuide's camera system to read overhead signs using OCR and YOLO.
Directional decision: uses distance between text and object bounded boxes to determine the correct directions to give.
Testing video: A simple short video that mimics airport signage and environment that the UI program is being tested on.

# Getting Started

1. Clone the repository through 'git clone https://github.com/danielkim1129/GateGuide-UI.git'
2. See requirements.txt for full documentation on installing dependencies, run 'pip install -r requirements.txt'
3. Run the UI, 'python3 main.py'

# Usage

Once the script is running and a confirmation message appears, request a gate destination. As long as the word 'gate' and a valid number afterwards is included (e.g. 'Gate A20') the video will play.
