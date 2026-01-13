# Heimdall: Biometric Voice Assistant

Heimdall is an identity-aware voice assistant that secures smart home commands using
biometric speaker recognition. Unlike standard assistants (Alexa/Siri) that obey anyone,
Heimdall uses machine learning to verify _who_ is speaking before executing sensitive commands
like unlocking doors or disabling security systems.

## How It Works

1. Listen: Captures audio command via microphone.
2. Verify: Extracts MFCC features and runs them through an Ensemble Classifier (SVM +
    Random Forest + Gradient Boosting) to identify the speaker.
3. Transcribe: If the speaker is authorized, the audio is converted to text using Google
    Speech Recognition.
4. Act: Semantic intent analysis triggers simulated IoT actions (e.g., controlling GPIO pins on a
    Raspberry Pi).

## Installation

1. System Dependencies
This project requires FFmpeg (for audio processing) and PortAudio (for microphone access).
Please install them for your specific OS before running Python.

 Linux
Arch Linux / Manjaro:

```
sudo pacman -S ffmpeg portaudio libsndfile
```
```
Ubuntu / Debian / Kali:
```
```
sudo apt update
sudo apt install ffmpeg portaudio19-dev libsndfile
```
```
Fedora / RHEL:
```

```
sudo dnf install ffmpeg portaudio-devel libsndfile
```
 macOS
You need Homebrew installed.

```
brew install ffmpeg portaudio
```
 Windows

1. FFmpeg:
    Open PowerShell as Administrator.
    Run: winget install "FFmpeg (Essentials)"
    Restart your terminal to ensure FFmpeg is in your PATH.
2. Visual C++ Build Tools:
    You may need the Visual C++ Build Tools if standard installation fails.
2. Python Setup
Clone the repository and install the dependencies.

```
git clone [https://github.com/Zakria0/Heimdall-Voice-Assistant.git](https://github.com/Zakria0/Heimdall-Voice-Assistant.git)
cd Heimdall-Voice-Assistant
# Create a virtual environment (Recommended)
python -m venv .venv
# Activate it:
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
# Install libraries
pip install -r requirements.txt
```
## Usage

Phase 1: Training the Brain


Heimdall needs to know what you sound like. You can train it on your local machine using the
included dataset tools.

1. Add Your Voice:
    Record yourself reading a book or speaking naturally for ~5 minutes. Save it as
       my_voice.m4a (or wav/mp3) in the project root.
    Run the dataset generator to slice your audio into training samples:

```
# This will create a 'Speaker_Owner' folder in the dataset
ffmpeg -i "my_voice.m4a" -f segment -segment_time 10 -c:a pcm_s16le -ar 22
```
2. Train the Model: Run the training script to download the base LibriSpeech dataset
    (background voices) and train the classifier on your voice.

```
python train.py
```
```
This generates speaker_model.pkl , scaler.pkl , and label_encoder.pkl.
```
Phase 2: Running the Assistant

1. Open main.py and update the HOME_OWNERS list with your speaker ID (e.g.,
    Speaker_Owner or the ID assigned to you).
2. Start the assistant:

```
python main.py
```
3. Commands: Press ENTER and say:
    _"Turn on the lights"_
    _"Open the front door"_
    _"What is the temperature?_

## Project Structure

```
train.py: Downloads data, extracts features, and trains the ML ensemble.
main.py: The live assistant loop (Record -> Verify -> Act).
record_training.py: Helper tool to record short command clips for fine-tuning.
```

```
dataset/: Folder containing raw audio samples (Git ignored).
```
## Privacy Note

This project runs biometric verification locally. However, the Speech-to-Text (ASR)
component currently uses Google's Web API, which requires an internet connection. Future
versions will support offline ASR via Vosk or Whisper.

## Author

Zakaria Oulhadj

