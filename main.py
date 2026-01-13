import sounddevice as sd
import numpy as np
import librosa
import joblib
import os
import sys
import time
import speech_recognition as sr  # NEW: For ASR
from scipy.io.wavfile import write

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
DURATION = 4.0   # Increased to 4s to allow for full sentences
THRESHOLD = 0.60 # Confidence threshold
TEMP_FILENAME = "temp_command.wav"

# REPLACE THIS with the ID from your training output (e.g., "Speaker_1272")
HOME_OWNERS = ["Speaker_26", "Speaker_19", "Speaker_Zakaria"] 

# --- FAKE RASPBERRY PI ---
class RaspberryPiController:
    def __init__(self):
        print("🔌 Connecting to Raspberry Pi GPIO...")
        time.sleep(0.5)
        print("✅ Connection Established (192.168.1.15)")

    def send_signal(self, device, state):
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[RASPBERRY PI LOG] {timestamp} | Sending Signal: GPIO_PIN_18 -> {state.upper()}")
        print(f"                     Action: Turning {device} {state}")
        print("                     Status: SUCCESS\n")

# Initialize the fake hardware
pi = RaspberryPiController()

# --- LOAD BRAINS ---
print("Loading AI Models...")
try:
    model = joblib.load('speaker_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    recognizer = sr.Recognizer() # NEW: ASR Engine
    print("✅ System Ready!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

def extract_features(y, sr):
    # Same feature extraction as training
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        features = np.concatenate([
            np.mean(mfcc, axis=1), np.var(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1), np.var(mfcc_delta, axis=1),
            np.mean(centroid, axis=1), np.var(centroid, axis=1),
            np.mean(contrast, axis=1), np.var(contrast, axis=1),
            np.mean(chroma, axis=1), np.var(chroma, axis=1),
            np.mean(rms, axis=1), np.var(rms, axis=1)
        ])
        return features.reshape(1, -1)
    except Exception:
        return None

def record_audio(duration, fs):
    print("\n🔴 REC (Say something like 'Turn on the lights')...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("⏹️ Stopped.")
    
    # Save to file so ASR can read it too
    # Scale to 16-bit integer for WAV format
    recording_int = (recording * 32767).astype(np.int16)
    write(TEMP_FILENAME, fs, recording_int)
    
    return recording.flatten()

def process_intent(text):
    text = text.lower()
    print(f"🧠 Understanding Intent from: '{text}'")
    
    if "light" in text and "on" in text:
        pi.send_signal("Living Room Lights", "ON")
    elif "light" in text and "off" in text:
        pi.send_signal("Living Room Lights", "OFF")
    elif "door" in text and "open" in text:
        pi.send_signal("Front Door Lock", "OPEN")
    elif "temperature" in text:
        print("\n[RASPBERRY PI] Reading Sensor DHT22...")
        print("               Result: 24°C / 45% Humidity\n")
    else:
        print("❓ Intent not understood. Available commands: lights on/off, open door.")

def speak(text):
    print(f"🤖 Assistant: {text}")
    os.system(f"espeak '{text}' 2>/dev/null")

def main():
    while True:
        input("\nPress ENTER to issue command (Ctrl+C to quit)...")
        
        # 1. Record Audio
        raw_audio = record_audio(DURATION, SAMPLE_RATE)
        
        # 2. Speaker Recognition
        print("🔍 Verifying Identity...")
        y, _ = librosa.load(TEMP_FILENAME, sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < SAMPLE_RATE:
            print("⚠️ Audio too short/quiet.")
            continue

        features = extract_features(y, SAMPLE_RATE)
        if features is None: continue
        
        features_scaled = scaler.transform(features)
        probs = model.predict_proba(features_scaled)[0]
        max_prob = np.max(probs)
        speaker_id = le.inverse_transform([np.argmax(probs)])[0]
        
        print(f"   Confidence: {max_prob*100:.1f}% | ID: {speaker_id}")

        # 3. Security Check
        if max_prob > THRESHOLD and speaker_id in HOME_OWNERS:
            print("✅ ACCESS GRANTED.")
            speak(f"Verified {speaker_id}. Processing command.")
            
            # 4. Speech to Text (ASR)
            try:
                with sr.AudioFile(TEMP_FILENAME) as source:
                    audio_data = recognizer.record(source)
                    # Using Google Web API (Accurate + Free for low usage)
                    text = recognizer.recognize_google(audio_data)
                    print(f"🗣️  Transcription: \"{text}\"")
                    
                    # 5. Intent & Action
                    process_intent(text)
                    
            except sr.UnknownValueError:
                speak("I heard you, but I could not understand the words.")
            except sr.RequestError:
                speak("Network error. Cannot reach speech recognition service.")
        else:
            print("⛔ ACCESS DENIED.")
            speak("Voice not recognized. Security protocols active.")

if __name__ == "__main__":
    main()
