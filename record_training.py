import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import time

# --- CONFIG ---
FOLDER = "dataset/Speaker_Zakaria"
SAMPLE_RATE = 22050
DURATION = 5  # Seconds per clip
NUM_FILES = 18 # How many clips to record
START_INDEX = 200 # Start file naming here to avoid overwriting iPhone data

def main():
    print(f"🎤 Recording {NUM_FILES} clips for training on THIS MICROPHONE.")
    print("   Get ready to read commands from the list...")
    
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    for i in range(NUM_FILES):
        filename = f"{FOLDER}/file_{START_INDEX + i}.wav"
        print(f"\n--- Clip {i+1}/{NUM_FILES} ---")
        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1... SPEAK!")
        
        # Record
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        
        # Save
        write(filename, SAMPLE_RATE, recording)
        print(f"   Saved: {filename}")
        time.sleep(1)

    print("\n✅ Done! Now run 'python train.py' to update the brain.")

if __name__ == "__main__":
    main()
