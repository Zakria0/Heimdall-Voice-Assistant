import os
import numpy as np
import librosa
import tarfile
import urllib.request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_NAME = "dev-clean.tar.gz"

# ==========================================
# MODULE 0: DATASET PREP (Local)
# ==========================================
def setup_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"🚀 Dataset not found. Downloading LibriSpeech dev-clean...")
        if not os.path.exists(ARCHIVE_NAME):
            urllib.request.urlretrieve(URL, ARCHIVE_NAME)
            print("   ✅ Download Complete.")
        
        print("   Extracting (this may take a moment)...")
        with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
            # Extract only what we need to keep it clean
            def is_flac(member):
                return member.isfile() and member.name.endswith(".flac")
            
            extracted_count = 0
            MAX_SPEAKERS = 10 # START SMALL for speed, increase later
            extracted_speakers = set()

            for member in tar.getmembers():
                if is_flac(member):
                    parts = member.name.split("/")
                    if len(parts) >= 3:
                        speaker_id = parts[2]
                        if len(extracted_speakers) >= MAX_SPEAKERS and speaker_id not in extracted_speakers:
                            continue
                        
                        extracted_speakers.add(speaker_id)
                        target_dir = os.path.join(DATASET_DIR, f"Speaker_{speaker_id}")
                        os.makedirs(target_dir, exist_ok=True)
                        
                        target_path = os.path.join(target_dir, os.path.basename(member.name))
                        if not os.path.exists(target_path):
                            with open(target_path, "wb") as out:
                                out.write(tar.extractfile(member).read())
            
        print(f"✅ Extracted {len(extracted_speakers)} speakers to {DATASET_DIR}")

# ==========================================
# MODULE 1: AUGMENTATION
# ==========================================
def augment_audio(y, sr):
    augmented_signals = [y]
    try:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y_noise = y + noise_amp * np.random.normal(size=y.shape)
        augmented_signals.append(y_noise)
        
        # Pitch shift is slow, use sparingly
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        augmented_signals.append(y_pitch)
    except Exception as e:
        pass
    return augmented_signals

# ==========================================
# MODULE 2: FEATURE EXTRACTION
# ==========================================
def extract_features(y, sr):
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
    return features

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    setup_dataset()

    features_list = []
    labels_list = []

    print(f"Scanning dataset at: {DATASET_DIR}")
    
    speakers = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Found {len(speakers)} speakers: {speakers}")

    for speaker_name in speakers:
        speaker_path = os.path.join(DATASET_DIR, speaker_name)
        print(f"-> Processing: {speaker_name}")
        
        for filename in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, filename)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y, _ = librosa.effects.trim(y, top_db=20)
                if len(y) < sr: continue

                for sig in augment_audio(y, sr):
                    feat = extract_features(sig, sr)
                    features_list.append(feat)
                    labels_list.append(speaker_name)
            except Exception as e:
                print(f"Error: {e}")

    X = np.array(features_list)
    y_raw = np.array(labels_list)

    print(f"Training on {len(X)} samples...")

    # Encode & Scale
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    print("🚀 Training Ensemble Classifier...")
    clf1 = SVC(probability=True, kernel='rbf', C=10)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = GradientBoostingClassifier(n_estimators=50)
    clf4 = KNeighborsClassifier(n_neighbors=5)

    voting_clf = VotingClassifier(
        estimators=[('svm', clf1), ('rf', clf2), ('gb', clf3), ('knn', clf4)],
        voting='soft'
    )
    
    voting_clf.fit(X_scaled, y)
    print("🏆 Training Complete.")

    # Save Models locally
    print("💾 Saving models to local disk...")
    joblib.dump(voting_clf, 'speaker_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("✅ Done! You can now run 'main.py' to use the assistant.")
