"""
SENSIA - Module d'analyse des émotions vocales
Enregistre ~3s audio, extrait MFCC, prédit avec le modèle PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa
import threading
import time

# ── Même architecture que train_voice.py ──────────────────────────────────────
class VoiceEmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


SAMPLE_RATE = 22050
DURATION    = 3
N_MFCC      = 40
HOP_LENGTH  = 512
T_FRAMES    = 130


class VoiceAnalyzer:
    def __init__(self, model_path="models/emotion_voice.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes         = checkpoint["classes"]
        self.anxiety_weights = checkpoint["anxiety_weights"]

        self.model = VoiceEmotionCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.current_emotion = "neutral"
        self.current_probs   = {}
        self.current_anxiety = 10
        self.is_recording    = False
        self._lock = threading.Lock()

    # ── Extraction de features (même que train_voice.py) ──────────────────────
    def _extract_features(self, y):
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc        = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        def norm(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        features = np.stack([norm(mfcc), norm(mfcc_delta), norm(mfcc_delta2)], axis=0)

        if features.shape[2] > T_FRAMES:
            features = features[:, :, :T_FRAMES]
        else:
            pad = T_FRAMES - features.shape[2]
            features = np.pad(features, ((0,0),(0,0),(0,pad)))

        return features.astype(np.float32)

    # ── Enregistrement + analyse (appelé depuis un thread) ───────────────────
    def record_and_analyze(self, on_done=None):
        """Lance un enregistrement de 3s puis met à jour les résultats."""
        def _run():
            with self._lock:
                self.is_recording = True

            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            y = audio.flatten()

            feats  = self._extract_features(y)
            tensor = torch.tensor(feats).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output  = self.model(tensor)
                softmax = torch.softmax(output, dim=1).squeeze().cpu().numpy()

            probs   = {cls: float(p) for cls, p in zip(self.classes, softmax)}
            emotion = max(probs, key=probs.get)
            anxiety = self._compute_anxiety(probs)

            with self._lock:
                self.current_emotion = emotion
                self.current_probs   = probs
                self.current_anxiety = anxiety
                self.is_recording    = False

            if on_done:
                on_done(emotion, anxiety)

        threading.Thread(target=_run, daemon=True).start()

    # ── Score d'anxiété ───────────────────────────────────────────────────────
    def _compute_anxiety(self, probs):
        score = 0.0
        for emotion, prob in probs.items():
            weight = self.anxiety_weights.get(emotion, 20)
            score += prob * weight
        return int(round(score))

    def get_result(self):
        with self._lock:
            return {
                "emotion": self.current_emotion,
                "probs":   self.current_probs,
                "anxiety": self.current_anxiety,
            }
