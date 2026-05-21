"""
SENSIA - Module d'analyse des émotions faciales
Utilise la caméra en temps réel + modèle PyTorch entraîné
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import threading

# ── Même architecture que train_face.py ───────────────────────────────────────
class FaceEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Poids d'anxiété par émotion (0-100)
ANXIETY_WEIGHTS = {
    "angry":    75,
    "disgust":  60,
    "fear":     90,
    "happy":    10,
    "neutral":  15,
    "sad":      55,
    "surprise": 40,
}


class FaceAnalyzer:
    def __init__(self, model_path="models/emotion_face.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Charger le modèle
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.model = FaceEmotionCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # Détecteur de visage OpenCV (Haar cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Résultats en temps réel
        self.current_emotion = "neutral"
        self.current_probs   = {}
        self.current_anxiety = 15
        self.current_frame   = None  # frame annotée pour affichage Kivy
        self._lock = threading.Lock()

        # Caméra
        self.cap     = None
        self._running = False

    # ── Démarrer la capture ────────────────────────────────────────────────────
    def start(self):
        self.cap = cv2.VideoCapture(0)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()

    # ── Boucle de capture (thread séparé) ─────────────────────────────────────
    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            emotion = "neutral"
            probs   = {}
            anxiety = 15

            for (x, y, w, h) in faces[:1]:  # on prend uniquement le visage principal
                face_roi = gray[y:y+h, x:x+w]
                tensor   = self.transform(face_roi).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(tensor)
                    softmax = torch.softmax(output, dim=1).squeeze().cpu().numpy()

                probs   = {cls: float(p) for cls, p in zip(self.classes, softmax)}
                emotion = max(probs, key=probs.get)
                anxiety = self._compute_anxiety(probs)

                # Annotation visuelle
                color = (0, 255, 0) if anxiety < 50 else (0, 165, 255) if anxiety < 75 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{emotion} ({anxiety}%)",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Convertir BGR → RGB pour Kivy
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self._lock:
                self.current_emotion = emotion
                self.current_probs   = probs
                self.current_anxiety = anxiety
                self.current_frame   = frame_rgb

    # ── Calcul du score d'anxiété (0-100) ─────────────────────────────────────
    def _compute_anxiety(self, probs):
        score = 0.0
        for emotion, prob in probs.items():
            weight = ANXIETY_WEIGHTS.get(emotion, 20)
            score += prob * weight
        return int(round(score))

    # ── Getters thread-safe ────────────────────────────────────────────────────
    def get_result(self):
        with self._lock:
            return {
                "emotion": self.current_emotion,
                "probs":   self.current_probs,
                "anxiety": self.current_anxiety,
            }

    def get_frame(self):
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None
