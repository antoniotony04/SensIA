"""
SENSIA - Entraînement modèle vocal (optimisé Windows + NVIDIA GPU)
Dataset: RAVDESS - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Structure attendue:
    data/ravdess/Actor_01/*.wav  ...  Actor_24/*.wav

Commande: python training/train_voice.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION    = 3
N_MFCC      = 40
HOP_LENGTH  = 512
T_FRAMES    = 130
NUM_CLASSES = 8

EMOTION_MAP = {
    "01": 0, "02": 1, "03": 2, "04": 3,
    "05": 4, "06": 5, "07": 6, "08": 7,
}
EMOTION_LABELS   = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
ANXIETY_WEIGHTS  = {
    "neutral": 10, "calm": 5, "happy": 10, "sad": 55,
    "angry": 75, "fearful": 90, "disgust": 60, "surprised": 40,
}


class VoiceEmotionCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 128),          nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def extract_features(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc        = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        def norm(x): return (x - x.mean()) / (x.std() + 1e-8)

        features = np.stack([norm(mfcc), norm(mfcc_delta), norm(mfcc_delta2)], axis=0)

        if features.shape[2] > T_FRAMES:
            features = features[:, :, :T_FRAMES]
        else:
            features = np.pad(features, ((0,0),(0,0),(0, T_FRAMES - features.shape[2])))

        return features.astype(np.float32)
    except Exception as e:
        print(f"[WARN] {file_path}: {e}")
        return None


class RAVDESSDataset(Dataset):
    def __init__(self, data_dir):
        self.samples, self.labels = [], []
        for actor_dir in sorted(os.listdir(data_dir)):
            actor_path = os.path.join(data_dir, actor_dir)
            if not os.path.isdir(actor_path):
                continue
            for fname in os.listdir(actor_path):
                if not fname.endswith(".wav"):
                    continue
                parts = fname.replace(".wav", "").split("-")
                if len(parts) < 3:
                    continue
                code = parts[2]
                if code not in EMOTION_MAP:
                    continue
                feats = extract_features(os.path.join(actor_path, fname))
                if feats is not None:
                    self.samples.append(feats)
                    self.labels.append(EMOTION_MAP[code])
        print(f"[INFO] {len(self.samples)} echantillons charges")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):
        return torch.tensor(self.samples[i]), torch.tensor(self.labels[i])


def main():
    DATA_DIR    = "data/ravdess"
    MODEL_OUT   = "models/emotion_voice.pth"
    NUM_EPOCHS  = 40
    BATCH_SIZE  = 64
    LR          = 1e-3
    NUM_WORKERS = 0   # Windows: toujours 0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  SENSIA - Entrainement modele vocal (RAVDESS)")
    print("=" * 55)
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    if not os.path.isdir(DATA_DIR):
        print(f"\n[ERREUR] Dataset introuvable dans '{DATA_DIR}/'")
        print("  Telechargez RAVDESS depuis Kaggle (voir README.md)")
        sys.exit(1)

    os.makedirs("models", exist_ok=True)

    print("[INFO] Chargement des features audio (patience...)") 
    full_ds = RAVDESSDataset(DATA_DIR)
    n_train = int(0.8 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = VoiceEmotionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    print("-" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(feats)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * feats.size(0)
            _, preds  = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(feats)
                _, preds     = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100. * val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train {train_acc:.1f}% | Val {val_acc:.1f}%", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "classes": EMOTION_LABELS,
                "anxiety_weights": ANXIETY_WEIGHTS, "val_acc": val_acc,
            }, MODEL_OUT)
            print("  <- sauvegarde OK")
        else:
            print()

    print(f"\n[DONE] Meilleure val acc : {best_acc:.1f}% -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
