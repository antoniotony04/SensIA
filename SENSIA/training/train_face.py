"""
SENSIA - Entraînement modèle facial (optimisé Windows + NVIDIA GPU)
Dataset: FER2013 - https://www.kaggle.com/datasets/msambare/fer2013

Structure attendue:
    data/fer2013/train/angry/, disgust/, fear/, happy/, neutral/, sad/, surprise/
    data/fer2013/test/  (même structure)

Commande: python training/train_face.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────────
# IMPORTANT Windows: le bloc if __name__ == '__main__' est
# OBLIGATOIRE pour eviter "RuntimeError: freeze_support"
# ─────────────────────────────────────────────────────────────────

class FaceEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Bloc 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Bloc 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Bloc 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 256),          nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    DATA_DIR    = "data/fer2013"
    MODEL_OUT   = "models/emotion_face.pth"
    NUM_EPOCHS  = 30
    BATCH_SIZE  = 128
    LR          = 1e-3
    IMG_SIZE    = 48
    NUM_CLASSES = 7
    NUM_WORKERS = 0   # Windows: toujours 0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  SENSIA - Entrainement modele visage (FER2013)")
    print("=" * 55)
    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True

    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        print(f"\n[ERREUR] Dataset introuvable dans '{DATA_DIR}/train/'")
        print("  Telechargez FER2013 depuis Kaggle (voir README.md)")
        sys.exit(1)

    os.makedirs("models", exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(48, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Classes    : {train_ds.classes}")
    print(f"Train      : {len(train_ds):,} images | Test : {len(test_ds):,} images")

    model     = FaceEmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=4, factor=0.5, verbose=True)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    print("-" * 55)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            _, preds  = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)

        train_loss = running_loss / total
        train_acc  = 100. * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                _, preds     = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val {val_acc:.1f}%", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "classes": train_ds.classes, "val_acc": val_acc,
            }, MODEL_OUT)
            print("  <- sauvegarde OK")
        else:
            print()

    print(f"\n[DONE] Meilleure val acc : {best_acc:.1f}% -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
