"""
Définition des modèles PyTorch entraînés de zéro pour SENSIA.

Deux réseaux :
  - EmotionFaceCNN  : CNN pour la reconnaissance des émotions faciales (FER2013, images 48x48 niveaux de gris).
  - EmotionVoiceCNN : CNN 2D appliqué aux spectrogrammes MFCC (RAVDESS).

Les deux produisent 7 classes d'émotions, mappées ensuite vers un score d'anxiété.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ordre des classes (identique pour les deux modèles, pour simplifier la fusion)
EMOTIONS = ["colere", "degout", "peur", "joie", "tristesse", "surprise", "neutre"]

# Poids de contribution de chaque émotion au niveau d'anxiété (0 = apaisant, 1 = anxiogène)
ANXIETY_WEIGHTS = {
    "colere":    0.75,
    "degout":    0.55,
    "peur":      1.00,
    "joie":      0.00,
    "tristesse": 0.85,
    "surprise":  0.45,
    "neutre":    0.20,
}


class EmotionFaceCNN(nn.Module):
    """CNN simple mais efficace pour FER2013 (entrée : 1x48x48)."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        # Bloc 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Bloc 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        # Après 3 poolings : 48 -> 24 -> 12 -> 6 ; 128 canaux * 6 * 6
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 24x24
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # 12x12
        x = self.pool(x)                                  # 6x6
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class EmotionVoiceCNN(nn.Module):
    """
    CNN 2D pour la reconnaissance d'émotions vocales.
    Entrée : spectrogramme MFCC de forme 1 x 40 x T (T = nombre de trames).
    On utilise un pooling adaptatif pour gérer les durées variables.
    """

    def __init__(self, num_classes: int = 7, n_mfcc: int = 40):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        # Réduit toute carte de caractéristiques à 4x4 quelle que soit la durée
        self.adaptive = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive(x)
        x = x