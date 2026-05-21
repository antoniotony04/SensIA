"""
SENSIA - Application principale (Kivy + PyTorch)
Point d'entrée : python main.py

Prérequis (installer une fois) :
    pip install torch torchvision torchaudio
    pip install kivy opencv-python librosa sounddevice numpy

Avant de lancer :
    1. Télécharger FER2013 (Kaggle) → data/fer2013/
    2. Télécharger RAVDESS (Kaggle) → data/ravdess/
    3. python training/train_face.py     (crée models/emotion_face.pth)
    4. python training/train_voice.py    (crée models/emotion_voice.pth)
    5. python main.py
"""

import os
import threading

# Kivy doit être configuré AVANT les imports kivy
os.environ["KIVY_NO_CONSOLELOG"] = "1"

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, FadeTransition
from kivy.clock import Clock
from kivy.core.window import Window

Window.size = (420, 800)   # Taille proche d'un smartphone pour la démo

# Charger le fichier de design
Builder.load_file("ui/sensia.kv")

from ui.screens import (
    AccueilScreen, AnalyseScreen, ResultatScreen, LibraryScreen,
    _load_analyzers
)


class SensiaApp(App):
    def build(self):
        self.title = "SENSIA — Gestion de l'Anxiété"

        sm = ScreenManager(transition=FadeTransition(duration=0.2))
        sm.add_widget(AccueilScreen(name="accueil"))
        sm.add_widget(AnalyseScreen(name="analyse"))
        sm.add_widget(ResultatScreen(name="resultat"))
        sm.add_widget(LibraryScreen(name="library"))

        # Charger les modèles PyTorch en arrière-plan (sans bloquer l'UI)
        threading.Thread(target=self._init_models, daemon=True).start()

        return sm

    def _init_models(self):
        """Initialise les analyseurs dans un thread séparé."""
        try:
            _load_analyzers()
            print("[SENSIA] Modèles chargés ✓")
        except FileNotFoundError as e:
            print(f"[SENSIA] Modèles introuvables : {e}")
            print("[SENSIA] Lance d'abord training/train_face.py et training/train_voice.py")
        except Exception as e:
            print(f"[SENSIA] Erreur chargement modèles : {e}")


if __name__ == "__main__":
    SensiaApp().run()
