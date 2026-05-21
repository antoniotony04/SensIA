"""
SENSIA - Logique des écrans Kivy
"""

import json
import os
import threading
import time
from datetime import datetime

import numpy as np
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle

# ── Singleton pour les analyseurs (initialisés une seule fois) ────────────────
_face_analyzer  = None
_voice_analyzer = None
_recommender    = None

# Résultats partagés entre les écrans
_session = {
    "face_emotion":  "—",
    "face_anxiety":  0,
    "voice_emotion": "—",
    "voice_anxiety": 0,
    "final_anxiety": 0,
    "voice_done":    False,
}

HISTORY_FILE = "data/history.json"


def _load_analyzers():
    """Charge les modèles PyTorch (appelé au démarrage, hors thread UI)."""
    global _face_analyzer, _voice_analyzer, _recommender
    from core.face_analyzer  import FaceAnalyzer
    from core.voice_analyzer import VoiceAnalyzer
    from core.recommender    import Recommender
    from core.fusion         import compute_final_anxiety

    _face_analyzer  = FaceAnalyzer()
    _voice_analyzer = VoiceAnalyzer()
    _recommender    = Recommender()


def _save_session(session: dict):
    """Sauvegarde la session dans l'historique JSON."""
    os.makedirs("data", exist_ok=True)
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        "date":          datetime.now().strftime("%d/%m/%Y %H:%M"),
        "face_emotion":  session["face_emotion"],
        "voice_emotion": session["voice_emotion"],
        "final_anxiety": session["final_anxiety"],
    }
    history.insert(0, entry)
    history = history[:20]  # garder les 20 dernières sessions

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


# ── Widgets réutilisables ──────────────────────────────────────────────────────

def make_exercise_card(ex: dict) -> BoxLayout:
    """Crée un widget carte pour un exercice."""
    level_colors = {"low": (0.18, 0.80, 0.44, 1),
                    "moderate": (0.95, 0.61, 0.07, 1),
                    "high": (0.91, 0.30, 0.24, 1)}
    color = level_colors.get(ex.get("level", "low"), (0.5, 0.5, 0.5, 1))

    card = BoxLayout(orientation="vertical", size_hint_y=None, height=110,
                     padding=[16, 10], spacing=6)

    with card.canvas.before:
        Color(rgba=(0.12, 0.18, 0.28, 1))
        card._bg = RoundedRectangle(pos=card.pos, size=card.size, radius=[14])

    def update_bg(inst, val):
        inst._bg.pos  = inst.pos
        inst._bg.size = inst.size
    card.bind(pos=update_bg, size=update_bg)

    title_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=30)
    title_row.add_widget(Label(
        text=f"{ex.get('icon', '🟢')}  {ex['title']}",
        bold=True, font_size=15, color=color, halign="left",
        text_size=(None, None)
    ))
    title_row.add_widget(Label(
        text=f"⏱ {ex['duration_min']} min",
        font_size=12, color=(0.6, 0.6, 0.6, 1), halign="right", size_hint_x=0.3
    ))
    card.add_widget(title_row)

    card.add_widget(Label(
        text=ex["description"],
        font_size=13, color=(0.85, 0.85, 0.85, 1),
        halign="left", valign="top",
        text_size=(None, None), size_hint_y=None, height=60
    ))
    return card


def make_history_row(entry: dict) -> BoxLayout:
    row = BoxLayout(orientation="horizontal", size_hint_y=None, height=36, spacing=10)
    anxiety = entry.get("final_anxiety", 0)
    color   = (0.18, 0.80, 0.44, 1) if anxiety < 30 else \
              (0.95, 0.61, 0.07, 1) if anxiety < 60 else \
              (0.91, 0.30, 0.24, 1)
    row.add_widget(Label(text=entry.get("date", "—"),
                         font_size=12, color=(0.6, 0.6, 0.6, 1), size_hint_x=0.4))
    row.add_widget(Label(text=f"Anxiété : {anxiety}%",
                         font_size=13, bold=True, color=color, size_hint_x=0.35))
    row.add_widget(Label(text=entry.get("face_emotion", "—"),
                         font_size=12, color=(0.7, 0.7, 0.7, 1), size_hint_x=0.25))
    return row


# ─────────────────────────────────────────────────────────────────────────────
# ÉCRANS
# ─────────────────────────────────────────────────────────────────────────────

class AccueilScreen(Screen):
    def on_enter(self):
        self.ids.history_box.clear_widgets()
        for entry in _load_history()[:5]:
            self.ids.history_box.add_widget(make_history_row(entry))

    def go_analyse(self):
        # Réinitialiser la session
        _session.update({
            "face_emotion": "—", "face_anxiety": 0,
            "voice_emotion": "—", "voice_anxiety": 0,
            "final_anxiety": 0, "voice_done": False
        })
        self.manager.current = "analyse"

    def go_library(self):
        self.manager.current = "library"


class AnalyseScreen(Screen):
    _camera_event = None

    def on_enter(self):
        if _face_analyzer:
            _face_analyzer.start()
        # Reset état bouton résultat
        self.ids.result_btn.opacity  = 0
        self.ids.result_btn.disabled = True
        self.ids.voice_label.text    = "Appuyez pour analyser"
        # Lancer la mise à jour du webcam (30fps)
        self._camera_event = Clock.schedule_interval(self._update_webcam, 1 / 30)

    def on_leave(self):
        if self._camera_event:
            self._camera_event.cancel()
        if _face_analyzer:
            _face_analyzer.stop()

    def _update_webcam(self, dt):
        if _face_analyzer is None:
            return
        frame = _face_analyzer.get_frame()
        if frame is None:
            return

        # Afficher dans le widget Image Kivy
        h, w, _ = frame.shape
        texture  = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        texture.flip_vertical()
        self.ids.webcam_img.texture = texture

        # Mettre à jour le label émotion visage
        result = _face_analyzer.get_result()
        _session["face_emotion"] = result["emotion"]
        _session["face_anxiety"] = result["anxiety"]
        self.ids.face_label.text = f"{result['emotion'].capitalize()}  ({result['anxiety']}%)"

    def record_voice(self):
        if _voice_analyzer is None or _voice_analyzer.is_recording:
            return
        self.ids.record_btn.text     = "⏺  Enregistrement..."
        self.ids.record_btn.disabled = True
        self.ids.voice_label.text    = "🎙️ Enregistrement en cours..."

        def on_done(emotion, anxiety):
            def _update(dt):
                _session["voice_emotion"] = emotion
                _session["voice_anxiety"] = anxiety
                _session["voice_done"]    = True
                self.ids.voice_label.text = f"{emotion.capitalize()}  ({anxiety}%)"
                self.ids.record_btn.text     = "🎙️  Analyser la voix (3s)"
                self.ids.record_btn.disabled = False
                self.ids.result_btn.opacity  = 1
                self.ids.result_btn.disabled = False
            Clock.schedule_once(_update, 0)

        _voice_analyzer.record_and_analyze(on_done=on_done)

    def go_result(self):
        # Calcul fusion
        from core.fusion import compute_final_anxiety
        final = compute_final_anxiety(
            _session["face_anxiety"], _session["voice_anxiety"]
        )
        _session["final_anxiety"] = final
        _save_session(_session)
        self.manager.current = "resultat"

    def go_back(self):
        self.manager.current = "accueil"


class ResultatScreen(Screen):
    def on_enter(self):
        from core.fusion import anxiety_level

        score   = _session["final_anxiety"]
        niveau, color_hex = anxiety_level(score)

        # Score + niveau
        self.ids.score_label.text  = f"{score}%"
        self.ids.level_label.text  = f"Niveau d'anxiété : {niveau}"
        self.ids.anxiety_bar.anxiety_score = score

        # Couleur du score
        color_map = {
            "Faible":  (0.18, 0.80, 0.44, 1),
            "Modéré":  (0.95, 0.61, 0.07, 1),
            "Élevé":   (0.91, 0.30, 0.24, 1),
        }
        self.ids.score_label.color = color_map.get(niveau, (1, 1, 1, 1))

        # Détails
        self.ids.face_detail.text  = f"😐 Visage :\n{_session['face_emotion'].capitalize()} ({_session['face_anxiety']}%)"
        self.ids.voice_detail.text = f"🎙️ Voix :\n{_session['voice_emotion'].capitalize()} ({_session['voice_anxiety']}%)"

        # Exercices
        self.ids.exercises_box.clear_widgets()
        if _recommender:
            for ex in _recommender.get_recommendations(score, n=2):
                self.ids.exercises_box.add_widget(make_exercise_card(ex))

    def new_analysis(self):
        self.manager.current = "analyse"

    def go_home(self):
        self.manager.current = "accueil"


class LibraryScreen(Screen):
    def on_enter(self):
        self.ids.library_box.clear_widgets()
        if _recommender:
            for ex in _recommender.get_all_exercises():
                self.ids.library_box.add_widget(make_exercise_card(ex))

    def go_back(self):
        self.manager.current = "accueil"
