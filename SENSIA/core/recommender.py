"""
SENSIA - Moteur de recommandations
Choisit les exercices adaptés au niveau d'anxiété détecté
"""

import json
import os
import random

EXERCISES_FILE = "data/exercises.json"

# ── Base d'exercices embarquée (aussi disponible en JSON) ─────────────────────
DEFAULT_EXERCISES = {
    "low": [
        {
            "id": "L1",
            "title": "Respiration 4-7-8",
            "description": "Inspirez 4s, bloquez 7s, expirez 8s. Répétez 3 fois.",
            "duration_min": 2,
            "icon": "🌬️"
        },
        {
            "id": "L2",
            "title": "Scan corporel rapide",
            "description": "Fermez les yeux, prenez conscience de chaque partie du corps de la tête aux pieds.",
            "duration_min": 3,
            "icon": "🧘"
        },
        {
            "id": "L3",
            "title": "Marche consciente",
            "description": "Marchez lentement 5 minutes en vous concentrant uniquement sur vos pas.",
            "duration_min": 5,
            "icon": "🚶"
        }
    ],
    "moderate": [
        {
            "id": "M1",
            "title": "Respiration carrée (Box Breathing)",
            "description": "Inspirez 4s → bloquez 4s → expirez 4s → bloquez 4s. Répétez 5 fois.",
            "duration_min": 3,
            "icon": "⬜"
        },
        {
            "id": "M2",
            "title": "Technique 5-4-3-2-1",
            "description": "Nommez 5 choses que vous voyez, 4 que vous touchez, 3 que vous entendez, 2 que vous sentez, 1 que vous goûtez.",
            "duration_min": 3,
            "icon": "👁️"
        },
        {
            "id": "M3",
            "title": "Relaxation musculaire progressive",
            "description": "Contractez puis relâchez chaque groupe musculaire pendant 5s, en commençant par les pieds.",
            "duration_min": 7,
            "icon": "💪"
        },
        {
            "id": "M4",
            "title": "Visualisation positive",
            "description": "Fermez les yeux et imaginez un lieu calme et sécurisant pendant 5 minutes.",
            "duration_min": 5,
            "icon": "🌊"
        }
    ],
    "high": [
        {
            "id": "H1",
            "title": "Cohérence cardiaque (3-5-3)",
            "description": "Inspirez 5s, expirez 5s pendant 5 minutes. Synchronise le rythme cardiaque.",
            "duration_min": 5,
            "icon": "❤️"
        },
        {
            "id": "H2",
            "title": "Respiration diaphragmatique",
            "description": "Une main sur la poitrine, une sur le ventre. Inspirez profondément par le nez : seul le ventre doit se lever. Expirez lentement.",
            "duration_min": 5,
            "icon": "🫁"
        },
        {
            "id": "H3",
            "title": "Méditation guidée courte",
            "description": "Asseyez-vous, fermez les yeux. Focalisez-vous sur votre respiration. Quand l'esprit s'égare, revenez doucement à la respiration.",
            "duration_min": 10,
            "icon": "🕯️"
        },
        {
            "id": "H4",
            "title": "EFT (Tapping)",
            "description": "Tapotez doucement les points d'acupression (sommet de la tête, sourcil, côté de l'œil, sous le nez, menton, clavicule) tout en respirant profondément.",
            "duration_min": 5,
            "icon": "🤲"
        }
    ]
}


class Recommender:
    def __init__(self):
        # Charger ou créer le fichier JSON
        if os.path.exists(EXERCISES_FILE):
            with open(EXERCISES_FILE, "r", encoding="utf-8") as f:
                self.exercises = json.load(f)
        else:
            self.exercises = DEFAULT_EXERCISES
            os.makedirs("data", exist_ok=True)
            with open(EXERCISES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.exercises, f, ensure_ascii=False, indent=2)

    def get_recommendations(self, anxiety_score: int, n: int = 2) -> list[dict]:
        """
        Retourne n exercices adaptés au score d'anxiété.
        Score < 30 → low | 30-60 → moderate | > 60 → high
        """
        if anxiety_score < 30:
            pool = self.exercises["low"]
        elif anxiety_score < 60:
            pool = self.exercises["moderate"]
        else:
            pool = self.exercises["high"]

        return random.sample(pool, min(n, len(pool)))

    def get_all_exercises(self) -> list[dict]:
        """Pour l'écran bibliothèque."""
        all_ex = []
        for level in ["low", "moderate", "high"]:
            for ex in self.exercises[level]:
                ex_copy = ex.copy()
                ex_copy["level"] = level
                all_ex.append(ex_copy)
        return all_ex
