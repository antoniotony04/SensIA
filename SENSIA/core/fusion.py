"""
SENSIA - Module de fusion multimodale
Combine les scores d'anxiété facial + vocal → niveau final
"""


def compute_final_anxiety(face_anxiety: int, voice_anxiety: int,
                           face_weight: float = 0.55,
                           voice_weight: float = 0.45) -> int:
    """
    Fusion pondérée des deux scores.
    Le visage a légèrement plus de poids car la caméra est continue
    et plus fiable que 3s d'audio.
    """
    score = face_weight * face_anxiety + voice_weight * voice_anxiety
    return int(round(min(100, max(0, score))))


def anxiety_level(score: int) -> tuple[str, str]:
    """
    Retourne (niveau_fr, couleur_hex) selon le score.
    """
    if score < 30:
        return "Faible",  "#2ECC71"   # vert
    elif score < 60:
        return "Modéré",  "#F39C12"   # orange
    else:
        return "Élevé",   "#E74C3C"   # rouge
