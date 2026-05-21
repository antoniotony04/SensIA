"""
SENSIA - Script de vérification et installation pour Windows + NVIDIA GPU
Lance ce script UNE SEULE FOIS avant tout le reste.

Usage: python setup_windows.py
"""

import subprocess
import sys
import os


def run(cmd, check=True):
    print(f"\n>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"[ERREUR] La commande a échoué. Code: {result.returncode}")
        sys.exit(1)
    return result.returncode == 0


def check_nvidia():
    """Vérifie que le GPU NVIDIA est détectable."""
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERREUR] nvidia-smi non trouvé.")
        print("  → Installez les drivers NVIDIA depuis https://www.nvidia.com/drivers")
        sys.exit(1)

    # Extraire la version CUDA supportée
    lines = result.stdout
    print(result.stdout[:500])

    cuda_version = None
    for line in lines.split("\n"):
        if "CUDA Version" in line:
            # Format: "| CUDA Version: 12.x |"
            import re
            m = re.search(r"CUDA Version:\s*([\d.]+)", line)
            if m:
                cuda_version = m.group(1)

    return cuda_version


def check_python_version():
    v = sys.version_info
    print(f"[INFO] Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or v.minor < 9:
        print("[ERREUR] Python 3.9 ou supérieur requis.")
        sys.exit(1)
    print("[OK] Version Python compatible")


def install_pytorch(cuda_version_str):
    """
    Choisit la bonne commande pip PyTorch selon la version CUDA.
    Voir https://pytorch.org/get-started/locally/
    """
    if cuda_version_str is None:
        print("[WARN] Version CUDA non détectée, installation CPU par défaut")
        cmd = "pip install torch torchvision torchaudio"
    else:
        major = int(cuda_version_str.split(".")[0])
        minor = int(cuda_version_str.split(".")[1]) if "." in cuda_version_str else 0

        if major >= 12 and minor >= 1:
            # CUDA 12.1+ → PyTorch cu121
            index_url = "https://download.pytorch.org/whl/cu121"
        elif major >= 11 and minor >= 8:
            # CUDA 11.8 → PyTorch cu118
            index_url = "https://download.pytorch.org/whl/cu118"
        else:
            # Fallback CPU
            print(f"[WARN] CUDA {cuda_version_str} trop ancien, installation CPU")
            index_url = None

        if index_url:
            cmd = f"pip install torch torchvision torchaudio --index-url {index_url}"
        else:
            cmd = "pip install torch torchvision torchaudio"

    print(f"\n[PyTorch] Commande d'installation : {cmd}")
    run(cmd)


def install_other_deps():
    """Installe les autres dépendances une par une pour détecter les erreurs."""
    packages = [
        "kivy[base]",
        "opencv-python",
        "librosa",
        "sounddevice",
        "soundfile",
        "numpy",
    ]
    for pkg in packages:
        print(f"\n[pip] Installation de {pkg}...")
        run(f"pip install {pkg}")


def verify_install():
    """Vérifie que tout est bien installé."""
    print("\n" + "="*50)
    print("VÉRIFICATION DE L'INSTALLATION")
    print("="*50)

    tests = [
        ("torch", "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA dispo: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"),
        ("torchvision", "import torchvision; print(f'  torchvision {torchvision.__version__}')"),
        ("kivy", "import kivy; print(f'  Kivy {kivy.__version__}')"),
        ("cv2", "import cv2; print(f'  OpenCV {cv2.__version__}')"),
        ("librosa", "import librosa; print(f'  librosa {librosa.__version__}')"),
        ("sounddevice", "import sounddevice as sd; print(f'  sounddevice OK'); print(f'  Micros dispo: {sd.query_devices()}')[:100]"),
    ]

    all_ok = True
    for name, code in tests:
        result = subprocess.run(
            f'python -c "{code}"',
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[OK] {name}")
            print(result.stdout.strip())
        else:
            print(f"[FAIL] {name}")
            print(result.stderr.strip()[:200])
            all_ok = False

    return all_ok


def check_cuda_pytorch():
    """Vérifie que PyTorch voit bien le GPU."""
    code = """
import torch
if torch.cuda.is_available():
    print(f"GPU detecte: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Test rapide
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T
    print("Test calcul GPU: OK")
else:
    print("ATTENTION: GPU non detecte par PyTorch !")
    print("Verifiez que vous avez installe la bonne version CUDA.")
"""
    result = subprocess.run(f'python -c "{code}"', shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[WARN]", result.stderr[:300])


if __name__ == "__main__":
    print("=" * 50)
    print("  SENSIA - Setup Windows + NVIDIA GPU")
    print("=" * 50)

    # 1. Vérifier Python
    check_python_version()

    # 2. Vérifier NVIDIA
    print("\n[NVIDIA] Vérification du GPU...")
    cuda_ver = check_nvidia()
    if cuda_ver:
        print(f"[OK] CUDA Version supportée: {cuda_ver}")
    else:
        print("[WARN] Impossible de détecter la version CUDA automatiquement")

    # 3. Installer PyTorch avec CUDA
    print("\n[PyTorch] Installation avec support CUDA...")
    install_pytorch(cuda_ver)

    # 4. Installer les autres dépendances
    print("\n[Deps] Installation des autres dépendances...")
    install_other_deps()

    # 5. Vérification finale
    all_ok = verify_install()
    check_cuda_pytorch()

    print("\n" + "="*50)
    if all_ok:
        print("SETUP TERMINÉ - Tout est prêt !")
        print("Prochaine étape : télécharger les datasets (voir README.md)")
    else:
        print("SETUP INCOMPLET - Vérifiez les erreurs ci-dessus")
    print("="*50)
