import re
import subprocess
import sys
import urllib.request

PYTORCH_WHEEL_INDEX = "https://download.pytorch.org/whl/torch/"
PYTORCH_BASE_URL    = "https://download.pytorch.org/whl/{tag}"
TORCH_PACKAGES      = ["torch", "torchvision", "torchaudio"]

# Used only when the PyTorch index is unreachable
CUDA_FALLBACK = [
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]
ROCM_FALLBACK = [
    ((6, 0), "rocm6.2"),
    ((5, 7), "rocm5.7"),
]


# ── Detection ────────────────────────────────────────────────────────────────

def _run(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_nvidia():
    result = _run(["nvidia-smi"])
    if result and result.returncode == 0:
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def detect_rocm():
    result = _run(["rocm-smi", "--version"])
    if result and result.returncode == 0:
        match = re.search(r"(\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


# ── PyTorch index ─────────────────────────────────────────────────────────────

def fetch_available_tags():
    """Returns (cuda_tag_nums: set[str], rocm_tag_strs: set[str]) or (None, None)."""
    try:
        print("Fetching available PyTorch wheel versions from pytorch.org ...")
        with urllib.request.urlopen(PYTORCH_WHEEL_INDEX, timeout=10) as resp:
            html = resp.read().decode("utf-8")
        cuda_tags = set(re.findall(r"cu(\d+)", html))   # e.g. {"118", "121", "124", "126"}
        rocm_tags = set(re.findall(r"rocm(\d+\.\d+)", html))  # e.g. {"5.7", "6.2"}
        return cuda_tags, rocm_tags
    except Exception as e:
        print(f"Could not reach PyTorch wheel index ({e}), using built-in fallback list.")
        return None, None


# ── Tag selection ─────────────────────────────────────────────────────────────

def _parse_cuda_num(tag_num: str):
    """'124' → (12, 4)"""
    n = int(tag_num)
    return n // 10, n % 10


def _parse_rocm_str(tag_str: str):
    """'6.2' → (6, 2)"""
    major, minor = tag_str.split(".")
    return int(major), int(minor)


def best_cuda_tag(detected, cuda_tag_nums):
    candidates = [
        (_parse_cuda_num(t), f"cu{t}")
        for t in cuda_tag_nums
        if _parse_cuda_num(t) <= detected
    ]
    return max(candidates)[1] if candidates else None


def best_rocm_tag(detected, rocm_tag_strs):
    candidates = [
        (_parse_rocm_str(t), f"rocm{t}")
        for t in rocm_tag_strs
        if _parse_rocm_str(t) <= detected
    ]
    return max(candidates)[1] if candidates else None


def _fallback_tag(detected, table):
    for req, tag in table:
        if detected >= req:
            return tag
    return None


# ── Install ───────────────────────────────────────────────────────────────────

def install_torch(tag):
    url = PYTORCH_BASE_URL.format(tag=tag)
    print(f"Installing PyTorch ({tag}) from {url} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + TORCH_PACKAGES + ["--index-url", url],
        check=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== MuscleMap setup: detecting GPU ===\n")

    cuda = detect_nvidia()
    rocm = detect_rocm()
    cuda_tags, rocm_tags = fetch_available_tags()

    if cuda:
        major, minor = cuda
        print(f"NVIDIA GPU detected  —  CUDA {major}.{minor}")
        tag = (best_cuda_tag(cuda, cuda_tags) if cuda_tags
               else _fallback_tag(cuda, CUDA_FALLBACK))
        if not tag:
            print(f"No compatible PyTorch wheel found for CUDA {major}.{minor}. Installing CPU version.")
            tag = "cpu"
        install_torch(tag)
        print('\nDone. Verify with: python -c "import torch; print(torch.cuda.is_available())"')
        return

    if rocm:
        major, minor = rocm
        print(f"AMD GPU detected  —  ROCm {major}.{minor}")
        if sys.platform == "win32":
            print("ROCm is not supported on Windows. Installing CPU version.\n"
                  "For GPU acceleration on Windows, an NVIDIA GPU is required.")
            install_torch("cpu")
        else:
            tag = (best_rocm_tag(rocm, rocm_tags) if rocm_tags
                   else _fallback_tag(rocm, ROCM_FALLBACK))
            if not tag:
                print(f"No compatible PyTorch wheel found for ROCm {major}.{minor}. Installing CPU version.")
                tag = "cpu"
            install_torch(tag)
        print("\nDone.")
        return

    print("No GPU detected  —  installing CPU version.")
    print("If you have a GPU, make sure the drivers are installed and retry.")
    install_torch("cpu")
    print("\nDone.")


if __name__ == "__main__":
    main()
