"""Pick the device Demucs runs on without importing PyTorch into the window's process.

Importing torch (CUDA build) keeps about 500 MB of libraries in memory for as long as the window is open,
even though separation itself runs in a separate process. So a short-lived child process asks instead.
"""
import subprocess
import sys

CUDA_CHECK_FLAG = "--cuda-check"
_CHECK_SCRIPT = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"


def cuda_check_main() -> int:
    """Entry point of the child process: exit code 0 means a usable CUDA GPU."""
    import torch

    return 0 if torch.cuda.is_available() else 1


def detect_device(timeout: float = 120) -> str:
    # A packaged app has no python.exe to run `-c`, so it starts itself in check mode (see main.py).
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, CUDA_CHECK_FLAG]
    else:
        cmd = [sys.executable, "-c", _CHECK_SCRIPT]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return "cpu"
    return "cuda" if result.returncode == 0 else "cpu"
