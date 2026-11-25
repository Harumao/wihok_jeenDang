import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.appearance import configure_high_dpi, load_svg_icon, setup_fonts
from ui.main_window import MainWindow

ASSETS = Path(__file__).parent / "assets"


def run_demucs_worker(arguments: list[str]) -> int:
    """Packaged builds run separation by starting this program again with --demucs (see core.separator)."""
    from demucs.separate import main as demucs_main

    demucs_main(arguments)
    return 0


def run_selftest(report_path: str) -> int:
    """Check that this build can do its job and write the result as JSON. Used to verify packaged builds,
    which are windowed and have no console to print to."""
    import json
    import shutil
    import tempfile
    import traceback

    report: dict = {"frozen": bool(getattr(sys, "frozen", False)), "python": sys.version.split()[0]}
    try:
        import numpy as np
        import soundfile as sf
        import torch

        from core.analysis import analyze
        from ui.appearance import FONT_FILE

        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
        report["font_file_present"] = FONT_FILE.exists()
        report["ffmpeg_on_path"] = shutil.which("ffmpeg") is not None
        sr = 44100
        clicks = np.zeros(sr * 12, dtype="float32")
        for beat in range(24):
            start = int(beat * sr * 0.5)  # 120 BPM
            clicks[start:start + 800] = np.sin(2 * np.pi * 1000 * np.arange(800) / sr) * np.exp(-np.arange(800) / 150)
        with tempfile.TemporaryDirectory() as tmp:
            song = f"{tmp}/clicks.wav"
            sf.write(song, clicks, sr)
            result = analyze(song)
        report["bpm"] = round(result.bpm, 1)
        report["key"] = result.key
        report["ok"] = True
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return 0 if report["ok"] else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--demucs":
        return run_demucs_worker(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--cuda-check":
        from core.device import cuda_check_main
        return cuda_check_main()
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest":
        return run_selftest(sys.argv[2])
    if sys.platform == "win32":
        # Without this the taskbar groups the window under python.exe and shows its icon.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Wihok.Wihok")
    configure_high_dpi()
    app = QApplication(sys.argv)
    app.setOrganizationName("Wihok")
    app.setApplicationName("Wihok")
    app.setWindowIcon(load_svg_icon(ASSETS / "wihok.svg"))
    setup_fonts(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
