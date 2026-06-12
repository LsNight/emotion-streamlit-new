"""
Cross-platform launcher for the emotion analysis app.

Usage:
  python start.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Windows 控制台默认 GBK，中文/emoji 会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
APP_ENTRY = ROOT / "app" / "app.py"
REQ_FILE = ROOT / "requirements.txt"
VENV_CANDIDATES = [ROOT / "venv", ROOT / ".venv"]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _pick_python() -> tuple[Path, Path | None]:
    for venv_dir in VENV_CANDIDATES:
        python_exe = _venv_python(venv_dir)
        if python_exe.exists():
            return python_exe, venv_dir
    return Path(sys.executable), None


def _ensure_venv() -> Path:
    python_exe, venv_dir = _pick_python()
    if venv_dir is not None:
        return python_exe

    venv_dir = ROOT / ".venv"
    print("[1/3] 创建虚拟环境...")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    return _venv_python(venv_dir)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd or ROOT))


def main() -> int:
    print("========================================")
    print("  多模态情感分析系统")
    print("  CV (ResNet18) + NLP (RoBERTa) → Fusion")
    print("========================================")

    python_exe = _ensure_venv()

    print("[2/3] 安装依赖 (首次约 5 分钟，CPU 版 torch 约 200MB)...")
    try:
        _run([str(python_exe), "-m", "pip", "install", "-r", str(REQ_FILE),
              "--extra-index-url", "https://download.pytorch.org/whl/cpu"])
    except subprocess.CalledProcessError:
        _run([str(python_exe), "-m", "ensurepip", "--upgrade"])
        _run([str(python_exe), "-m", "pip", "install", "-r", str(REQ_FILE),
              "--extra-index-url", "https://download.pytorch.org/whl/cpu"])

    print("[3/3] 启动 Web 界面...")
    print("")
    return subprocess.call([str(python_exe), str(APP_ENTRY)], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
