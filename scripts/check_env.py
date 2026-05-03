from __future__ import annotations

import os
import platform
import shutil
import sys


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(not set)')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '(not set)')}")
    print(f"MODELSCOPE_CACHE: {os.environ.get('MODELSCOPE_CACHE', '(not set)')}")
    print(
        "MODELSCOPE_CREDENTIALS_PATH: "
        f"{os.environ.get('MODELSCOPE_CREDENTIALS_PATH', '(not set)')}"
    )

    for command in ["ffmpeg", "uv", "git"]:
        print(f"{command}: {shutil.which(command) or '(not found)'}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch mps available: {torch.backends.mps.is_available()}")
    except Exception as exc:
        print(f"torch: unavailable ({exc})")

    for module in [
        "funasr",
        "modelscope",
        "huggingface_hub",
        "onnxruntime",
        "sherpa_onnx",
        "sounddevice",
        "soundfile",
    ]:
        try:
            imported = __import__(module)
            print(f"{module}: {getattr(imported, '__version__', 'installed')}")
        except Exception as exc:
            print(f"{module}: unavailable ({exc})")


if __name__ == "__main__":
    main()
