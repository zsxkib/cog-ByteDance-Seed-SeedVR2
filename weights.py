import subprocess
import time
from pathlib import Path
from typing import Iterable

MODEL_CACHE = Path("model_cache")
BASE_URL = "https://weights.replicate.delivery/default/seedvr2/model_cache/"
MODEL_FILES: Iterable[str] = [
    ".cache.tar",
    "version.txt",
    "weights.tar",
    "wheels.tar",
    "xet.tar",
]

CKPT_DIR = MODEL_CACHE / "weights"
WHEEL_DIR = MODEL_CACHE / "wheels"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)


def download_weights(url: str, dest: Path) -> None:
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    target = dest if dest.suffix != ".tar" else dest.parent
    command = ["pget", "-vf" + ("x" if dest.suffix == ".tar" else ""), url, str(target)]
    print(f"[~] Running command: {' '.join(command)}")
    subprocess.check_call(command, close_fds=False)
    print("[+] Download completed in:", time.time() - start, "seconds")


def ensure_model_cache() -> None:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    for model_file in MODEL_FILES:
        url = BASE_URL + model_file
        dest_path = MODEL_CACHE / model_file
        if model_file.endswith(".tar"):
            extracted_path = dest_path.parent / model_file.replace(".tar", "")
            if extracted_path.exists():
                continue
        elif dest_path.exists():
            continue
        download_weights(url, dest_path)


def ensure_weight(filename: str) -> Path:
    path = CKPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected weight {path} missing. Did the CDN download succeed?")
    return path
