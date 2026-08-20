"""Download optional speaker-diarization model files from sherpa-onnx releases."""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "sherpa-onnx"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_ARCHIVE = (
    f"{BASE}/speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    f"{BASE}/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
SEGMENTATION_MODEL = MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx"
EMBEDDING_MODEL = MODEL_DIR / "campplus.onnx"


def _download(url: str, destination: Path) -> None:
    print(f"Downloading {destination.name} ...", flush=True)
    urllib.request.urlretrieve(url, destination)
    print(f"  {destination.stat().st_size / 1024 / 1024:.1f} MB")


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Refusing unsafe archive path: {member.name}")
    archive.extractall(destination)


def download_speaker_models() -> None:
    """Download the segmentation and speaker-embedding models if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not SEGMENTATION_MODEL.exists():
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "segmentation.tar.bz2"
            _download(SEGMENTATION_ARCHIVE, archive_path)
            with tarfile.open(archive_path, "r:bz2") as archive:
                _safe_extract(archive, MODEL_DIR)
    else:
        print(f"Segmentation model already exists: {SEGMENTATION_MODEL}")

    if not EMBEDDING_MODEL.exists():
        with tempfile.TemporaryDirectory() as td:
            temp_model = Path(td) / "campplus.onnx"
            _download(EMBEDDING_URL, temp_model)
            shutil.move(str(temp_model), EMBEDDING_MODEL)
    else:
        print(f"Embedding model already exists: {EMBEDDING_MODEL}")

    missing = [str(path) for path in (SEGMENTATION_MODEL, EMBEDDING_MODEL) if not path.exists()]
    if missing:
        raise RuntimeError("Model download finished but files are missing: " + ", ".join(missing))

    print(f"Speaker models are ready in {MODEL_DIR}")


def main() -> int:
    print(
        "This command downloads third-party pretrained model files. "
        "Review the upstream model terms before redistribution or commercial packaging."
    )
    try:
        download_speaker_models()
        return 0
    except Exception as exc:
        print(f"Failed to download speaker models: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
