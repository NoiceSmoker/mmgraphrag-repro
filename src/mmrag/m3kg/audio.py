"""Audio decoding / resampling helpers (no model dependencies)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def decode_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an encoded audio file (wav/flac/ogg/mp3) to mono float32."""
    try:
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        import librosa

        wav, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, int(sr)


def load_audio(path: str | Path, target_sr: int | None = None, max_seconds: float | None = None) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if target_sr and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    if max_seconds:
        wav = wav[: int(max_seconds * sr)]
    return wav


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr).astype(np.float32)


def write_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="PCM_16")
