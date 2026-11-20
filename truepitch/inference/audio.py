"""Audio ingestion and prosody analysis for the app."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..speech.features import SR, load_wav

MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 30.0


def decode_audio(data: bytes, filename: str) -> np.ndarray:
    """Any common format -> 16 kHz mono float32 (no trimming/normalisation, same as training)."""
    if not data:
        raise ValueError("Empty audio file.")
    if len(data) > MAX_BYTES:
        raise ValueError("Audio file larger than 25 MB.")
    suffix = Path(filename or "").suffix.lower() or ".bin"
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / f"in{suffix}"
        src.write_bytes(data)
        wav = None
        try:
            wav = load_wav(src)
        except Exception:
            pass
        if wav is None:
            if shutil.which("ffmpeg") is None:
                raise ValueError("Could not read this audio format and ffmpeg is not installed. Upload a WAV file.")
            dst = Path(d) / "out.wav"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
                                "-ar", str(SR), str(dst)], capture_output=True, text=True)
            if r.returncode:
                raise ValueError("Could not decode audio: " + r.stderr.strip()[:200])
            wav = load_wav(dst)
    if len(wav) / SR > MAX_SECONDS:
        raise ValueError(f"Audio longer than {MAX_SECONDS:.0f} s. Upload a single short utterance "
                         "(training clips are about 5 s).")
    return wav


def _num(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _track_pitch(wav):
    """Praat autocorrelation pitch (robust on noisy audio); falls back to pYIN. Returns (times, f0 with NaN unvoiced)."""
    try:
        import parselmouth
        pitch = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=SR).to_pitch_ac(
            time_step=0.016, pitch_floor=60.0, pitch_ceiling=450.0)
        f0 = pitch.selected_array["frequency"].astype(float)
        f0[f0 == 0] = np.nan
        return np.asarray(pitch.xs()), f0, "praat"
    except ImportError:
        import librosa
        f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=450, sr=SR, frame_length=2048, hop_length=256)
        f0 = np.where(voiced, f0, np.nan)
        return librosa.times_like(f0, sr=SR, hop_length=256), f0, "pyin"


def analyze_audio(wav: np.ndarray) -> dict:
    """Waveform envelope, pitch contour, energy and simple prosodic summaries."""
    import librosa

    hop = 256
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop)[0]
    f_times, f0, tracker = _track_pitch(wav)
    times = librosa.times_like(rms, sr=SR, hop_length=hop)
    active = rms > 0.1 * rms.max() if rms.max() > 0 else np.zeros(len(rms), bool)
    f0v = f0[np.isfinite(f0)]
    voiced_frac = float(len(f0v) / max(len(f0), 1))
    warning = None
    if voiced_frac < 0.10:
        warning = ("Pitch tracking found very little voiced speech (%.0f%% of frames). Background noise, music or "
                   "laughter can break pitch tracking; pitch values below are unreliable." % (100 * voiced_frac))
    elif len(f0v) and not (60 <= np.median(f0v) <= 400):
        warning = "Median pitch is outside the typical speech range; pitch values may contain octave errors."
    bins = 600
    edges = np.linspace(0, len(wav), bins + 1, dtype=int)
    env = [float(np.abs(wav[a:b]).max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    step, fstep = max(1, len(rms) // 400), max(1, len(f0) // 400)
    return {
        "duration_s": round(len(wav) / SR, 3), "pitch_tracker": tracker, "pitch_warning": warning,
        "waveform": [round(v, 4) for v in env],
        "times": [round(float(t), 3) for t in times[::step]],
        "f0_times": [round(float(t), 3) for t in f_times[::fstep]],
        "f0_hz": [_num(v) for v in f0[::fstep]],
        "rms": [round(float(v), 5) for v in rms[::step]],
        "features": {
            "mean_f0_hz": _num(f0v.mean()) if len(f0v) else None,
            "f0_std_hz": _num(f0v.std()) if len(f0v) else None,
            "f0_range_p5_p95_hz": _num(np.percentile(f0v, 95) - np.percentile(f0v, 5)) if len(f0v) > 5 else None,
            "voiced_fraction": _num(voiced_frac),
            "mean_rms": _num(rms.mean()), "max_rms": _num(rms.max()),
            "active_speech_s (heuristic: energy > 10% of peak)": _num(active.sum() * hop / SR),
            "pause_fraction (heuristic)": _num(1 - active.mean()) if len(rms) else None,
        },
    }
