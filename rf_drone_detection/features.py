from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig


def load_iq_or_vector(path: str | Path) -> np.ndarray:
    """Load a signal from .npy or CSV-like text into a one-dimensional vector."""

    path = Path(path)
    if path.suffix == ".npy":
        values = np.load(path)
    else:
        values = np.loadtxt(path, delimiter=",")

    values = np.asarray(values)
    if np.iscomplexobj(values):
        values = np.abs(values)
    elif values.ndim == 2 and values.shape[1] == 2:
        values = values[:, 0] + 1j * values[:, 1]
        values = np.abs(values)
    return values.astype(np.float32).reshape(-1)


def log_spectrogram(values: np.ndarray, config: PipelineConfig | None = None) -> np.ndarray:
    config = config or PipelineConfig()
    spectrum = _magnitude_spectrogram(values, config)
    logged = np.log1p(spectrum)
    denom = logged.std() + 1e-8
    return ((logged - logged.mean()) / denom).astype(np.float32)


def _magnitude_spectrogram(values: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Compute a small STFT magnitude spectrogram using only NumPy."""

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    nperseg = min(config.spectrogram_nperseg, len(values))
    noverlap = min(config.spectrogram_noverlap, max(0, nperseg - 1))
    step = max(1, nperseg - noverlap)
    if len(values) < nperseg:
        values = np.pad(values, (0, nperseg - len(values)))

    windows = []
    for start in range(0, len(values) - nperseg + 1, step):
        windowed = values[start : start + nperseg] * np.hanning(nperseg)
        windows.append(np.abs(np.fft.rfft(windowed)))

    if not windows:
        windows.append(np.abs(np.fft.rfft(values[:nperseg] * np.hanning(nperseg))))
    return np.stack(windows, axis=1)


def spectral_summary_features(spectrogram: np.ndarray) -> np.ndarray:
    """Create compact baseline features from a normalized log-spectrogram."""

    freq_mean = spectrogram.mean(axis=1)
    time_mean = spectrogram.mean(axis=0)
    return np.array(
        [
            float(spectrogram.mean()),
            float(spectrogram.std()),
            float(spectrogram.max()),
            float(spectrogram.min()),
            float(np.percentile(spectrogram, 25)),
            float(np.percentile(spectrogram, 50)),
            float(np.percentile(spectrogram, 75)),
            float(freq_mean.std()),
            float(time_mean.std()),
            float(np.argmax(freq_mean) / max(1, len(freq_mean) - 1)),
        ],
        dtype=np.float32,
    )


def build_feature_matrix(paths: list[str | Path], config: PipelineConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    config = config or PipelineConfig()
    spectrograms = [log_spectrogram(load_iq_or_vector(path), config=config) for path in paths]
    baseline_features = np.vstack([spectral_summary_features(spec) for spec in spectrograms])
    return baseline_features, np.asarray(spectrograms, dtype=object)
