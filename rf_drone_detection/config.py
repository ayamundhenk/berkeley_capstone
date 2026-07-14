from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration shared by notebook, scripts, and tests."""

    random_state: int = 42
    sample_rate_hz: float = 1.0
    spectrogram_nperseg: int = 256
    spectrogram_noverlap: int = 128
    max_validation_fpr: float = 0.05
    test_size: float = 0.2
    validation_size: float = 0.2
