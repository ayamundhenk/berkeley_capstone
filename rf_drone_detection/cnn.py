from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CnnTrainingResult:
    model: object
    validation_scores: np.ndarray
    history: list[dict[str, float]]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def pad_spectrograms(spectrograms: list[np.ndarray] | np.ndarray) -> np.ndarray:
    arrays = [np.asarray(item, dtype=np.float32) for item in spectrograms]
    max_freq = max(array.shape[0] for array in arrays)
    max_time = max(array.shape[1] for array in arrays)
    output = np.zeros((len(arrays), 1, max_freq, max_time), dtype=np.float32)

    for idx, array in enumerate(arrays):
        output[idx, 0, : array.shape[0], : array.shape[1]] = array
    return output


def train_compact_cnn(
    train_spectrograms: list[np.ndarray] | np.ndarray,
    y_train: np.ndarray,
    validation_spectrograms: list[np.ndarray] | np.ndarray,
    y_validation: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> CnnTrainingResult:
    """Train a compact spectrogram CNN when PyTorch is installed."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise ImportError("PyTorch is required for CNN training. Install requirements.txt first.") from exc

    torch.manual_seed(random_state)
    x_train = torch.tensor(pad_spectrograms(train_spectrograms), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)
    x_validation = torch.tensor(pad_spectrograms(validation_spectrograms), dtype=torch.float32)

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_train, y_train_tensor), batch_size=batch_size, shuffle=True)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "train_loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        validation_scores = torch.sigmoid(model(x_validation)).cpu().numpy().reshape(-1)

    return CnnTrainingResult(model=model, validation_scores=validation_scores, history=history)
