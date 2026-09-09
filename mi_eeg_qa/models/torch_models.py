"""Minimal PyTorch EEG decoders with predict_proba wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EEGNet(nn.Module):
    """Compact EEGNet-style architecture for epochs shaped (channels, times)."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.net(x))


class ShallowConvNet(nn.Module):
    """ShallowConvNet baseline used as a lightweight stand-in for future FBCNet work."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.Conv2d(24, 24, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


@dataclass
class TorchTrainConfig:
    epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    device: str = "cpu"


class _TorchClassifier:
    model_cls: type[nn.Module]

    def __init__(self, epochs: int = 8, batch_size: int = 32, lr: float = 1e-3, seed: int = 7, dropout: float = 0.25, device: str = "cpu"):
        self.config = TorchTrainConfig(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, device=device)
        self.dropout = dropout

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            self.class_to_index_ = {cls: i for i, cls in enumerate(self.classes_)}
            y_train = np.array([self.class_to_index_[cls] for cls in y], dtype=np.int64)
        else:
            self.class_to_index_ = {int(cls): int(cls) for cls in self.classes_}
            y_train = y
        self.model_ = self.model_cls(X.shape[1], X.shape[2], len(self.classes_), dropout=self.dropout).to(self.config.device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_train))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.config.seed))
        self.model_.train()
        for _ in range(self.config.epochs):
            for xb, yb in loader:
                xb = xb.to(self.config.device)
                yb = yb.to(self.config.device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: np.ndarray, mc_dropout: bool = False, mc_samples: int = 20) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before predict_proba")
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.config.device)
        if mc_dropout:
            self.model_.train()
            probs = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    probs.append(torch.softmax(self.model_(X_tensor), dim=1).cpu().numpy())
            return np.mean(probs, axis=0)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(X_tensor), dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]


class EEGNetClassifier(_TorchClassifier):
    model_cls = EEGNet


class ShallowConvNetClassifier(_TorchClassifier):
    model_cls = ShallowConvNet
