"""A small LSTM that predicts the next day's log return.

Modelling returns, not price levels, makes the target roughly stationary, so the scaler
fitted on the past still fits the future (price levels trend out of the training range).
The tuple (scaler, network, window) is bundled in ``FittedModel`` so callers only ever
handle raw returns and cannot mix scaled and unscaled values.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402

from .features import Scaler, make_windows  # noqa: E402

# Every fold/seed builds a new model, and TF warns about 'retracing' for each one. Harmless here.
logging.getLogger("tensorflow").setLevel(logging.ERROR)


@dataclass(frozen=True)
class ModelConfig:
    window: int = 60
    units: tuple[int, ...] = (32, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    patience: int = 6


def build_model(cfg: ModelConfig) -> keras.Model:
    layers = [keras.Input(shape=(cfg.window, 1))]
    for i, units in enumerate(cfg.units):
        layers.append(keras.layers.LSTM(units, return_sequences=i < len(cfg.units) - 1))
        if cfg.dropout:
            layers.append(keras.layers.Dropout(cfg.dropout))
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(cfg.learning_rate))
    return model


@dataclass
class FittedModel:
    network: keras.Model
    scaler: Scaler
    window: int
    epochs_run: int
    best_val_loss: float

    def predict_returns(self, returns: np.ndarray, start: int, stop: int) -> np.ndarray:
        """Predicted raw log returns for target positions [start, stop) of ``returns``."""
        z = self.scaler.transform(returns)
        X, _, _ = make_windows(z, self.window, start, stop)
        return self.scaler.inverse(self.network.predict(X, batch_size=512, verbose=0)[:, 0])

    def predict_next(self, last_window: np.ndarray) -> float:
        """Predicted raw log return for the day after ``last_window`` (raw log returns).

        The window must be exactly ``self.window`` long; anything else is an error rather
        than something to pad or truncate.
        """
        w = np.asarray(last_window, dtype=float)
        if w.shape != (self.window,):
            raise ValueError(f"expected a window of exactly {self.window} returns, got shape {w.shape}")
        z = self.scaler.transform(w).reshape(1, self.window, 1)
        return float(self.scaler.inverse(self.network.predict(z, verbose=0)[0, 0]))


def fit_model(returns: np.ndarray, train_end: int, val_end: int, cfg: ModelConfig, seed: int) -> FittedModel:
    """Train on returns[:train_end]; early-stop on returns[train_end:val_end].

    Nothing at or after ``val_end`` is used: not for the scaler, not for the weights,
    not for choosing when to stop.
    """
    returns = np.asarray(returns, dtype=float)
    if not cfg.window < train_end < val_end <= len(returns):
        raise ValueError("need window < train_end < val_end <= len(returns)")
    scaler = Scaler.fit(returns[:train_end])        # train statistics only
    z = scaler.transform(returns[:val_end])
    X_tr, y_tr, _ = make_windows(z, cfg.window, cfg.window, train_end)
    X_va, y_va, _ = make_windows(z, cfg.window, train_end, val_end)

    keras.utils.set_random_seed(seed)
    model = build_model(cfg)
    hist = model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=cfg.epochs, batch_size=cfg.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                                 restore_best_weights=True)],
        verbose=0,
    )
    val_curve = hist.history["val_loss"]
    return FittedModel(model, scaler, cfg.window, epochs_run=len(val_curve), best_val_loss=float(min(val_curve)))
