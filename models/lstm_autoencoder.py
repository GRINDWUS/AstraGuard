"""
AstraGuard 3.0 — LSTM Autoencoder Temporal Anomaly Detector
============================================================
Module C, Layer 2: Temporal Behavioural Intelligence (Eye 3B)

Answers the question:
  "Does the WAY this component's parameters evolved over time look normal?"

Unlike Isolation Forest (which works on static feature snapshots), the LSTM
Autoencoder operates on the full 0h–24h hourly temporal sequence and learns
what NORMAL degradation dynamics look like.

Architecture:
  Input sequence shape: (25 timesteps × 3 channels: iddq, voffset, snr)
  Encoder: LSTM(32) → latent representation
  Decoder: RepeatVector(25) → LSTM(32) → TimeDistributed Dense(3)
  Loss: MSE (reconstruction error on nominal sequences)

Scientific grounding:
  - Trained ONLY on NOMINAL Class 0 sequences → learns the distribution of
    healthy temporal behaviour.
  - Reconstruction error on anomalous sequences is high because the encoder
    was never exposed to those degradation dynamics.
  - This is orthogonal to Isolation Forest: IF works on STATIC 24h feature
    snapshots; LSTM AE captures TEMPORAL EVOLUTION patterns.

Requires: numpy, pandas, scikit-learn
Optional: tensorflow/keras for neural backend (falls back to StatisticalAE)

IMPORTANT — Epistemic Boundary:
  This is a research prototype. The LSTM autoencoder detects temporal
  anomalies relative to the ASQD 2.1 synthetic healthy distribution.
  Reconstruction thresholds are configurable and should be calibrated
  against actual operator-defined acceptance criteria in production.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─── Statistical Fallback AE (No TF required) ─────────────────────────────────

class StatisticalBaselineAE:
    """
    Lightweight PCA-based temporal autoencoder fallback.
    
    Strategy: Flatten 25×3 sequence → 75-dim vector → PCA compression → reconstruct.
    Reconstruction error in 75-dim space = temporal anomaly score.
    
    Used when TensorFlow/Keras is not available. Provides equivalent OOD
    detection capability for research-grade evaluation.
    """
    def __init__(self, n_components: int = 12):
        self.n_components = n_components
        self.scaler = RobustScaler()
        self.is_fitted = False
        self._mean = None
        self._components = None
        self.threshold_ = None

    def _flatten(self, sequences: np.ndarray) -> np.ndarray:
        """Flatten (N, T, C) → (N, T*C)"""
        return sequences.reshape(len(sequences), -1)

    def fit(self, nominal_sequences: np.ndarray):
        """
        Fit on NOMINAL sequences only.
        nominal_sequences: shape (N, 25, 3)
        """
        X = self._flatten(nominal_sequences)
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA via SVD
        self._mean = X_scaled.mean(axis=0)
        X_centered = X_scaled - self._mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self._components = Vt[:self.n_components]  # Top-k principal components
        
        # Compute training reconstruction errors
        train_errors = self._reconstruction_errors(nominal_sequences)
        # Set threshold at 95th percentile of nominal reconstruction errors
        self.threshold_ = float(np.percentile(train_errors, 95))
        self.is_fitted = True
        return self

    def _reconstruction_errors(self, sequences: np.ndarray) -> np.ndarray:
        X = self._flatten(sequences)
        X_scaled = self.scaler.transform(X)
        X_centered = X_scaled - self._mean
        # Project → reconstruct
        codes   = X_centered @ self._components.T           # (N, k)
        X_hat   = codes @ self._components + self._mean     # (N, T*C)
        errors  = np.mean((X_scaled - X_hat) ** 2, axis=1) # MSE per sample
        return errors

    def anomaly_scores(self, sequences: np.ndarray) -> np.ndarray:
        """Return reconstruction error per sequence (higher = more anomalous)."""
        assert self.is_fitted, "Call fit() first."
        return self._reconstruction_errors(sequences)

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Return -1 (anomaly) / +1 (normal), compatible with IsolationForest API."""
        scores = self.anomaly_scores(sequences)
        return np.where(scores > self.threshold_, -1, 1)


# ─── LSTM Autoencoder (TensorFlow backend) ────────────────────────────────────

def _build_lstm_ae(seq_len: int = 25, n_features: int = 3, units: int = 32):
    """Build LSTM encoder-decoder autoencoder with Keras."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        # Encoder
        LSTM(units, activation='tanh', input_shape=(seq_len, n_features),
             return_sequences=False),
        # Bridge
        RepeatVector(seq_len),
        # Decoder
        LSTM(units, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(n_features)),
    ], name="lstm_autoencoder")

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model


class LSTMAnomalyDetector:
    """
    AstraGuard LSTM Autoencoder Temporal Anomaly Detector.

    Dual-backend:
      - TensorFlow/Keras if available (LSTM AE)
      - PCA-based StatisticalBaselineAE otherwise (research fallback)

    Input:  Hourly telemetry sequences, shape (N, 25, 3)
            Channels: [iddq_ua, v_offset_mv, snr_db] at hours 0..24
    Output: Anomaly score (float, higher = more anomalous)
            Binary prediction (-1 = anomaly, +1 = normal)

    See module docstring for scientific grounding.
    """

    CHANNELS = ["iddq_ua", "v_offset_mv", "snr_db"]
    SEQ_LEN   = 25   # Hours 0 to 24 inclusive

    def __init__(self, units: int = 32, epochs: int = 30, batch_size: int = 32,
                 percentile_threshold: float = 95.0):
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.percentile_threshold = percentile_threshold
        self.scaler = RobustScaler()
        self.threshold_: Optional[float] = None
        self.backend = "none"
        self._model = None
        self.is_fitted = False

    # ── Sequence Preparation ───────────────────────────────────────────────────

    @classmethod
    def build_sequences_from_telemetry(
        cls, df_telemetry: pd.DataFrame, max_hour: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract per-component 0h–24h windows from raw hourly telemetry DataFrame.
        
        Returns:
            sequences: (N, SEQ_LEN, 3) float32 array
            component_ids: (N,) string array for tracing
        """
        seqs, ids = [], []
        grouped = df_telemetry[df_telemetry["hour"] <= max_hour].groupby("component_id")
        for comp_id, grp in grouped:
            grp_sorted = grp.sort_values("hour")
            if len(grp_sorted) < cls.SEQ_LEN:
                continue  # Skip incomplete sequences
            seq = grp_sorted[cls.CHANNELS].values[:cls.SEQ_LEN].astype(np.float32)
            seqs.append(seq)
            ids.append(comp_id)
        return np.array(seqs), np.array(ids)

    @classmethod
    def build_sequences_from_checkpoint(
        cls, df_checkpoint: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lightweight fallback: Build pseudo-sequences from checkpoint summaries
        when full hourly telemetry is not available.
        
        Linearly interpolates between 0h → 24h checkpoint values.
        Less informative than true telemetry but preserves the interface.
        """
        seqs, ids = [], []
        for _, row in df_checkpoint.iterrows():
            t = np.linspace(0, 1, cls.SEQ_LEN)
            iddq = row["iddq_0h"] + t * (row["iddq_24h"] - row["iddq_0h"])
            voff = row["v_offset_0h"] + t * (row["v_offset_24h"] - row["v_offset_0h"])
            snr  = row["snr_0h"]  + t * (row["snr_24h"]  - row["snr_0h"])
            seq  = np.stack([iddq, voff, snr], axis=1).astype(np.float32)
            seqs.append(seq)
            ids.append(row["component_id"])
        return np.array(seqs), np.array(ids)

    # ── Scale ─────────────────────────────────────────────────────────────────

    def _scale(self, sequences: np.ndarray, fit: bool = False) -> np.ndarray:
        """Robust-scale each channel independently across the training corpus."""
        N, T, C = sequences.shape
        flat = sequences.reshape(-1, C)
        if fit:
            flat_s = self.scaler.fit_transform(flat)
        else:
            flat_s = self.scaler.transform(flat)
        return flat_s.reshape(N, T, C)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, nominal_sequences: np.ndarray):
        """
        Train on NOMINAL sequences only.
        nominal_sequences: shape (N, SEQ_LEN, 3)
        """
        assert len(nominal_sequences) >= 10, "Need at least 10 nominal sequences."
        X = self._scale(nominal_sequences, fit=True)

        # Try TensorFlow LSTM backend first
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            self._model = _build_lstm_ae(self.SEQ_LEN, 3, self.units)
            self._model.fit(
                X, X,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=0,
            )
            self.backend = "lstm_keras"
            print(f"    [LSTM AE] Trained with TensorFlow backend — {self.epochs} epochs.")
        except ImportError:
            # Fallback to statistical PCA-based AE
            flat = StatisticalBaselineAE(n_components=min(12, len(nominal_sequences) // 2))
            flat.fit(nominal_sequences)
            self._model = flat
            self.backend = "pca_fallback"
            print(f"    [LSTM AE] TensorFlow not found — using PCA Statistical AE fallback.")

        # Calibrate threshold on training nominal reconstruction errors
        train_errors = self._reconstruction_errors_internal(X, nominal_sequences)
        self.threshold_ = float(np.percentile(train_errors, self.percentile_threshold))
        self.is_fitted = True
        print(f"    [LSTM AE] Anomaly threshold: {self.threshold_:.6f} "
              f"(p{self.percentile_threshold:.0f} of nominal errors)")
        return self

    def _reconstruction_errors_internal(
        self, X_scaled: np.ndarray, X_orig: np.ndarray
    ) -> np.ndarray:
        """Compute per-sequence reconstruction MSE."""
        if self.backend == "lstm_keras":
            X_hat = self._model.predict(X_scaled, verbose=0)
            return np.mean((X_scaled - X_hat) ** 2, axis=(1, 2))
        else:  # pca_fallback
            return self._model.anomaly_scores(X_orig)

    # ── Inference ─────────────────────────────────────────────────────────────

    def anomaly_scores(self, sequences: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error per sequence.
        Higher score = more anomalous temporal behaviour.
        """
        assert self.is_fitted, "Call fit() first."
        if self.backend == "lstm_keras":
            X = self._scale(sequences, fit=False)
            X_hat = self._model.predict(X, verbose=0)
            return np.mean((X - X_hat) ** 2, axis=(1, 2))
        else:
            return self._model.anomaly_scores(sequences)

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Return -1 (anomaly) / +1 (normal) per sequence. Consistent with IsolationForest API."""
        scores = self.anomaly_scores(sequences)
        return np.where(scores > self.threshold_, -1, 1)

    def summary(self) -> dict:
        """Return model metadata for provenance logging."""
        return {
            "model_type": "LSTMAnomalyDetector",
            "backend": self.backend,
            "seq_len": self.SEQ_LEN,
            "channels": self.CHANNELS,
            "units": self.units if self.backend == "lstm_keras" else "N/A (PCA)",
            "epochs": self.epochs if self.backend == "lstm_keras" else "N/A",
            "threshold_percentile": self.percentile_threshold,
            "anomaly_threshold": self.threshold_,
        }
