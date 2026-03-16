"""
Legacy TFF learning-process API implemented with Flower-compatible logic.

This file preserves old symbol names but does not depend on
tensorflow_federated or adapter-based shims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def _require_tff() -> None:
    """Backward-compatible no-op; TFF is no longer required."""
    return None


class TFFModelFactory:
    """
    Backward-compatible model factory.

    Returns callables that produce fresh Keras model clones.
    """

    def __init__(
        self,
        keras_model: tf.keras.Model,
        input_spec: Tuple[tf.TensorSpec, tf.TensorSpec],
        loss: Optional[tf.keras.losses.Loss] = None,
        metrics: Optional[list] = None,
    ) -> None:
        self._ref_model = keras_model
        self._input_spec = input_spec
        self._loss = loss or tf.keras.losses.BinaryCrossentropy()
        self._metrics = metrics or [tf.keras.metrics.BinaryAccuracy()]

    def create_model_fn(self) -> Callable[[], tf.keras.Model]:
        ref = self._ref_model

        def model_fn() -> tf.keras.Model:
            clone = tf.keras.models.clone_model(ref)
            clone.build(ref.input_shape)
            clone.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=self._loss,
                metrics=self._metrics,
            )
            return clone

        return model_fn


def tff_weights_to_keras(model_weights: Any, keras_model: tf.keras.Model) -> None:
    """
    Copy provided weights into a Keras model.

    Accepts either a dict-like payload with trainable/non_trainable keys
    or a plain list returned by keras_weights_to_tff.
    """
    if isinstance(model_weights, dict):
        trainable = model_weights.get("trainable", [])
        non_trainable = model_weights.get("non_trainable", [])
        all_weights = list(trainable) + list(non_trainable)
        if all_weights:
            keras_model.set_weights(all_weights)
        return

    if isinstance(model_weights, (list, tuple)):
        keras_model.set_weights(list(model_weights))
        return

    raise TypeError("Unsupported model_weights payload for conversion.")


def keras_weights_to_tff(keras_model: tf.keras.Model) -> Dict[str, List[np.ndarray]]:
    """Return a lightweight TFF-like weight structure."""
    return {
        "trainable": [v.numpy() for v in keras_model.trainable_variables],
        "non_trainable": [v.numpy() for v in keras_model.non_trainable_variables],
    }


@dataclass
class TFFProcessConfig:
    client_lr: float = 1e-4
    server_lr: float = 0.1
    client_optimizer: str = "adam"
    server_optimizer: str = "sgd"


class _FlowerLearningProcessShim:
    """
    Minimal process shim preserving old LearningProcess method names.

    This performs simple weighted FedAvg over provided local datasets.
    """

    def __init__(self, model_fn: Callable[[], tf.keras.Model], config: TFFProcessConfig):
        self._model_fn = model_fn
        self._config = config

    def initialize(self) -> Dict[str, Any]:
        model = self._model_fn()
        return {"weights": model.get_weights()}

    def get_model_weights(self, state: Dict[str, Any]) -> List[np.ndarray]:
        return state["weights"]

    def set_model_weights(
        self,
        state: Dict[str, Any],
        weights: List[np.ndarray],
    ) -> Dict[str, Any]:
        state["weights"] = list(weights)
        return state

    def next(
        self,
        state: Dict[str, Any],
        federated_data: List[tf.data.Dataset],
    ) -> Dict[str, Any]:
        if not federated_data:
            return {"state": state, "metrics": {"client_work": {"train": {}}}}

        base_weights = state["weights"]
        updates: List[List[np.ndarray]] = []
        counts: List[int] = []

        for ds in federated_data:
            local_model = self._model_fn()
            local_model.set_weights(base_weights)
            local_model.compile(
                optimizer=tf.keras.optimizers.Adam(self._config.client_lr),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            local_model.fit(ds, epochs=1, verbose=0)
            updates.append(local_model.get_weights())

            card = tf.data.experimental.cardinality(ds).numpy()
            counts.append(int(card) if card > 0 else 1)

        total = max(sum(counts), 1)
        averaged: List[np.ndarray] = []
        for layer_idx in range(len(base_weights)):
            layer_sum = np.zeros_like(base_weights[layer_idx])
            for idx, upd in enumerate(updates):
                layer_sum += upd[layer_idx] * (counts[idx] / total)
            averaged.append(layer_sum)

        state["weights"] = averaged
        return {
            "state": state,
            "metrics": {
                "client_work": {
                    "train": {
                        "num_clients": float(len(federated_data)),
                        "num_examples": float(total),
                    }
                }
            },
        }


def build_tff_learning_process(
    model_fn: Callable[[], Any],
    config: Optional[TFFProcessConfig] = None,
):
    """Build a Flower-backed shim exposing TFF LearningProcess methods."""
    cfg = config or TFFProcessConfig()
    return _FlowerLearningProcessShim(model_fn=model_fn, config=cfg)


class TFFRoundExecutor:
    """
    Backward-compatible round executor.

    Executes one round using a learning process shim and copies updated
    weights back into the provided Keras model.
    """

    def __init__(self, process: Any, keras_model: tf.keras.Model) -> None:
        self.process = process
        self.keras_model = keras_model
        self.state = process.initialize()

    def set_keras_weights(self) -> None:
        self.state = self.process.set_model_weights(
            self.state,
            self.keras_model.get_weights(),
        )

    def run_round(self, federated_data: List[tf.data.Dataset]) -> Dict[str, Any]:
        out = self.process.next(self.state, federated_data)
        self.state = out["state"]
        self.keras_model.set_weights(self.process.get_model_weights(self.state))
        return out.get("metrics", {})


__all__ = [
    "TFFModelFactory",
    "TFFProcessConfig",
    "TFFRoundExecutor",
    "build_tff_learning_process",
    "keras_weights_to_tff",
    "tff_weights_to_keras",
]
