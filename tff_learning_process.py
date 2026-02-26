"""
TFF Learning Process — Model Wrapping & Federated Training
===========================================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Wraps the EfficientNet Keras model into TFF's learning framework and
builds a customised Federated Averaging learning process.

Provides
--------
* ``TFFModelFactory``    — creates the ``model_fn`` callable required by
  ``tff.learning.algorithms.build_weighted_fed_avg()``.
* ``build_tff_learning_process``  — builds the TFF ``LearningProcess``.
* ``TFFRoundExecutor``   — executes TFF rounds and converts weights
  between TFF ``ModelWeights`` and Keras ``model.get_weights()`` format.
* Weight-conversion utilities: ``tff_weights_to_keras``,
  ``keras_weights_to_tff``.

Architecture
------------
TFF handles the federated computation (model broadcast → client-side
local training → data-weighted aggregation a.k.a. FedAvg).  Our custom
enhancements (contribution weighting, distillation, reputation) operate
as a **post-aggregation refinement** in the outer Python loop.

Environment
-----------
Requires ``tensorflow-federated >= 0.48.0``.
See ``requirements_tff.txt`` and ``tff_data_utils.py`` for details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------- Conditional TFF import ------------------------------------- #
# Primary: use the Flower-backed adapter (works on Python 3.12 + TF 2.19).
# Fallback: real TFF if the adapter is unavailable.
try:
    from flwr_adapter import tff_compat as tff  # type: ignore[assignment]
    TFF_AVAILABLE = True
except ImportError:
    try:
        import tensorflow_federated as tff
        TFF_AVAILABLE = True
    except ImportError:
        tff = None  # type: ignore[assignment]
        TFF_AVAILABLE = False

from tff_data_utils import _require_tff

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  1.  MODEL FACTORY                                                      #
# ====================================================================== #

class TFFModelFactory:
    """
    Creates the ``model_fn`` callable that TFF's learning algorithms
    require.

    Each call to ``model_fn()`` must return a **fresh** TFF-wrapped
    Keras model instance with the same architecture.  TFF traces the
    computation graph on first call and reuses it, so deterministic
    architecture is critical.

    Parameters
    ----------
    keras_model : tf.keras.Model
        Reference Keras model whose architecture will be cloned.
        Must **not** be compiled — TFF handles compilation internally.
    input_spec : tuple[tf.TensorSpec, tf.TensorSpec]
        Batched element spec ``(x_spec, y_spec)`` matching the dataset
        structure.  Obtain via ``TFFDataManager.get_element_spec()``.
    loss : tf.keras.losses.Loss | None
        Loss function; defaults to ``BinaryCrossentropy(from_logits=False)``.
    metrics : list[tf.keras.metrics.Metric] | None
        Metrics; defaults to ``[BinaryAccuracy()]``.
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

    # ------------------------------------------------------------------ #
    #  model_fn builder                                                   #
    # ------------------------------------------------------------------ #

    def create_model_fn(self) -> Callable[[], Any]:
        """
        Return a **no-args callable** compatible with
        ``tff.learning.algorithms.build_weighted_fed_avg(model_fn=...)``.

        The closure captures the reference model, spec, loss, and metrics
        so that every invocation produces an architecturally identical
        (but freshly initialised) TFF model.
        """
        _require_tff()

        ref = self._ref_model
        spec = self._input_spec
        loss = self._loss
        metrics_list = self._metrics

        def model_fn():
            # Clone architecture (fresh random weights — TFF will inject
            # the server weights before each client round).
            keras_clone = tf.keras.models.clone_model(ref)
            keras_clone.build(ref.input_shape)
            return tff.learning.models.from_keras_model(
                keras_model=keras_clone,
                input_spec=spec,
                loss=loss,
                metrics=metrics_list,
            )

        return model_fn


# ====================================================================== #
#  2.  WEIGHT CONVERSION  (TFF ↔ Keras)                                  #
# ====================================================================== #

def tff_weights_to_keras(
    model_weights,
    keras_model: tf.keras.Model,
) -> None:
    """
    Copy TFF ``ModelWeights`` into a Keras model's variables.

    Uses TFF's built-in ``assign_weights_to`` when available; falls back
    to manual assignment otherwise.

    Parameters
    ----------
    model_weights : tff.learning.models.ModelWeights
        ``ModelWeights(trainable=[...], non_trainable=[...])``.
    keras_model : tf.keras.Model
        Target Keras model — must share the same architecture.
    """
    _require_tff()

    try:
        # Preferred: TFF's built-in helper
        model_weights.assign_weights_to(keras_model)
    except AttributeError:
        # Fallback: manual assignment
        trainable = list(model_weights.trainable)
        non_trainable = list(model_weights.non_trainable)
        for var, val in zip(keras_model.trainable_variables, trainable):
            var.assign(val)
        for var, val in zip(keras_model.non_trainable_variables, non_trainable):
            var.assign(val)
    logger.debug("TFF → Keras weight transfer complete.")


def keras_weights_to_tff(
    keras_model: tf.keras.Model,
):
    """
    Convert Keras model weights to TFF ``ModelWeights``.

    Returns
    -------
    tff.learning.models.ModelWeights
    """
    _require_tff()

    trainable = [v.numpy() for v in keras_model.trainable_variables]
    non_trainable = [v.numpy() for v in keras_model.non_trainable_variables]
    return tff.learning.models.ModelWeights(
        trainable=trainable,
        non_trainable=non_trainable,
    )


# ====================================================================== #
#  3.  LEARNING-PROCESS BUILDER                                           #
# ====================================================================== #

@dataclass
class TFFProcessConfig:
    """Hyper-parameters for the TFF weighted-FedAvg process."""
    client_lr: float = 1e-4
    server_lr: float = 1.0
    client_optimizer: str = "adam"     # "adam" | "sgd"
    server_optimizer: str = "sgd"     # typically SGD with lr=1.0


def build_tff_learning_process(
    model_fn: Callable[[], Any],
    config: Optional[TFFProcessConfig] = None,
):
    """
    Build a TFF ``LearningProcess`` using weighted Federated Averaging.

    This wraps ``tff.learning.algorithms.build_weighted_fed_avg`` and
    returns a process with the following methods:

    * ``initialize()`` → initial server state
    * ``next(state, federated_data)`` → ``LearningProcessOutput``
    * ``get_model_weights(state)`` → ``ModelWeights``
    * ``set_model_weights(state, weights)`` → updated state

    Parameters
    ----------
    model_fn : callable
        No-args callable that returns a ``tff.learning.models.VariableModel``.
        Obtain from ``TFFModelFactory.create_model_fn()``.
    config : TFFProcessConfig | None
        Optimiser and learning-rate settings.

    Returns
    -------
    tff.learning.templates.LearningProcess
    """
    _require_tff()
    config = config or TFFProcessConfig()

    def client_optimizer_fn():
        if config.client_optimizer == "adam":
            return tf.keras.optimizers.Adam(config.client_lr)
        return tf.keras.optimizers.SGD(config.client_lr)

    def server_optimizer_fn():
        if config.server_optimizer == "adam":
            return tf.keras.optimizers.Adam(config.server_lr)
        return tf.keras.optimizers.SGD(config.server_lr)

    process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
    )

    logger.info(
        "TFF LearningProcess built — client_opt=%s(lr=%g), "
        "server_opt=%s(lr=%g).",
        config.client_optimizer, config.client_lr,
        config.server_optimizer, config.server_lr,
    )
    return process


# ====================================================================== #
#  4.  ROUND EXECUTOR                                                     #
# ====================================================================== #

class TFFRoundExecutor:
    """
    Thin wrapper around a TFF ``LearningProcess`` that handles state
    management and weight conversion.

    Typical workflow
    ----------------
    >>> executor = TFFRoundExecutor(process, keras_model)
    >>> executor.initialize()
    >>> executor.inject_pretrained_weights()     # start from .h5 model
    >>> for t in range(num_rounds):
    ...     metrics = executor.execute_round(federated_data)
    ...     keras_weights = executor.get_keras_weights()
    ...     # ... apply enhancements ...
    ...     executor.set_keras_weights(enhanced_weights)

    Parameters
    ----------
    process : tff.learning.templates.LearningProcess
        TFF learning process (from ``build_tff_learning_process``).
    keras_model : tf.keras.Model
        Reference Keras model — used for weight conversion only.
    """

    def __init__(
        self,
        process,                        # tff.learning.templates.LearningProcess
        keras_model: tf.keras.Model,
    ) -> None:
        _require_tff()
        self.process = process
        self.keras_model = keras_model
        self.state = None
        self._round_count = 0

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """
        Call TFF's ``process.initialize()`` to create the initial server
        state (with random model weights).
        """
        self.state = self.process.initialize()
        self._round_count = 0
        logger.info("TFF process initialised (random server weights).")

    def inject_pretrained_weights(self) -> None:
        """
        Inject the weights from ``self.keras_model`` into the TFF server
        state.  Call this after ``initialize()`` to start federated
        training from a pre-trained checkpoint (e.g. EfficientNet).
        """
        assert self.state is not None, "Call initialize() first."
        tff_weights = keras_weights_to_tff(self.keras_model)
        self.state = self.process.set_model_weights(self.state, tff_weights)
        logger.info("Pre-trained Keras weights injected into TFF state.")

    # ------------------------------------------------------------------ #
    #  Round execution                                                    #
    # ------------------------------------------------------------------ #

    def execute_round(
        self,
        federated_data: List[tf.data.Dataset],
    ) -> Dict[str, Any]:
        """
        Execute one TFF federated round (broadcast → local training →
        FedAvg aggregation).

        Parameters
        ----------
        federated_data : list[tf.data.Dataset]
            Batched datasets for the selected clients.  Obtain from
            ``TFFDataManager.make_federated_data()``.

        Returns
        -------
        metrics : dict
            TFF round metrics (client loss, accuracy, num_examples, …).
        """
        assert self.state is not None, "Call initialize() first."

        result = self.process.next(self.state, federated_data)
        self.state = result.state
        self._round_count += 1

        # Extract metrics — TFF returns nested OrderedDicts
        metrics = _flatten_tff_metrics(result.metrics)
        logger.info(
            "TFF round %d complete — %s",
            self._round_count, _format_metrics(metrics),
        )
        return metrics

    # ------------------------------------------------------------------ #
    #  Weight access                                                      #
    # ------------------------------------------------------------------ #

    def get_keras_weights(self) -> List[np.ndarray]:
        """
        Extract model weights from the TFF state and set them on the
        Keras reference model.

        Returns the weights as a list of numpy arrays (same format
        as ``keras_model.get_weights()``).
        """
        model_weights = self.process.get_model_weights(self.state)
        tff_weights_to_keras(model_weights, self.keras_model)
        return self.keras_model.get_weights()

    def set_keras_weights(self, weights: List[np.ndarray]) -> None:
        """
        Inject (possibly enhanced) Keras weights back into the TFF
        server state so the next round broadcasts the updated model.
        """
        self.keras_model.set_weights(weights)
        tff_weights = keras_weights_to_tff(self.keras_model)
        self.state = self.process.set_model_weights(self.state, tff_weights)
        logger.debug("Enhanced weights injected into TFF state.")

    def get_tff_model_weights(self):
        """Return the raw TFF ``ModelWeights`` from the current state."""
        return self.process.get_model_weights(self.state)


# ====================================================================== #
#  5.  METRIC HELPERS                                                     #
# ====================================================================== #

def _flatten_tff_metrics(metrics_struct) -> Dict[str, float]:
    """
    Flatten TFF's nested ``OrderedDict`` metrics into a simple dict.

    TFF returns metrics like::

        OrderedDict([
            ('distributor', OrderedDict()),
            ('client_work', OrderedDict([
                ('train', OrderedDict([('loss', 0.693), ...])),
            ])),
            ...
        ])

    This flattens all leaf values into ``{'client_work/train/loss': 0.693}``.
    """
    flat: Dict[str, float] = {}

    def _walk(obj, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}{k}/")
        elif hasattr(obj, "_asdict"):
            _walk(obj._asdict(), prefix)
        elif hasattr(obj, "items"):
            for k, v in obj.items():
                _walk(v, f"{prefix}{k}/")
        else:
            key = prefix.rstrip("/")
            try:
                flat[key] = float(obj)
            except (TypeError, ValueError):
                flat[key] = str(obj)

    try:
        _walk(metrics_struct)
    except Exception:
        flat["raw"] = str(metrics_struct)
    return flat


def _format_metrics(metrics: Dict[str, float], max_items: int = 5) -> str:
    """Compact single-line metrics string for logging."""
    items = list(metrics.items())[:max_items]
    return ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in items)


# ====================================================================== #
#  DEMO / SMOKE-TEST                                                      #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  TFF Learning Process — Demo  ===\n")

    np.random.seed(42)
    tf.random.set_seed(42)

    # ---- 1. Build a tiny reference model ----------------------------- #
    INPUT_DIM = 16
    ref_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    # Do NOT compile — TFF handles compilation internally
    print(f"Reference model: {ref_model.count_params()} params, "
          f"input shape {ref_model.input_shape}")

    # ---- 2. Element spec --------------------------------------------- #
    from tff_data_utils import TFFDataManager, generate_synthetic_data, partition_data_iid_tff

    dm = TFFDataManager(input_shape=(INPUT_DIM,))
    input_spec = dm.get_element_spec()
    print(f"Input spec: {input_spec}")

    # ---- 3. Model factory -------------------------------------------- #
    factory = TFFModelFactory(
        keras_model=ref_model,
        input_spec=input_spec,
    )
    print("TFFModelFactory created.")

    # ---- 4. TFF-specific tests --------------------------------------- #
    if TFF_AVAILABLE:
        print("\n--- TFF available — running full demo ---\n")

        model_fn = factory.create_model_fn()
        print(f"model_fn created: {model_fn}")

        # Build learning process
        process = build_tff_learning_process(
            model_fn=model_fn,
            config=TFFProcessConfig(client_lr=1e-3, server_lr=1.0),
        )
        print("Learning process built.")

        # Create round executor
        executor = TFFRoundExecutor(process, ref_model)
        executor.initialize()
        executor.inject_pretrained_weights()
        print("Executor initialised with pre-trained weights.")

        # Synthesise federated data
        full_ds = generate_synthetic_data(240, (INPUT_DIM,), seed=10)
        client_data = partition_data_iid_tff(full_ds, 8)
        selected = ["client_001", "client_003", "client_005"]
        fed_data = dm.make_federated_data(
            client_data, selected, batch_size=16, local_epochs=2,
        )

        # Run 3 TFF rounds
        for rnd in range(1, 4):
            metrics = executor.execute_round(fed_data)
            keras_w = executor.get_keras_weights()
            print(f"  Round {rnd}: {len(keras_w)} weight arrays, "
                  f"metrics keys: {list(metrics.keys())[:4]}")

            # Simulate enhancement: perturb weights slightly
            enhanced = [w + np.random.randn(*w.shape).astype(np.float32) * 0.001
                        for w in keras_w]
            executor.set_keras_weights(enhanced)

        print("\nTFF demo complete.")

    else:
        print(
            "\n⚠  TFF not installed — running structural checks only.\n"
            "   Install: pip install tensorflow-federated==0.48.0\n"
            "   See requirements_tff.txt for the full stack.\n"
        )

        # Structural checks
        print("✓ TFFModelFactory instantiable (model_fn creation requires TFF)")
        print("✓ TFFProcessConfig:", TFFProcessConfig())

        try:
            _ = factory.create_model_fn()
        except RuntimeError as e:
            print(f"✓ model_fn correctly raises: {e.__class__.__name__}")

        # Weight conversion type check (without TFF, just show shapes)
        w = ref_model.get_weights()
        tr = [v.numpy() for v in ref_model.trainable_variables]
        ntr = [v.numpy() for v in ref_model.non_trainable_variables]
        print(f"✓ Keras → TFF mapping: {len(tr)} trainable, "
              f"{len(ntr)} non-trainable arrays")

        print("\nStructural checks passed.")

    print("\nDone.")
