"""
Flower Federated Learning Cycle — Main Orchestrator
===================================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

This module runs the full enhanced FL pipeline **natively on Flower**:

 1. Enhanced Client Selection
 2. Update Validation & Contribution Weighing
 3. Server-side Knowledge Distillation
 4. Client Reputation Ledger
 5. Evaluation Metrics & Reporting

It replaces the prior TFF/adapter-based flow and uses Flower's
simulation APIs (``flwr.simulation.start_simulation``) with a custom
strategy that integrates the thesis enhancements.
"""

from __future__ import annotations

import json
import logging
import time
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

try:
    import flwr as fl
    from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
except ImportError:
    fl = None  # type: ignore[assignment]

from enhanced_client_selection import (
    ClientMetrics,
    FederatedClient,
    ReputationLedger,
    SelectionWeights,
    EnhancedClientSelector,
)
from update_validation import (
    ContributionWeights,
    ClippingConfig,
    ClientUpdateRecord,
    UpdateValidator,
)
from knowledge_distillation import (
    DistillationConfig,
    run_distillation_round,
)
from client_reputation_ledger import (
    ReputationConfig,
    ClientReputationLedger,
    update_ledger_from_records,
)
from evaluation_metrics import (
    FederatedModelEvaluator,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def _require_flwr() -> None:
    if fl is None:
        raise RuntimeError(
            "Flower (flwr) is not installed.\n"
            "Install it with:  pip install flwr\n"
        )


# ====================================================================== #
#  1.  CONFIGURATION                                                      #
# ====================================================================== #

@dataclass
class FLWRCycleConfig:
    """
    Central configuration for the Flower-based Federated Learning cycle.
    """
    # -- Core FL settings ---------------------------------------------- #
    model_path: str = "efficientnetb4_final.keras"
    num_devices: int = 100
    local_epochs: int = 5
    global_rounds: int = 50
    clients_per_round: int = 15
    local_batch_size: int = 32
    local_lr: float = 1e-4
    eval_every: int = 5

    # -- Distillation (Part 3) ---------------------------------------- #
    enable_distillation: bool = True
    distillation_config: DistillationConfig = field(
        default_factory=lambda: DistillationConfig(
            temperature=2.0,
            lam=0.5,
            epochs=3,
            batch_size=32,
            learning_rate=1e-4,
        )
    )

    # -- Client selection (Part 1) ------------------------------------- #
    selection_weights: SelectionWeights = field(
        default_factory=lambda: SelectionWeights(
            w_v=0.30,
            w_d=0.20,
            w_l=0.10,
            w_r=0.25,
            w_s=0.15,
        )
    )

    # -- Update validation (Part 2) ----------------------------------- #
    contribution_weights: ContributionWeights = field(
        default_factory=lambda: ContributionWeights(
            alpha=0.35,
            beta=0.20,
            gamma=0.20,
            delta=0.25,
        )
    )
    clipping_config: ClippingConfig = field(
        default_factory=lambda: ClippingConfig(
            clip_threshold=10.0,
            clip_value=5.0,
        )
    )
    harmful_threshold: float = 0.02

    # -- Reputation ledger (Part 4) ------------------------------------ #
    reputation_config: ReputationConfig = field(
        default_factory=lambda: ReputationConfig(
            theta=0.0,
            gamma=0.10,
            decay_rate=0.99,
            floor=0.05,
            ceiling=1.0,
            initial_reputation=0.50,
            penalty_factor=0.05,
        )
    )

    # -- Evaluation & output (Part 5) --------------------------------- #
    reports_dir: str = "reports"
    enable_round_checkpoints: bool = True
    checkpoints_dir: str = "reports/checkpoints_flwr"
    checkpoint_every: int = 1
    simulation_client_cpus: float = 2.0
    simulation_client_gpus: float = 0.0
    simulation_local_mode: bool = False
    auto_resume_from_checkpoint: bool = True
    tflite_output_path: str = "effnet_global_flwr_final.tflite"
    input_shape: Tuple[int, ...] = (224, 224, 3)


# ====================================================================== #
#  2.  DATA HELPERS                                                       #
# ====================================================================== #

def generate_synthetic_data(
    num_samples: int,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32) * 0.1
    y = rng.randint(0, 2, size=(num_samples,)).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y))


def generate_proxy_data(
    num_samples: int,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32) * 0.1
    return tf.data.Dataset.from_tensor_slices(x)


def partition_data_iid_flwr(
    full_dataset: tf.data.Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[str, tf.data.Dataset]:
    """
    IID partitioning with **Flower-compatible client IDs** ("0", "1", ...).
    """
    all_data = list(full_dataset.shuffle(buffer_size=10_000, seed=seed))
    total = len(all_data)
    shard_size = max(1, total // num_clients)

    partitions: Dict[str, tf.data.Dataset] = {}
    for i in range(num_clients):
        cid = str(i)
        start = i * shard_size
        end = min(start + shard_size, total)
        if start >= total:
            start = start % total
            end = start + shard_size

        shard_x = [elem[0].numpy() for elem in all_data[start:end]]
        shard_y = [elem[1].numpy() for elem in all_data[start:end]]

        if not shard_x:
            shard_x = [all_data[0][0].numpy()]
            shard_y = [all_data[0][1].numpy()]

        partitions[cid] = tf.data.Dataset.from_tensor_slices(
            (np.stack(shard_x), np.array(shard_y))
        )
    return partitions


# ====================================================================== #
#  3.  TF LITE CONVERSION                                                 #
# ====================================================================== #

def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantise: bool = False,
) -> str:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantise:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    Path(output_path).write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / (1024 * 1024)
    logger.info(
        "TF Lite model saved -> %s  (%.2f MB, quantised=%s)",
        output_path, size_mb, quantise,
    )
    return output_path


# ====================================================================== #
#  4.  FLOWER CLIENT                                                      #
# ====================================================================== #

class FLWRClient(fl.client.NumPyClient):
    """
    Flower NumPyClient wrapper around a Keras model and local dataset.
    """

    def __init__(
        self,
        client_id: str,
        model_fn: callable,
        train_data: Union[tf.data.Dataset, str],
        batch_size: int,
        input_shape: Tuple[int, ...],
        compression_type: str = "GZIP",
    ) -> None:
        _require_flwr()
        self.client_id = client_id
        self.model = model_fn()
        self.train_data = train_data
        self.batch_size = batch_size
        self.input_shape = tuple(input_shape)
        self.compression_type = compression_type
        self._cached_batch_size = None
        self._cached_train = None
        self._base_train_data = None
        self._num_examples = None
        self._compiled_lr = None

    def _parse_tfrecord_example(
        self,
        example_proto: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        feature_desc = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.float32),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_desc)

        image = tf.io.decode_jpeg(parsed["image/encoded"], channels=3)
        image = tf.image.resize(image, tuple(self.input_shape[:2]))
        image = tf.cast(image, tf.float32)

        label = tf.cast(parsed["label"], tf.float32)
        return image, label

    def _get_base_train_data(self) -> tf.data.Dataset:
        if self._base_train_data is not None:
            return self._base_train_data

        if isinstance(self.train_data, tf.data.Dataset):
            self._base_train_data = self.train_data
            return self._base_train_data

        if isinstance(self.train_data, str):
            ds = tf.data.TFRecordDataset(
                self.train_data,
                compression_type=self.compression_type,
                num_parallel_reads=tf.data.AUTOTUNE,
            )
            ds = ds.map(self._parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
            self._base_train_data = ds
            return self._base_train_data

        raise TypeError(
            f"Unsupported client train_data type: {type(self.train_data).__name__}"
        )

    def _get_train_data(self, batch_size: int) -> tf.data.Dataset:
        if self._cached_train is None or self._cached_batch_size != batch_size:
            base_train = self._get_base_train_data()
            self._cached_train = (
                base_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            )
            self._cached_batch_size = batch_size
        return self._cached_train

    def _num_train_examples(self) -> int:
        if self._num_examples is None:
            base_train = self._get_base_train_data()
            card = tf.data.experimental.cardinality(base_train).numpy()
            if card > 0:
                self._num_examples = int(card)
            else:
                self._num_examples = sum(1 for _ in base_train)
        return int(self._num_examples)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return self.model.get_weights()

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "resource_exhausted" in msg
            or "out of memory" in msg
            or "cuda_error_out_of_memory" in msg
        )

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.model.set_weights(parameters)

        lr = float(config.get("learning_rate", 1e-4))
        if self._compiled_lr != lr:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            self._compiled_lr = lr

        batch_size = int(config.get("batch_size", self.batch_size))
        local_epochs = int(config.get("local_epochs", 1))
        effective_batch_size = max(1, batch_size)
        last_exc: Optional[BaseException] = None

        while effective_batch_size >= 1:
            train_data = self._get_train_data(effective_batch_size)
            try:
                t0 = time.perf_counter()
                self.model.fit(train_data, epochs=local_epochs, verbose=0)
                elapsed = time.perf_counter() - t0
                break
            except (tf.errors.ResourceExhaustedError, tf.errors.UnknownError) as exc:
                if not self._is_oom_error(exc) or effective_batch_size == 1:
                    raise
                last_exc = exc
                next_batch = max(1, effective_batch_size // 2)
                logger.warning(
                    "Client %s OOM at batch_size=%d, retrying with batch_size=%d",
                    self.client_id,
                    effective_batch_size,
                    next_batch,
                )
                effective_batch_size = next_batch
                self._cached_train = None
                self._cached_batch_size = None
                gc.collect()
        else:
            raise RuntimeError(
                f"Client {self.client_id} failed to train due to OOM. Last error: {last_exc}"
            )

        # Evaluate with the effective batch size used for fit.
        train_data = self._get_train_data(effective_batch_size)
        result = self.model.evaluate(train_data, verbose=0, return_dict=True)
        local_acc = float(result.get("accuracy", 0.0))

        num_examples = self._num_train_examples()
        metrics = {
            "local_accuracy": local_acc,
            "data_volume": float(num_examples),
            "inference_latency": float(elapsed / max(num_examples, 1)),
            "effective_batch_size": float(effective_batch_size),
        }
        return self.model.get_weights(), num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_weights(parameters)
        batch_size = int(config.get("batch_size", self.batch_size))
        train_data = self._get_train_data(batch_size)
        result = self.model.evaluate(train_data, verbose=0, return_dict=True)
        loss = float(result.get("loss", 0.0))
        accuracy = float(result.get("accuracy", 0.0))
        return loss, self._num_train_examples(), {"accuracy": accuracy}


# ====================================================================== #
#  5.  CUSTOM FLOWER STRATEGY                                             #
# ====================================================================== #

class EnhancedFlowerStrategy(fl.server.strategy.Strategy):
    """
    Flower Strategy that integrates update validation, distillation,
    reputation updates, and enhanced client selection.
    """

    def __init__(
        self,
        config: FLWRCycleConfig,
        global_model: tf.keras.Model,
        clients: Dict[str, FederatedClient],
        selector: EnhancedClientSelector,
        validator: UpdateValidator,
        reputation_ledger: ClientReputationLedger,
        evaluator: FederatedModelEvaluator,
        server_val_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
        round_offset: int = 0,
        total_rounds_target: Optional[int] = None,
    ) -> None:
        _require_flwr()
        self.config = config
        self.global_model = global_model
        self.clients = clients
        self.selector = selector
        self.validator = validator
        self.reputation_ledger = reputation_ledger
        self.evaluator = evaluator
        self.server_val_data = server_val_data
        self.test_data = test_data
        self.proxy_data = proxy_data
        self.supervised_data = supervised_data
        self.round_offset = int(round_offset)
        self.total_rounds_target = int(total_rounds_target or config.global_rounds)
        self._current_weights = global_model.get_weights()

        self.history: Dict[str, list] = {
            "round": [],
            "enhanced_accuracy": [],
            "selected_clients": [],
            "num_accepted": [],
            "num_rejected": [],
            "distillation_loss": [],
        }
        self.last_failures: List[str] = []

    def _save_round_checkpoint(
        self,
        server_round: int,
        enhanced_acc: float,
        num_accepted: int,
        num_rejected: int,
        distill_loss: Optional[float],
    ) -> None:
        if not self.config.enable_round_checkpoints:
            return

        every = max(1, int(self.config.checkpoint_every))
        if server_round % every != 0:
            return

        root_dir = Path(self.config.checkpoints_dir)
        round_dir = root_dir / f"round_{server_round:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        model_path = round_dir / "global_model.keras"
        self.global_model.save(model_path)

        metadata = {
            "round": int(server_round),
            "enhanced_accuracy": float(enhanced_acc),
            "num_accepted": int(num_accepted),
            "num_rejected": int(num_rejected),
            "distillation_loss": (None if distill_loss is None else float(distill_loss)),
            "model_path": str(model_path),
            "created_at_epoch": time.time(),
        }
        (round_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved checkpoint: %s", model_path)

    # ------------------------------------------------------------------ #
    #  Strategy API                                                      #
    # ------------------------------------------------------------------ #

    def initialize_parameters(self, client_manager):
        self._current_weights = self.global_model.get_weights()
        return ndarrays_to_parameters(self._current_weights)

    def configure_fit(self, server_round, parameters, client_manager):
        abs_round = self.round_offset + int(server_round)
        self._current_weights = parameters_to_ndarrays(parameters)

        available = client_manager.all()
        selected = self.selector.select(current_round=abs_round)
        selected_ids = [c.client_id for c in selected if c.client_id in available]

        if len(selected_ids) < self.config.clients_per_round:
            sampled = client_manager.sample(
                num_clients=self.config.clients_per_round,
                min_num_clients=min(self.config.clients_per_round, client_manager.num_available()),
            )
            selected_ids = [c.cid for c in sampled]

        fit_config = {
            "local_epochs": self.config.local_epochs,
            "batch_size": self.config.local_batch_size,
            "learning_rate": self.config.local_lr,
        }
        instructions = []
        for cid in selected_ids:
            instructions.append((available[cid], FitIns(parameters, fit_config)))
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        abs_round = self.round_offset + int(server_round)
        self.last_failures = []
        if failures:
            for failure in failures:
                self.last_failures.append(repr(failure))
            logger.error(
                "Round %d had %d client failures. First failure: %s",
                server_round,
                len(self.last_failures),
                self.last_failures[0],
            )

        if not results:
            logger.warning("No client results to aggregate in round %d.", abs_round)
            return None, {}

        # Update local metrics from clients
        client_updates: Dict[str, List[np.ndarray]] = {}
        data_volumes: Dict[str, int] = {}
        for client, fit_res in results:
            cid = client.cid
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_updates[cid] = weights
            data_volumes[cid] = int(fit_res.num_examples)

            metrics = fit_res.metrics or {}
            if cid in self.clients:
                client_obj = self.clients[cid]
                client_obj.metrics.local_validation_metric = float(
                    metrics.get("local_accuracy", 0.0)
                )
                client_obj.metrics.data_volume = int(metrics.get("data_volume", data_volumes[cid]))
                client_obj.metrics.inference_latency = float(
                    metrics.get("inference_latency", client_obj.metrics.inference_latency)
                )
                client_obj.metrics.last_selected_round = abs_round

        # Validate and aggregate updates
        self.global_model.set_weights(self._current_weights)
        self.validator.global_model.set_weights(self._current_weights)

        records: List[ClientUpdateRecord] = self.validator.validate_updates(
            client_updates=client_updates,
            data_volumes=data_volumes,
            server_val_data=self.server_val_data,
        )
        enhanced_weights = self.validator.aggregate_weighted(
            records, self._current_weights,
        )

        num_accepted = sum(1 for r in records if not r.rejected)
        num_rejected = sum(1 for r in records if r.rejected)

        self.global_model.set_weights(enhanced_weights)
        self.validator.global_model.set_weights(enhanced_weights)

        distill_loss = None
        if self.config.enable_distillation and self.proxy_data is not None:
            contribution_weights = {
                r.client_id: r.contribution_weight
                for r in records
                if not r.rejected and r.contribution_weight > 0
            }
            if contribution_weights:
                kd_history = run_distillation_round(
                    global_model=self.global_model,
                    client_weights={
                        cid: client_updates[cid]
                        for cid in contribution_weights
                    },
                    contribution_weights=contribution_weights,
                    proxy_data=self.proxy_data,
                    supervised_data=self.supervised_data,
                    config=self.config.distillation_config,
                )
                distill_loss = kd_history.get("loss_total", [None])[-1]

        # Reputation updates
        update_ledger_from_records(self.reputation_ledger, records, abs_round)
        self.validator.update_reputations(records)

        # Sync ledger back to selector
        basic_ledger = self.reputation_ledger.as_basic_ledger()
        self.selector.ledger._scores = basic_ledger._scores

        # Quick accuracy on server val set
        enhanced_result = self.global_model.evaluate(
            self.server_val_data.batch(self.config.local_batch_size),
            verbose=0, return_dict=True,
        )
        enhanced_acc = float(enhanced_result.get("accuracy", 0.0))

        # History
        self.history["round"].append(abs_round)
        self.history["enhanced_accuracy"].append(enhanced_acc)
        self.history["selected_clients"].append(list(client_updates.keys()))
        self.history["num_accepted"].append(num_accepted)
        self.history["num_rejected"].append(num_rejected)
        self.history["distillation_loss"].append(distill_loss)

        # Periodic full evaluation and report
        if (
            server_round == 1
            or abs_round == self.total_rounds_target
            or abs_round % self.config.eval_every == 0
        ):
            report = self.evaluator.evaluate(
                test_data=self.test_data,
                batch_size=self.config.local_batch_size,
                federated_round=abs_round,
                extra_info={
                    "enhanced_acc": enhanced_acc,
                    "accepted": num_accepted,
                    "rejected": num_rejected,
                },
            )
            self.evaluator.save_report(report, tag=f"flwr_round_{abs_round:03d}")

        self._save_round_checkpoint(
            server_round=abs_round,
            enhanced_acc=enhanced_acc,
            num_accepted=num_accepted,
            num_rejected=num_rejected,
            distill_loss=distill_loss,
        )

        self._current_weights = enhanced_weights
        params = ndarrays_to_parameters(enhanced_weights)
        return params, {
            "enhanced_accuracy": enhanced_acc,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
        }

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


# ====================================================================== #
#  6.  FLOWER FEDERATED CYCLE                                             #
# ====================================================================== #

class FLWRFederatedLearningCycle:
    """
    End-to-end Flower federated learning cycle integrating Parts 1–5.
    """

    def __init__(self, config: Optional[FLWRCycleConfig] = None) -> None:
        _require_flwr()
        self.config = config or FLWRCycleConfig()
        self.global_model: Optional[tf.keras.Model] = None
        self.clients: Dict[str, FederatedClient] = {}
        self.client_datasets: Dict[str, Union[tf.data.Dataset, str]] = {}

        self.reputation_ledger: Optional[ClientReputationLedger] = None
        self.basic_ledger: Optional[ReputationLedger] = None
        self.selector: Optional[EnhancedClientSelector] = None
        self.validator: Optional[UpdateValidator] = None
        self.evaluator: Optional[FederatedModelEvaluator] = None

        self.history: Dict[str, list] = {}

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def load_global_model(self) -> tf.keras.Model:
        cfg = self.config
        logger.info("Loading global model from %s ...", cfg.model_path)
        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as _effnet_preprocess,
        )
        _custom = {"preprocess_input": _effnet_preprocess}
        model = tf.keras.models.load_model(
            cfg.model_path, custom_objects=_custom, compile=False,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.local_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "Global model loaded -> %s params, input shape %s",
            f"{model.count_params():,}", model.input_shape,
        )
        self.global_model = model
        return model

    def _load_model_from_path(self, model_path: str) -> tf.keras.Model:
        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as _effnet_preprocess,
        )

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"preprocess_input": _effnet_preprocess},
            compile=False,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.local_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _find_latest_checkpoint(self) -> Tuple[int, Optional[Path]]:
        root = Path(self.config.checkpoints_dir)
        if not root.exists():
            return 0, None

        best_round = 0
        best_model_path: Optional[Path] = None
        for round_dir in root.glob("round_*"):
            if not round_dir.is_dir():
                continue
            try:
                round_num = int(round_dir.name.split("_")[-1])
            except ValueError:
                continue

            model_path = round_dir / "global_model.keras"
            if not model_path.exists():
                continue

            if round_num > best_round:
                best_round = round_num
                best_model_path = model_path

        return best_round, best_model_path

    def create_clients(
        self,
        client_data: Dict[str, Union[tf.data.Dataset, str]],
    ) -> Dict[str, FederatedClient]:
        rng = np.random.RandomState(42)
        clients: Dict[str, FederatedClient] = {}

        for cid, local_data in client_data.items():
            if isinstance(local_data, tf.data.Dataset):
                card = tf.data.experimental.cardinality(local_data).numpy()
                n_samples = int(card) if card > 0 else sum(1 for _ in local_data)
                local_dataset_for_client = local_data
            elif isinstance(local_data, str):
                raw_ds = tf.data.TFRecordDataset(
                    local_data,
                    compression_type="GZIP",
                    num_parallel_reads=tf.data.AUTOTUNE,
                )
                n_samples = sum(1 for _ in raw_ds)
                local_dataset_for_client = None
            else:
                raise TypeError(
                    f"Unsupported client data type for cid={cid}: {type(local_data).__name__}"
                )

            metrics = ClientMetrics(
                local_validation_metric=float(rng.uniform(0.4, 0.9)),
                data_volume=n_samples,
                inference_latency=float(rng.uniform(0.01, 0.15)),
                last_selected_round=0,
            )
            clients[cid] = FederatedClient(
                client_id=cid,
                local_data=local_dataset_for_client,
                metrics=metrics,
            )

        logger.info("Created %d federated clients.", len(clients))
        self.clients = clients
        self.client_datasets = client_data
        return clients

    def setup_enhancement_modules(self) -> None:
        cfg = self.config
        assert self.global_model is not None
        assert self.clients

        self.reputation_ledger = ClientReputationLedger(
            config=cfg.reputation_config,
        )
        for c in self.clients.values():
            self.reputation_ledger.register(c.client_id)
        self.basic_ledger = self.reputation_ledger.as_basic_ledger()

        self.selector = EnhancedClientSelector(
            clients=list(self.clients.values()),
            reputation_ledger=self.basic_ledger,
            weights=cfg.selection_weights,
            target_k=cfg.clients_per_round,
        )

        self.validator = UpdateValidator(
            global_model=self.global_model,
            reputation_ledger=self.basic_ledger,
            weights=cfg.contribution_weights,
            clipping=cfg.clipping_config,
            harmful_threshold=cfg.harmful_threshold,
            batch_size=cfg.local_batch_size,
        )

        self.evaluator = FederatedModelEvaluator(
            model=self.global_model,
            model_name="effnet_global_flwr",
            reports_dir=cfg.reports_dir,
        )

        logger.info("Enhancement modules (Parts 1–5) initialised.")

    # ------------------------------------------------------------------ #
    #  Run full cycle                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        server_val_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, list]:
        cfg = self.config
        assert self.global_model is not None
        assert self.selector is not None
        assert self.validator is not None
        assert self.reputation_ledger is not None
        assert self.evaluator is not None

        total_rounds_target = int(cfg.global_rounds)
        round_offset = 0
        if cfg.auto_resume_from_checkpoint:
            ckpt_round, ckpt_model_path = self._find_latest_checkpoint()
            if ckpt_model_path is not None:
                logger.info(
                    "Auto-resume: loading checkpoint from round %d at %s",
                    ckpt_round,
                    ckpt_model_path,
                )
                self.global_model = self._load_model_from_path(str(ckpt_model_path))
                round_offset = int(ckpt_round)
                self.evaluator.model = self.global_model
                self.validator.global_model = self.global_model
            else:
                logger.info("Auto-resume: no checkpoint found, starting fresh run.")

        rounds_to_run = max(0, total_rounds_target - round_offset)
        if rounds_to_run == 0:
            logger.info(
                "Requested %d total rounds and checkpoint already at round %d. Nothing to run.",
                total_rounds_target,
                round_offset,
            )
            self.history = {
                "round": [],
                "enhanced_accuracy": [],
                "selected_clients": [],
                "num_accepted": [],
                "num_rejected": [],
                "distillation_loss": [],
            }
            return self.history

        logger.info(
            "FL Cycle (Flower): %d devices, %d rounds to run (%d target, offset=%d), %d local epochs",
            cfg.num_devices,
            rounds_to_run,
            total_rounds_target,
            round_offset,
            cfg.local_epochs,
        )

        baseline_report = self.evaluator.evaluate(
            test_data=test_data,
            batch_size=cfg.local_batch_size,
            federated_round=0,
            extra_info={"stage": "baseline"},
        )
        self.evaluator.save_report(baseline_report, tag="round_000_baseline_flwr")
        logger.info(
            "Baseline -> Acc: %.4f, F1: %.4f, AUC: %.4f",
            baseline_report.classification.accuracy,
            baseline_report.classification.f1_macro,
            baseline_report.classification.roc_auc,
        )

        strategy = EnhancedFlowerStrategy(
            config=cfg,
            global_model=self.global_model,
            clients=self.clients,
            selector=self.selector,
            validator=self.validator,
            reputation_ledger=self.reputation_ledger,
            evaluator=self.evaluator,
            server_val_data=server_val_data,
            test_data=test_data,
            proxy_data=proxy_data,
            supervised_data=supervised_data,
            round_offset=round_offset,
            total_rounds_target=total_rounds_target,
        )

        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as _effnet_preprocess,
        )
        _custom = {"preprocess_input": _effnet_preprocess}

        def _build_client_model() -> tf.keras.Model:
            # Build from disk to avoid serializing the in-memory global model
            # into Ray actors.
            return tf.keras.models.load_model(
                cfg.model_path,
                custom_objects=_custom,
                compile=False,
            )

        client_data_map = dict(self.client_datasets)
        local_batch_size = int(cfg.local_batch_size)
        input_shape = tuple(cfg.input_shape)

        def _client_fn(cid: str):
            dataset = client_data_map[cid]
            return FLWRClient(
                client_id=cid,
                model_fn=_build_client_model,
                train_data=dataset,
                batch_size=local_batch_size,
                input_shape=input_shape,
                compression_type="GZIP",
            )

        start = time.time()
        fl.simulation.start_simulation(
            client_fn=_client_fn,
            num_clients=len(client_data_map),
            config=fl.server.ServerConfig(num_rounds=rounds_to_run),
            strategy=strategy,
            client_resources={
                "num_cpus": float(cfg.simulation_client_cpus),
                "num_gpus": float(cfg.simulation_client_gpus),
            },
            ray_init_args={
                "include_dashboard": False,
                "ignore_reinit_error": True,
                "local_mode": bool(cfg.simulation_local_mode),
            },
        )
        elapsed = time.time() - start
        logger.info("Cycle complete -> %.1fs", elapsed)

        self.history = strategy.history
        if not self.history.get("round"):
            failure_hint = ""
            if getattr(strategy, "last_failures", None):
                failure_hint = (
                    " Last Flower client failure: "
                    f"{strategy.last_failures[0]}"
                )
            raise RuntimeError(
                "Flower simulation completed with zero successful rounds. "
                "In constrained environments, reduce clients_per_round/num_devices, "
                "lower simulation_client_cpus, or set simulation_local_mode=True."
                + failure_hint
            )

        # Save ledger
        ledger_path = Path(cfg.reports_dir) / "reputation_ledger_flwr_final.json"
        self.reputation_ledger.save(str(ledger_path))

        # TF Lite export
        convert_to_tflite(self.global_model, cfg.tflite_output_path, quantise=False)
        convert_to_tflite(
            self.global_model,
            cfg.tflite_output_path.replace(".tflite", "_quantised.tflite"),
            quantise=True,
        )

        self._print_summary()
        return self.history

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def _print_summary(self) -> None:
        h = self.history
        if not h or not h.get("round"):
            return

        best_idx = int(np.argmax(h["enhanced_accuracy"]))
        best_round = h["round"][best_idx]
        best_acc = h["enhanced_accuracy"][best_idx]
        final_acc = h["enhanced_accuracy"][-1]

        stats = self.reputation_ledger.statistics() if self.reputation_ledger else {}

        lines = [
            "FINAL SUMMARY",
            f"  Rounds: {len(h['round'])}  |  Best acc: {best_acc:.4f} (R{best_round})  |  Final acc: {final_acc:.4f}",
        ]
        lines.append(
            f"  Accepted: {sum(h['num_accepted'])}  Rejected: {sum(h['num_rejected'])}  Rep mu={stats.get('mean_reputation', 0):.4f} sigma={stats.get('std_reputation', 0):.4f}"
        )
        kd = [l for l in h["distillation_loss"] if l is not None]
        if kd:
            lines.append(f"  Distillation loss: {np.mean(kd):.5f}")

        lines.append(f"  TFLite: {self.config.tflite_output_path}")
        print("\n".join(lines))
