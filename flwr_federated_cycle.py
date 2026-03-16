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

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        train_data: tf.data.Dataset,
        batch_size: int,
    ) -> None:
        _require_flwr()
        self.client_id = client_id
        self.model = model_fn()
        self.train_data = train_data
        self.batch_size = batch_size
        self._cached_batch_size = None
        self._cached_train = None
        self._num_examples = None
        self._compiled_lr = None

    def _get_train_data(self, batch_size: int) -> tf.data.Dataset:
        if self._cached_train is None or self._cached_batch_size != batch_size:
            self._cached_train = (
                self.train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            )
            self._cached_batch_size = batch_size
        return self._cached_train

    def _num_train_examples(self) -> int:
        if self._num_examples is None:
            card = tf.data.experimental.cardinality(self.train_data).numpy()
            if card > 0:
                self._num_examples = int(card)
            else:
                self._num_examples = sum(1 for _ in self.train_data)
        return int(self._num_examples)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return self.model.get_weights()

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
        train_data = self._get_train_data(batch_size)

        t0 = time.perf_counter()
        self.model.fit(train_data, epochs=local_epochs, verbose=0)
        elapsed = time.perf_counter() - t0

        result = self.model.evaluate(train_data, verbose=0, return_dict=True)
        local_acc = float(result.get("accuracy", 0.0))

        num_examples = self._num_train_examples()
        metrics = {
            "local_accuracy": local_acc,
            "data_volume": float(num_examples),
            "inference_latency": float(elapsed / max(num_examples, 1)),
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
        self._current_weights = global_model.get_weights()

        self.history: Dict[str, list] = {
            "round": [],
            "enhanced_accuracy": [],
            "selected_clients": [],
            "num_accepted": [],
            "num_rejected": [],
            "distillation_loss": [],
        }

    # ------------------------------------------------------------------ #
    #  Strategy API                                                      #
    # ------------------------------------------------------------------ #

    def initialize_parameters(self, client_manager):
        self._current_weights = self.global_model.get_weights()
        return ndarrays_to_parameters(self._current_weights)

    def configure_fit(self, server_round, parameters, client_manager):
        self._current_weights = parameters_to_ndarrays(parameters)

        available = client_manager.all()
        selected = self.selector.select(current_round=server_round)
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
        if not results:
            logger.warning("No client results to aggregate in round %d.", server_round)
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
                client_obj.metrics.last_selected_round = server_round

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
        update_ledger_from_records(self.reputation_ledger, records, server_round)
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
        self.history["round"].append(server_round)
        self.history["enhanced_accuracy"].append(enhanced_acc)
        self.history["selected_clients"].append(list(client_updates.keys()))
        self.history["num_accepted"].append(num_accepted)
        self.history["num_rejected"].append(num_rejected)
        self.history["distillation_loss"].append(distill_loss)

        # Periodic full evaluation and report
        if server_round == 1 or server_round == self.config.global_rounds or server_round % self.config.eval_every == 0:
            report = self.evaluator.evaluate(
                test_data=self.test_data,
                batch_size=self.config.local_batch_size,
                federated_round=server_round,
                extra_info={
                    "enhanced_acc": enhanced_acc,
                    "accepted": num_accepted,
                    "rejected": num_rejected,
                },
            )
            self.evaluator.save_report(report, tag=f"flwr_round_{server_round:03d}")

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
        self.client_datasets: Dict[str, tf.data.Dataset] = {}

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

    def create_clients(
        self,
        client_data: Dict[str, tf.data.Dataset],
    ) -> Dict[str, FederatedClient]:
        rng = np.random.RandomState(42)
        clients: Dict[str, FederatedClient] = {}

        for cid, local_ds in client_data.items():
            card = tf.data.experimental.cardinality(local_ds).numpy()
            n_samples = int(card) if card > 0 else sum(1 for _ in local_ds)
            metrics = ClientMetrics(
                local_validation_metric=float(rng.uniform(0.4, 0.9)),
                data_volume=n_samples,
                inference_latency=float(rng.uniform(0.01, 0.15)),
                last_selected_round=0,
            )
            clients[cid] = FederatedClient(
                client_id=cid,
                local_data=local_ds,
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

        logger.info(
            "FL Cycle (Flower): %d devices, %d rounds, %d local epochs",
            cfg.num_devices, cfg.global_rounds, cfg.local_epochs,
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
        )

        def _client_fn(cid: str):
            dataset = self.client_datasets[cid]
            return FLWRClient(
                client_id=cid,
                model_fn=lambda: tf.keras.models.clone_model(self.global_model),
                train_data=dataset,
                batch_size=cfg.local_batch_size,
            )

        start = time.time()
        fl.simulation.start_simulation(
            client_fn=_client_fn,
            num_clients=cfg.num_devices,
            config=fl.server.ServerConfig(num_rounds=cfg.global_rounds),
            strategy=strategy,
        )
        elapsed = time.time() - start
        logger.info("Cycle complete -> %.1fs", elapsed)

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

        self.history = strategy.history
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
