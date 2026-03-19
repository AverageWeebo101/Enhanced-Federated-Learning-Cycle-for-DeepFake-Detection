"""

TFF Federated Learning Cycle — Main Orchestrator  (RAM-optimised)

==================================================================

Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)



Integrates all five enhancement modules into one end-to-end pipeline:



 1. **Enhanced Client Selection**   (``enhanced_client_selection.py``)

 2. **Update Validation & Weighing** (``update_validation.py``)

 3. **Knowledge Distillation**      (``knowledge_distillation.py``)

 4. **Client Reputation Ledger**    (``client_reputation_ledger.py``)

 5. **Evaluation Metrics**          (``evaluation_metrics.py``)



RAM-optimisation changes (vs. original):

  • Eliminated the redundant TFF/Flower federated round that

    duplicated all training.  Only the manual per-client loop is

    used, saving ~5-10 GB of model clones per round.

  • Removed the ``_comparison_model`` clone (comparison mode was

    only re-evaluating the same FedAvg result).

  • Added explicit ``del`` / ``gc.collect()`` between pipeline

    stages to release weight arrays promptly.

  • Client deltas are freed immediately after aggregation.

"""

from __future__ import annotations


import gc

import logging

import os

import time

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple


import numpy as np

import tensorflow as tf

# ---------- Existing modules  (Parts 1–5) ----------------------------- #

from enhanced_client_selection import (  # Part 1
    ClientMetrics,
    FederatedClient,
    ReputationLedger,
    SelectionWeights,
    EnhancedClientSelector,
)

from update_validation import (  # Part 2
    ContributionWeights,
    ClippingConfig,
    ClientUpdateRecord,
    UpdateValidator,
)

from knowledge_distillation import (  # Part 3
    DistillationConfig,
    run_distillation_round,
)

from client_reputation_ledger import (  # Part 4
    ReputationConfig,
    ClientReputationLedger,
    update_ledger_from_records,
)

from evaluation_metrics import (  # Part 5
    FederatedModelEvaluator,
    evaluate_and_report,
)

# ---------------------------------------------------------------------------

# Logging

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

logger = logging.getLogger(__name__)


# ====================================================================== #

#  1.  CONFIGURATION                                                      #

# ====================================================================== #


@dataclass
class TFFCycleConfig:
    """

    Central configuration for the Federated Learning cycle.

    """

    # -- Core FL settings ---------------------------------------------- #

    model_path: str = "efficientnetb4_final.keras"

    num_devices: int = 100

    local_epochs: int = 5

    global_rounds: int = 50

    clients_per_round: int = 15

    local_batch_size: int = 32

    local_lr: float = 1e-4

    server_lr: float = 0.1

    eval_every: int = 5

    # -- TFF process settings (kept for config compat) ----------------- #

    client_optimizer: str = "adam"

    server_optimizer: str = "sgd"

    # -- Comparison mode (disabled to save RAM) ------------------------ #

    enable_comparison: bool = False

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

    # -- Reputation (Part 4) ------------------------------------------ #

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

    tflite_output_path: str = "effnet_global_tff_final.tflite"

    input_shape: Tuple[int, ...] = (224, 224, 3)


# ====================================================================== #

#  2.  TF LITE CONVERSION                                                 #

# ====================================================================== #


def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantise: bool = False,
) -> str:
    """Convert a Keras model to TF Lite format."""

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantise:

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_bytes = converter.convert()

    Path(output_path).write_bytes(tflite_bytes)

    size_mb = len(tflite_bytes) / (1024 * 1024)

    logger.info(
        "TF Lite model saved → %s  (%.2f MB, quantised=%s)",
        output_path,
        size_mb,
        quantise,
    )

    return output_path


# ====================================================================== #

#  3.  FEDERATED LEARNING CYCLE (RAM-optimised)                           #

# ====================================================================== #


class TFFFederatedLearningCycle:
    """

    End-to-end Federated Learning cycle integrating all five

    enhancement modules.



    RAM-optimised: uses a single cached model for local training

    instead of cloning per-client via TFF/Flower.

    """

    def __init__(self, config: Optional[TFFCycleConfig] = None) -> None:

        self.config = config or TFFCycleConfig()

        self.global_model: Optional[tf.keras.Model] = None

        self.clients: List[FederatedClient] = []

        self.client_datasets: Dict[str, tf.data.Dataset] = {}

        # Cached reusable model (avoid clone+build+compile per call)

        self._local_model: Optional[tf.keras.Model] = None

        # Enhancement components (Parts 1–5)

        self.reputation_ledger: Optional[ClientReputationLedger] = None

        self.basic_ledger: Optional[ReputationLedger] = None

        self.selector: Optional[EnhancedClientSelector] = None

        self.validator: Optional[UpdateValidator] = None

        self.evaluator: Optional[FederatedModelEvaluator] = None

        # History

        self.history: Dict[str, list] = {
            "round": [],
            "tff_fedavg_accuracy": [],
            "enhanced_accuracy": [],
            "selected_clients": [],
            "num_accepted": [],
            "num_rejected": [],
            "distillation_loss": [],
        }

    # ------------------------------------------------------------------ #

    #  Initialisation                                                     #

    # ------------------------------------------------------------------ #

    def load_global_model(self) -> tf.keras.Model:
        """Load the pre-trained EfficientNet model."""

        import shutil, h5py

        cfg = self.config

        logger.info("Loading global model from %s …", cfg.model_path)

        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as _effnet_preprocess,
        )

        from tensorflow.keras.applications import EfficientNetB2

        def _build_model(input_shape=(260, 260, 3)):

            base = EfficientNetB2(
                include_top=False, weights=None, input_shape=input_shape
            )

            x = tf.keras.layers.GlobalAveragePooling2D()(base.output)

            x = tf.keras.layers.Dropout(0.3)(x)

            x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

            return tf.keras.Model(inputs=base.input, outputs=x)

        def _is_hdf5(path):
            """Check if a file is HDF5 by reading its magic bytes."""

            try:

                with open(path, "rb") as fh:

                    return fh.read(4) == b"\\x89HDF"

            except OSError:

                return False

        model = None

        model_path = cfg.model_path

        # ── Strategy 0: Native Keras 3 .keras format (zip-based) ──

        # Native .keras files are NOT HDF5 — skip h5py entirely.

        if not _is_hdf5(model_path):

            try:

                model = tf.keras.models.load_model(model_path, compile=False)

                logger.info("Loaded as native Keras 3 model (.keras format).")

            except Exception as e:

                logger.warning("Native .keras load_model() failed: %s", e)

        # ── HDF5-based strategies (only if the file is actually HDF5) ──

        if model is None and _is_hdf5(model_path):

            with h5py.File(model_path, "r") as f:

                top_keys = list(f.keys())

            is_full_model = any(
                k in top_keys
                for k in ("model_weights", "model_config", "keras_version")
            )

            # Strategy 1: full saved model (rename away from .weights.h5 if needed)

            if is_full_model:

                load_path = model_path

                tmp_path = None

                if model_path.endswith(".weights.h5"):

                    tmp_path = model_path.replace(".weights.h5", "_full_tmp.h5")

                    shutil.copy(model_path, tmp_path)

                    load_path = tmp_path

                try:

                    _custom = {"preprocess_input": _effnet_preprocess}

                    model = tf.keras.models.load_model(
                        load_path, custom_objects=_custom, compile=False
                    )

                    logger.info("Loaded as full saved model.")

                except Exception as e:

                    logger.warning("load_model() failed: %s", e)

                finally:

                    if tmp_path and os.path.exists(tmp_path):

                        os.remove(tmp_path)

            # Strategy 2: weights-only (Keras 3 .weights.h5 format)

            if model is None:

                model = _build_model()

                try:

                    model.load_weights(model_path)

                    logger.info("Weights loaded (Keras 3 format).")

                except Exception:

                    try:

                        model.load_weights(model_path, skip_mismatch=True)

                        logger.info("Weights loaded with skip_mismatch=True.")

                    except Exception:

                        model = None

            # Strategy 3: legacy HDF5 weights (rename to .h5)

            if model is None and model_path.endswith(".weights.h5"):

                model = _build_model()

                legacy_path = model_path.replace(".weights.h5", "_legacy_tmp.h5")

                shutil.copy(model_path, legacy_path)

                try:

                    model.load_weights(legacy_path)

                    logger.info("Weights loaded via legacy HDF5.")

                except Exception:

                    try:

                        model.load_weights(legacy_path, skip_mismatch=True)

                        logger.info("Legacy weights loaded with skip_mismatch=True.")

                    except Exception:

                        model = None

                finally:

                    if os.path.exists(legacy_path):

                        os.remove(legacy_path)

        if model is None:

            raise RuntimeError(
                f"Could not load model from {model_path}. "
                f"Check that the file is a valid Keras model and the architecture/version match."
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.local_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(
            "Global model loaded — %s params, input shape %s",
            f"{model.count_params():,}",
            model.input_shape,
        )

        self.global_model = model

        return model

    def create_clients(
        self,
        client_data: Dict[str, tf.data.Dataset],
    ) -> List[FederatedClient]:
        """Create ``FederatedClient`` objects and store the dataset dict."""

        rng = np.random.RandomState(42)

        clients: List[FederatedClient] = []

        for cid, local_ds in client_data.items():

            card = tf.data.experimental.cardinality(local_ds).numpy()

            n_samples = int(card) if card > 0 else sum(1 for _ in local_ds)

            metrics = ClientMetrics(
                local_validation_metric=float(rng.uniform(0.4, 0.9)),
                data_volume=n_samples,
                inference_latency=float(rng.uniform(0.01, 0.15)),
                last_selected_round=0,
            )

            clients.append(
                FederatedClient(
                    client_id=cid,
                    local_data=local_ds,
                    metrics=metrics,
                )
            )

        logger.info("Created %d federated clients.", len(clients))

        self.clients = clients

        self.client_datasets = client_data

        return clients

    def setup_tff_process(self) -> None:
        """

        Kept as a no-op for API compatibility.



        The TFF/Flower round has been removed to save ~5-10 GB RAM.

        All training now goes through the single-model _local_train() path.

        """

        logger.info(
            "setup_tff_process(): skipped (RAM-optimised mode — "
            "using manual local training only)."
        )

    def setup_enhancement_modules(self) -> None:
        """Wire Parts 1–5 (same logic as federated_learning_cycle.py)."""

        cfg = self.config

        assert self.global_model is not None

        assert len(self.clients) > 0

        # Part 4: Reputation ledger

        self.reputation_ledger = ClientReputationLedger(
            config=cfg.reputation_config,
        )

        for c in self.clients:

            self.reputation_ledger.register(c.client_id)

        self.basic_ledger = self.reputation_ledger.as_basic_ledger()

        # Part 1: Client selector

        self.selector = EnhancedClientSelector(
            clients=self.clients,
            reputation_ledger=self.basic_ledger,
            weights=cfg.selection_weights,
            target_k=cfg.clients_per_round,
        )

        # Part 2: Update validator

        self.validator = UpdateValidator(
            global_model=self.global_model,
            reputation_ledger=self.basic_ledger,
            weights=cfg.contribution_weights,
            clipping=cfg.clipping_config,
            harmful_threshold=cfg.harmful_threshold,
            batch_size=cfg.local_batch_size,
        )

        # Part 5: Evaluator

        self.evaluator = FederatedModelEvaluator(
            model=self.global_model,
            model_name="effnet_global_tff",
            reports_dir=cfg.reports_dir,
        )

        logger.info("Enhancement modules (Parts 1–5) initialised.")

    # ------------------------------------------------------------------ #

    #  Local training (single cached model — RAM efficient)               #

    # ------------------------------------------------------------------ #

    def _local_train(
        self,
        client: FederatedClient,
        global_weights: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int, float]:
        """

        Local training using a single cached model clone.



        Returns

        -------

        updated_weights, data_volume, local_accuracy

        """

        cfg = self.config

        # Lazily create & cache the local training model

        if self._local_model is None:

            self._local_model = tf.keras.models.clone_model(self.global_model)

            self._local_model.build(self.global_model.input_shape)

            self._local_model.compile(
                optimizer=tf.keras.optimizers.Adam(cfg.local_lr),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

        self._local_model.set_weights(global_weights)

        # Re-compile to create a fresh optimizer with zeroed moment

        # estimates; prevents stale Adam state from degrading training.

        self._local_model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.local_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        if client.local_data is None:

            return global_weights, 0, 0.0

        dataset = client.local_data.batch(cfg.local_batch_size).prefetch(1)

        self._local_model.fit(dataset, epochs=cfg.local_epochs, verbose=0)

        result = self._local_model.evaluate(dataset, verbose=0, return_dict=True)

        local_acc = result.get("accuracy", 0.0)

        return self._local_model.get_weights(), client.metrics.data_volume, local_acc

    # ------------------------------------------------------------------ #

    #  Reputation sync                                                    #

    # ------------------------------------------------------------------ #

    def _sync_reputation_to_basic_ledger(self) -> None:
        """Copy extended ledger → basic ledger used by Parts 1 & 2."""

        updated_basic = self.reputation_ledger.as_basic_ledger()

        self.basic_ledger._scores = updated_basic._scores

    # ------------------------------------------------------------------ #

    #  Single round  (RAM-optimised — no TFF/Flower round)                #

    # ------------------------------------------------------------------ #

    def execute_round(
        self,
        current_round: int,
        server_val_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, Any]:
        """

        Execute one complete enhanced federated round.



        Pipeline:

          1. Client selection (Part 1)

          2. Per-client local training (single cached model)

          3. Update validation & contribution aggregation (Part 2)

          4. Knowledge distillation (Part 3)

          5. Reputation update (Part 4)

          6. Accuracy check

        """

        cfg = self.config

        logger.info("── Round %d / %d ──", current_round, cfg.global_rounds)

        # ── 1. Client selection  (Part 1) ──────────────────────────── #

        selected: List[FederatedClient] = self.selector.select(
            current_round=current_round,
        )

        selected_ids = [c.client_id for c in selected]

        # Save pre-round global weights

        global_weights_before = self.global_model.get_weights()

        # ── 2. Per-client local training ──────────────────────────── #

        client_updates: Dict[str, List[np.ndarray]] = {}

        data_volumes: Dict[str, int] = {}

        for client in selected:

            updated_w, n, local_acc = self._local_train(client, global_weights_before)

            client_updates[client.client_id] = updated_w

            data_volumes[client.client_id] = n

            client.metrics.local_validation_metric = local_acc

        # ── 3. Update validation & contribution aggregation (Part 2) ─ #

        records: List[ClientUpdateRecord] = self.validator.validate_updates(
            client_updates=client_updates,
            data_volumes=data_volumes,
            server_val_data=server_val_data,
        )

        enhanced_weights = self.validator.aggregate_weighted(
            records,
            global_weights_before,
        )

        num_accepted = sum(1 for r in records if not r.rejected)

        num_rejected = sum(1 for r in records if r.rejected)

        # Apply enhanced weights to global model

        self.global_model.set_weights(enhanced_weights)

        self.validator.global_model.set_weights(enhanced_weights)

        # ── 4. Knowledge distillation  (Part 3) ───────────────────── #

        distill_loss = None

        if cfg.enable_distillation and proxy_data is not None:

            contribution_weights = {
                r.client_id: r.contribution_weight
                for r in records
                if not r.rejected and r.contribution_weight > 0
            }

            if len(contribution_weights) >= 1:

                kd_history = run_distillation_round(
                    global_model=self.global_model,
                    client_weights={
                        cid: client_updates[cid] for cid in contribution_weights
                    },
                    contribution_weights=contribution_weights,
                    proxy_data=proxy_data,
                    supervised_data=supervised_data,
                    config=cfg.distillation_config,
                )

                distill_loss = kd_history.get("loss_total", [None])[-1]

        # ── Free weight arrays that are no longer needed ──────────── #

        for rec in records:

            rec.delta = None  # free ~36 MB per client

        del client_updates, enhanced_weights

        gc.collect()

        # ── 5. Reputation update  (Part 4) ────────────────────────── #

        update_ledger_from_records(
            self.reputation_ledger,
            records,
            current_round,
        )

        self.validator.update_reputations(records)

        self._sync_reputation_to_basic_ledger()

        # ── 6. Accuracy check ─────────────────────────────────────── #

        enhanced_result = self.global_model.evaluate(
            server_val_data.batch(cfg.local_batch_size),
            verbose=0,
            return_dict=True,
        )

        enhanced_acc = enhanced_result.get("accuracy", 0.0)

        logger.info(
            "R%d  Enhanced=%.4f  acc=%d rej=%d",
            current_round,
            enhanced_acc,
            num_accepted,
            num_rejected,
        )

        return {
            "round": current_round,
            "selected": selected_ids,
            "tff_fedavg_accuracy": None,
            "enhanced_accuracy": enhanced_acc,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "records": records,
            "distillation_loss": distill_loss,
            "tff_metrics": None,
        }

    # ------------------------------------------------------------------ #

    #  Full cycle                                                         #

    # ------------------------------------------------------------------ #

    def run(
        self,
        server_val_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, list]:
        """

        Run the full Federated Learning cycle.

        """

        cfg = self.config

        logger.info(
            "FL Cycle: %d devices, %d rounds, %d local epochs",
            cfg.num_devices,
            cfg.global_rounds,
            cfg.local_epochs,
        )

        # -- Baseline evaluation --------------------------------------- #

        baseline_report = self.evaluator.evaluate(
            test_data=test_data,
            batch_size=cfg.local_batch_size,
            federated_round=0,
            extra_info={"stage": "baseline", "eval_mode": "lightweight"},
            full_metrics=False,
            run_latency=False,
        )

        self.evaluator.save_report(baseline_report, tag="round_000_baseline_tff")

        logger.info(
            "Baseline — Acc: %.4f, F1: %.4f, AUC: %.4f",
            baseline_report.classification.accuracy,
            baseline_report.classification.f1_macro,
            baseline_report.classification.roc_auc,
        )

        all_reports = [baseline_report]

        cycle_start = time.time()

        # ============================================================== #

        #  MAIN LOOP                                                      #

        # ============================================================== #

        for t in range(1, cfg.global_rounds + 1):

            round_start = time.time()

            info = self.execute_round(
                current_round=t,
                server_val_data=server_val_data,
                proxy_data=proxy_data,
                supervised_data=supervised_data,
            )

            # Record history

            self.history["round"].append(t)

            self.history["tff_fedavg_accuracy"].append(info["tff_fedavg_accuracy"])

            self.history["enhanced_accuracy"].append(info["enhanced_accuracy"])

            self.history["selected_clients"].append(info["selected"])

            self.history["num_accepted"].append(info["num_accepted"])

            self.history["num_rejected"].append(info["num_rejected"])

            self.history["distillation_loss"].append(info["distillation_loss"])

            round_elapsed = time.time() - round_start

            logger.info("  └─ %.1fs", round_elapsed)

            # -- Periodic full evaluation (Part 5) --------------------- #

            is_eval_round = t % cfg.eval_every == 0 or t == 1 or t == cfg.global_rounds

            if is_eval_round:

                report = self.evaluator.evaluate(
                    test_data=test_data,
                    batch_size=cfg.local_batch_size,
                    federated_round=t,
                    extra_info={
                        "enhanced_acc": info["enhanced_accuracy"],
                        "accepted": info["num_accepted"],
                        "rejected": info["num_rejected"],
                    },
                )

                self.evaluator.save_report(report, tag=f"tff_round_{t:03d}")

                all_reports.append(report)

                logger.info(
                    "  Eval R%d — Acc=%.4f  F1=%.4f  AUC=%.4f",
                    t,
                    report.classification.accuracy,
                    report.classification.f1_macro,
                    report.classification.roc_auc,
                )

        total_elapsed = time.time() - cycle_start

        # ============================================================== #

        #  POST-CYCLE                                                     #

        # ============================================================== #

        logger.info(
            "Cycle complete — %d rounds in %.1fs", cfg.global_rounds, total_elapsed
        )

        # Comparison report

        if len(all_reports) > 1:

            self.evaluator.save_comparison_report(all_reports)

        # Save reputation ledger

        ledger_path = Path(cfg.reports_dir) / "reputation_ledger_tff_final.json"

        self.reputation_ledger.save(str(ledger_path))

        # TF Lite export

        convert_to_tflite(self.global_model, cfg.tflite_output_path, quantise=False)

        convert_to_tflite(
            self.global_model,
            cfg.tflite_output_path.replace(".tflite", "_quantised.tflite"),
            quantise=True,
        )

        # Final summary

        self._print_summary()

        return self.history

    # ------------------------------------------------------------------ #

    #  Summary                                                            #

    # ------------------------------------------------------------------ #

    def _print_summary(self) -> None:
        """Print a compact final training summary."""

        h = self.history

        if not h["round"]:

            return

        best_idx = int(np.argmax(h["enhanced_accuracy"]))

        best_round = h["round"][best_idx]

        best_acc = h["enhanced_accuracy"][best_idx]

        final_acc = h["enhanced_accuracy"][-1]

        stats = self.reputation_ledger.statistics()

        lines = [
            "FINAL SUMMARY",
            f"  Rounds: {len(h['round'])}  |  Best acc: {best_acc:.4f} (R{best_round})  |  Final acc: {final_acc:.4f}  |  Mean acc: {np.mean(h['enhanced_accuracy']):.4f}",
        ]

        lines.append(
            f"  Accepted: {sum(h['num_accepted'])}  Rejected: {sum(h['num_rejected'])}  Rep μ={stats.get('mean_reputation', 0):.4f} σ={stats.get('std_reputation', 0):.4f}"
        )

        kd = [l for l in h["distillation_loss"] if l is not None]

        if kd:

            lines.append(f"  Distillation loss: {np.mean(kd):.5f}")

        lines.append(f"  TFLite: {self.config.tflite_output_path}")

        print("\n".join(lines))
