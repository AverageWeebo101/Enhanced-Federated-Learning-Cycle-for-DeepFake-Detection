"""
Legacy TFF module names mapped to native Flower execution.

This module preserves historical class/function names so existing
notebooks and scripts can keep importing:

- TFFCycleConfig
- TFFFederatedLearningCycle
- convert_to_tflite

Internally, execution is fully Flower-based via flwr_federated_cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import tensorflow as tf

from client_reputation_ledger import ReputationConfig
from enhanced_client_selection import SelectionWeights
from flwr_federated_cycle import (
    FLWRCycleConfig,
    FLWRFederatedLearningCycle,
    convert_to_tflite,
    generate_proxy_data,
    generate_synthetic_data,
    partition_data_iid_flwr,
)
from knowledge_distillation import DistillationConfig
from update_validation import ClippingConfig, ContributionWeights

logger = logging.getLogger(__name__)


@dataclass
class TFFCycleConfig:
    """
    Backward-compatible config surface using Flower underneath.

    Legacy TFF-only options are accepted and ignored so older notebook
    cells do not break.
    """

    model_path: str = "efficientnetb4_final.keras"
    num_devices: int = 100
    local_epochs: int = 5
    global_rounds: int = 50
    clients_per_round: int = 15
    local_batch_size: int = 32
    local_lr: float = 1e-4
    server_lr: float = 0.1
    eval_every: int = 5

    client_optimizer: str = "adam"
    server_optimizer: str = "sgd"
    enable_comparison: bool = False

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

    selection_weights: SelectionWeights = field(
        default_factory=lambda: SelectionWeights(
            w_v=0.30,
            w_d=0.20,
            w_l=0.10,
            w_r=0.25,
            w_s=0.15,
        )
    )

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

    reports_dir: str = "reports"
    tflite_output_path: str = "effnet_global_flwr_final.tflite"
    input_shape: Tuple[int, ...] = (224, 224, 3)

    def to_flwr_config(self) -> FLWRCycleConfig:
        return FLWRCycleConfig(
            model_path=self.model_path,
            num_devices=self.num_devices,
            local_epochs=self.local_epochs,
            global_rounds=self.global_rounds,
            clients_per_round=self.clients_per_round,
            local_batch_size=self.local_batch_size,
            local_lr=self.local_lr,
            eval_every=self.eval_every,
            enable_distillation=self.enable_distillation,
            distillation_config=self.distillation_config,
            selection_weights=self.selection_weights,
            contribution_weights=self.contribution_weights,
            clipping_config=self.clipping_config,
            harmful_threshold=self.harmful_threshold,
            reputation_config=self.reputation_config,
            reports_dir=self.reports_dir,
            tflite_output_path=self.tflite_output_path,
            input_shape=self.input_shape,
        )


class TFFFederatedLearningCycle(FLWRFederatedLearningCycle):
    """
    Backward-compatible class name executing native Flower rounds.
    """

    def __init__(
        self,
        config: Optional[TFFCycleConfig | FLWRCycleConfig] = None,
    ) -> None:
        if config is None:
            self.legacy_config = TFFCycleConfig()
            flwr_config = self.legacy_config.to_flwr_config()
        elif isinstance(config, TFFCycleConfig):
            self.legacy_config = config
            flwr_config = config.to_flwr_config()
        else:
            self.legacy_config = TFFCycleConfig(
                model_path=config.model_path,
                num_devices=config.num_devices,
                local_epochs=config.local_epochs,
                global_rounds=config.global_rounds,
                clients_per_round=config.clients_per_round,
                local_batch_size=config.local_batch_size,
                local_lr=config.local_lr,
                eval_every=config.eval_every,
                enable_distillation=config.enable_distillation,
                distillation_config=config.distillation_config,
                selection_weights=config.selection_weights,
                contribution_weights=config.contribution_weights,
                clipping_config=config.clipping_config,
                harmful_threshold=config.harmful_threshold,
                reputation_config=config.reputation_config,
                reports_dir=config.reports_dir,
                tflite_output_path=config.tflite_output_path,
                input_shape=config.input_shape,
            )
            flwr_config = config

        super().__init__(flwr_config)

    def setup_tff_process(self) -> None:
        """Legacy no-op kept for old notebook cells."""
        logger.info(
            "setup_tff_process() is deprecated and now a no-op. "
            "Flower is used natively for federated execution."
        )

    def run(
        self,
        server_val_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, list]:
        history = super().run(
            server_val_data=server_val_data,
            test_data=test_data,
            proxy_data=proxy_data,
            supervised_data=supervised_data,
        )
        if "tff_fedavg_accuracy" not in history:
            history["tff_fedavg_accuracy"] = [None] * len(history.get("round", []))
        return history


def partition_data_iid_tff(*args, **kwargs):
    """Legacy alias to the Flower IID partition helper."""
    return partition_data_iid_flwr(*args, **kwargs)


__all__ = [
    "TFFCycleConfig",
    "TFFFederatedLearningCycle",
    "convert_to_tflite",
    "generate_proxy_data",
    "generate_synthetic_data",
    "partition_data_iid_tff",
]
