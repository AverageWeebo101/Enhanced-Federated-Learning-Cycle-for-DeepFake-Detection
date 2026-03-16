"""
Legacy TFF data utilities mapped to Flower-compatible data pipelines.

This module keeps the old names used by notebook cells while removing
any TensorFlow Federated dependency.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from flwr_federated_cycle import (
    generate_proxy_data,
    generate_synthetic_data,
    partition_data_iid_flwr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def _require_tff() -> None:
    """
    Backward-compatible guard.

    The project no longer requires TFF; this function now exists only
    to avoid breaking legacy imports and call sites.
    """
    return None


class TFFDataManager:
    """
    Backward-compatible manager for federated dataset preparation.

    Methods return plain TensorFlow datasets that can be consumed by
    Flower clients.
    """

    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = input_shape

    def get_element_spec(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        return (
            tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )

    def get_unbatched_spec(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        return (
            tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

    def make_federated_data(
        self,
        client_datasets: Dict[str, tf.data.Dataset],
        selected_ids: List[str],
        batch_size: int = 32,
        local_epochs: int = 1,
        shuffle_buffer: int = 1000,
    ) -> List[tf.data.Dataset]:
        federated: List[tf.data.Dataset] = []
        for cid in selected_ids:
            if cid not in client_datasets:
                logger.warning("Client %s has no dataset, skipping.", cid)
                continue
            ds = (
                client_datasets[cid]
                .repeat(local_epochs)
                .shuffle(buffer_size=shuffle_buffer)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            federated.append(ds)
        return federated

    def create_tff_client_data(
        self,
        client_datasets: Dict[str, tf.data.Dataset],
    ):
        """
        Kept for API compatibility only.

        Flower does not require a TFF ClientData wrapper, so this now
        returns the input mapping unchanged.
        """
        return client_datasets

    @staticmethod
    def preprocess_dataset(
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        local_epochs: int = 1,
        shuffle_buffer: int = 1000,
    ) -> tf.data.Dataset:
        return (
            dataset
            .repeat(local_epochs)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


def partition_data_iid_tff(
    full_dataset: tf.data.Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[str, tf.data.Dataset]:
    """Legacy alias to Flower IID partitioning."""
    return partition_data_iid_flwr(
        full_dataset=full_dataset,
        num_clients=num_clients,
        seed=seed,
    )


__all__ = [
    "TFFDataManager",
    "_require_tff",
    "generate_proxy_data",
    "generate_synthetic_data",
    "partition_data_iid_tff",
]
