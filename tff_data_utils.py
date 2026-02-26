"""
TFF Data Utilities — Federated Dataset Management
===================================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Bridges existing ``tf.data.Dataset`` client partitions into
TensorFlow Federated (TFF) compatible data structures.

Provides
--------
* ``TFFDataManager``  — element specs, federated-data creation,
  optional wrapping into ``tff.simulation.datasets.ClientData``.
* ``partition_data_iid_tff``  — IID partition helper.
* ``generate_synthetic_data`` / ``generate_proxy_data``  — quick
  generators for smoke-testing.

Environment
-----------
Requires ``tensorflow-federated >= 0.48.0``.
See ``requirements_tff.txt`` for the exact compatible stack.
Recommended runtime: **Google Colab** (TFF pre-installed).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  Guard                                                                  #
# ====================================================================== #

def _require_tff() -> None:
    """Raise a clear error if neither the Flower adapter nor TFF is available."""
    if not TFF_AVAILABLE:
        raise RuntimeError(
            "Neither the Flower adapter (flwr_adapter) nor TensorFlow Federated\n"
            "is available in this environment.\n"
            "Install Flower with:  pip install flwr\n"
            "The flwr_adapter module provides a drop-in replacement for TFF.\n"
            "Alternatively, install TFF:  pip install tensorflow-federated==0.48.0"
        )


# ====================================================================== #
#  1.  TFF DATA MANAGER                                                   #
# ====================================================================== #

class TFFDataManager:
    """
    Manages federated data for TFF integration.

    Converts per-client ``tf.data.Dataset`` partitions into the formats
    expected by ``tff.learning.algorithms`` and provides the
    ``element_spec`` / ``input_spec`` required by
    ``tff.learning.models.from_keras_model()``.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Spatial input shape *without* the batch dimension,
        e.g. ``(224, 224, 3)`` for the EfficientNet model.
    """

    def __init__(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = input_shape

    # ------------------------------------------------------------------ #
    #  Spec helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_element_spec(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        """
        Return the **batched** element spec for ``from_keras_model()``.

        Matches datasets that yield ``(images, labels)`` with a leading
        ``None`` batch dimension.
        """
        return (
            tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )

    def get_unbatched_spec(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        """Per-example spec (no batch dim) — useful for type annotations."""
        return (
            tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

    # ------------------------------------------------------------------ #
    #  Federated data creation (for process.next())                       #
    # ------------------------------------------------------------------ #

    def make_federated_data(
        self,
        client_datasets: Dict[str, tf.data.Dataset],
        selected_ids: List[str],
        batch_size: int = 32,
        local_epochs: int = 1,
        shuffle_buffer: int = 1000,
    ) -> List[tf.data.Dataset]:
        """
        Create a **list of batched datasets** for TFF's
        ``process.next(state, federated_data)``.

        Parameters
        ----------
        client_datasets : dict[str, tf.data.Dataset]
            Pre-partitioned datasets keyed by client ID.
        selected_ids : list[str]
            Client IDs chosen for this round (from Part 1).
        batch_size : int
            Batch size for local training.
        local_epochs : int
            Number of local training epochs — implemented via
            ``dataset.repeat(local_epochs)`` which is the standard
            TFF pattern.
        shuffle_buffer : int
            Buffer size for per-epoch shuffling.

        Returns
        -------
        list[tf.data.Dataset]
            One **batched** dataset per selected client.
        """
        federated: List[tf.data.Dataset] = []
        for cid in selected_ids:
            if cid not in client_datasets:
                logger.warning("Client %s has no dataset — skipping.", cid)
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

    # ------------------------------------------------------------------ #
    #  TFF ClientData wrapper (optional — for TFF simulation tools)       #
    # ------------------------------------------------------------------ #

    def create_tff_client_data(
        self,
        client_datasets: Dict[str, tf.data.Dataset],
    ):
        """
        Wrap per-client datasets into ``tff.simulation.datasets.ClientData``
        for advanced TFF simulation tools (e.g. ``ClientData.preprocess``,
        sampling utilities, etc.).

        Returns
        -------
        tff.simulation.datasets.ClientData
        """
        _require_tff()

        client_ids = sorted(client_datasets.keys())
        local_ref = client_datasets  # captured by closure

        def create_dataset_fn(client_id):
            cid = (
                client_id.numpy().decode("utf-8")
                if isinstance(client_id, tf.Tensor)
                else client_id
            )
            return local_ref[cid]

        return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
            client_ids=client_ids,
            serializable_dataset_fn=create_dataset_fn,
        )

    # ------------------------------------------------------------------ #
    #  Preprocessing pipeline                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess_dataset(
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        local_epochs: int = 1,
        shuffle_buffer: int = 1000,
    ) -> tf.data.Dataset:
        """
        Standard preprocessing pipeline applied to each client dataset
        before passing to TFF.
        """
        return (
            dataset
            .repeat(local_epochs)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )


# ====================================================================== #
#  2.  PARTITIONING HELPERS                                               #
# ====================================================================== #

def partition_data_iid_tff(
    full_dataset: tf.data.Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[str, tf.data.Dataset]:
    """
    IID partition: shuffle the dataset and split evenly across clients.

    Each shard is a ``tf.data.Dataset`` yielding ``(image, label)`` pairs.

    Parameters
    ----------
    full_dataset : tf.data.Dataset
        The complete labelled dataset.
    num_clients : int
        Number of federated client partitions.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    dict[str, tf.data.Dataset]
        ``{client_id: local_dataset}``
    """
    all_data = list(full_dataset.shuffle(buffer_size=10_000, seed=seed))
    total = len(all_data)
    shard_size = max(1, total // num_clients)

    partitions: Dict[str, tf.data.Dataset] = {}
    for i in range(num_clients):
        cid = f"client_{i:03d}"
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
#  3.  SYNTHETIC DATA GENERATORS  (for smoke-testing)                     #
# ====================================================================== #

def generate_synthetic_data(
    num_samples: int,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Synthetic labelled dataset ``(image, label)`` for smoke-testing.

    Replace with real FF++ c23 data loaders in production.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32) * 0.1
    y = rng.randint(0, 2, size=(num_samples,)).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y))


def generate_proxy_data(
    num_samples: int,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Unlabelled proxy data ``(image,)`` for knowledge distillation."""
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32) * 0.1
    return tf.data.Dataset.from_tensor_slices(x)


# ====================================================================== #
#  DEMO / SMOKE-TEST                                                      #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  TFF Data Utilities — Demo  ===\n")

    INPUT_SHAPE = (16,)          # tiny for fast demo
    NUM_CLIENTS = 8
    SAMPLES = NUM_CLIENTS * 30   # 30 per client

    # ---- 1. Synthetic dataset ---------------------------------------- #
    full_ds = generate_synthetic_data(SAMPLES, INPUT_SHAPE, seed=10)
    print(f"Full dataset: {SAMPLES} samples, shape {INPUT_SHAPE}")

    # ---- 2. Partition ------------------------------------------------ #
    client_data = partition_data_iid_tff(full_ds, NUM_CLIENTS)
    for cid, ds in sorted(client_data.items()):
        n = sum(1 for _ in ds)
        print(f"  {cid}: {n} samples")

    # ---- 3. TFF data manager ---------------------------------------- #
    dm = TFFDataManager(input_shape=INPUT_SHAPE)

    print(f"\nElement spec (batched): {dm.get_element_spec()}")
    print(f"Unbatched spec:        {dm.get_unbatched_spec()}")

    # ---- 4. Make federated data for a round -------------------------- #
    selected = ["client_001", "client_003", "client_005"]
    fed_data = dm.make_federated_data(
        client_data, selected, batch_size=16, local_epochs=2,
    )
    print(f"\nFederated data for {selected}: {len(fed_data)} datasets")
    for i, ds in enumerate(fed_data):
        n_batches = sum(1 for _ in ds)
        print(f"  Client {selected[i]}: {n_batches} batches (2 epochs)")

    # ---- 5. TFF ClientData (requires TFF) ---------------------------- #
    if TFF_AVAILABLE:
        tff_cd = dm.create_tff_client_data(client_data)
        print(f"\nTFF ClientData: {len(tff_cd.client_ids)} clients")
        print(f"  Client IDs: {tff_cd.client_ids[:5]} ...")
    else:
        print(
            "\n⚠  TFF not installed — skipping ClientData creation.\n"
            "   Install: pip install tensorflow-federated==0.48.0\n"
            "   See requirements_tff.txt for full compatible stack.\n"
            "   Recommended runtime: Google Colab."
        )

    # ---- 6. Proxy data ----------------------------------------------- #
    proxy = generate_proxy_data(50, INPUT_SHAPE, seed=20)
    print(f"\nProxy data: 50 unlabelled samples, shape {INPUT_SHAPE}")

    print("\nDone.")
