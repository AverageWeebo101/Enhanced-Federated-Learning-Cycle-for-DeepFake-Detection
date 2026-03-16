"""
Server-Side Knowledge Distillation  (RAM-optimised)
=====================================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

RAM-optimisation changes (vs. original):
  • TeacherEnsemble uses ONE cached logit model and iterates through
    client weights via set_weights() instead of cloning 10-15 models
    (~2-3 GB saved).
  • Teacher logits are computed per-batch during the training loop
    instead of pre-computing all proxy data into numpy (~0.8 GB saved).
  • The student logit model is built once and reused across epochs.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------- shared types from Parts 1 & 2 ----------------------------- #
from enhanced_client_selection import FederatedClient, ReputationLedger

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
class DistillationConfig:
    """
    Hyper-parameters for server-side knowledge distillation.

    Parameters
    ----------
    temperature : float
        Softmax temperature ``T`` — higher values produce softer
        probability distributions, transferring more "dark knowledge".
    lam : float
        Interpolation weight ``λ`` between the distillation loss and the
        optional supervised cross-entropy loss:
        ``L_total = λ · L_KD  +  (1 − λ) · L_sup``
    epochs : int
        Number of distillation training epochs.
    batch_size : int
        Batch size for iterating over proxy / supervised data.
    learning_rate : float
        Learning rate for the distillation optimiser.
    """
    temperature: float = 3.0
    lam: float = 0.7
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4


# ====================================================================== #
#  2.  TEACHER LOGIT BUILDER  (RAM-optimised: single cached model)        #
# ====================================================================== #

class TeacherEnsemble:
    """
    Builds a *virtual* teacher by computing the contribution-weighted
    average of per-client logits for every proxy input.

    RAM-optimised: uses ONE cached logit model and iterates through
    client weights via ``set_weights()`` instead of cloning N models.
    """

    def __init__(
        self,
        global_model: tf.keras.Model,
        client_weights: Dict[str, List[np.ndarray]],
        contribution_weights: Dict[str, float],
    ) -> None:
        self.global_model = global_model
        # Filter to clients that actually contribute
        self.client_weights = {
            cid: w for cid, w in client_weights.items()
            if contribution_weights.get(cid, 0.0) > 0
        }
        self.contribution_weights = {
            cid: contribution_weights[cid]
            for cid in self.client_weights
        }
        total_c = sum(self.contribution_weights.values())
        # Normalise so weights sum to 1
        self._norm_weights = {
            cid: c / total_c for cid, c in self.contribution_weights.items()
        }
        # Build ONE cached logit model (instead of N clones)
        self._logit_model = self._rebuild_with_linear_output(self.global_model)
        logger.debug(
            "TeacherEnsemble: %d client(s), single cached logit model",
            len(self.client_weights),
        )

    # ------------------------------------------------------------------ #
    #  Logit-model builder                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rebuild_with_linear_output(
        ref_model: tf.keras.Model,
    ) -> tf.keras.Model:
        """
        Rebuild *ref_model* identically, except the last ``Dense`` layer
        uses ``activation='linear'`` so the output is raw logits.
        """
        cloned = tf.keras.models.clone_model(ref_model)
        cloned.build(ref_model.input_shape)
        cloned.set_weights(ref_model.get_weights())

        # Override the last Dense layer's activation to linear
        for layer in reversed(cloned.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                layer.activation = tf.keras.activations.linear
                break

        return cloned

    def _build_logit_model(
        self,
        weights: List[np.ndarray],
    ) -> tf.keras.Model:
        """Set weights on the cached logit model and return it."""
        self._logit_model.set_weights(weights)
        return self._logit_model

    # ------------------------------------------------------------------ #
    #  Compute teacher logits for a batch (single-model iteration)        #
    # ------------------------------------------------------------------ #

    def compute_teacher_logits_batch(
        self,
        x_batch: tf.Tensor,
        temperature: float = 1.0,
    ) -> tf.Tensor:
        """
        Return the contribution-weighted average teacher logits for
        *x_batch*.  Uses the single cached logit model, cycling through
        client weights via set_weights().

        Temperature must match the student's temperature so both sides
        of the KL divergence use the same softening.
        """
        weighted_logits = None
        for cid, w in self.client_weights.items():
            self._logit_model.set_weights(w)
            logits = self._logit_model(x_batch, training=False)
            if logits.shape[-1] == 1:
                probs = tf.sigmoid(logits / temperature)
            else:
                probs = tf.nn.softmax(logits / temperature)
            scaled = tf.cast(probs, tf.float32) * self._norm_weights[cid]
            if weighted_logits is None:
                weighted_logits = scaled
            else:
                weighted_logits = weighted_logits + scaled
        return weighted_logits

    # ------------------------------------------------------------------ #
    #  Pre-compute teacher logits (kept for API compat, but now streamed) #
    # ------------------------------------------------------------------ #

    def precompute_teacher_logits(
        self,
        proxy_data: tf.data.Dataset,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute teacher logits for every sample in *proxy_data*.
        Uses the single cached logit model to save RAM.
        """
        all_inputs: List[np.ndarray] = []
        all_logits: List[np.ndarray] = []

        batched = proxy_data.batch(batch_size).prefetch(1)
        for batch in batched:
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0]
            else:
                x_batch = batch

            teacher_logits = self.compute_teacher_logits_batch(x_batch)
            all_inputs.append(x_batch.numpy())
            all_logits.append(teacher_logits.numpy())

        return np.concatenate(all_inputs), np.concatenate(all_logits)


# ====================================================================== #
#  3.  DISTILLATION LOSSES                                                #
# ====================================================================== #

def distillation_loss(
    teacher_logits: tf.Tensor,
    student_logits: tf.Tensor,
    temperature: float,
) -> tf.Tensor:
    """
    Compute the knowledge-distillation loss:

        ``L_KD = T² · KL( softmax(z_teach / T)  ‖  softmax(z_stud / T) )``

    Works for both multi-class (last dim > 1) and binary (last dim = 1).
    """
    # Ensure consistent dtype (mixed-precision may produce float16 logits)
    teacher_logits = tf.cast(teacher_logits, tf.float32)
    student_logits = tf.cast(student_logits, tf.float32)

    num_outputs = teacher_logits.shape[-1]

    if num_outputs == 1:
        # Binary case: use sigmoid + binary KL
        p_teach = teacher_logits
        p_stud  = tf.sigmoid(student_logits / temperature)
        eps = 1e-7
        p_teach = tf.clip_by_value(p_teach, eps, 1.0 - eps)
        p_stud  = tf.clip_by_value(p_stud,  eps, 1.0 - eps)
        kl = (
            p_teach * tf.math.log(p_teach / p_stud)
            + (1.0 - p_teach) * tf.math.log((1.0 - p_teach) / (1.0 - p_stud))
        )
        return temperature ** 2 * tf.reduce_mean(kl)
    else:
        # Multi-class case: standard KL on softmax outputs
        p_teach = tf.nn.softmax(teacher_logits / temperature)
        log_p_stud = tf.nn.log_softmax(student_logits / temperature)
        kl = tf.reduce_sum(
            p_teach * (tf.math.log(p_teach + 1e-12) - log_p_stud),
            axis=-1,
        )
        return temperature ** 2 * tf.reduce_mean(kl)


def supervised_loss(
    student_logits: tf.Tensor,
    labels: tf.Tensor,
) -> tf.Tensor:
    """
    Standard supervised cross-entropy loss.
    """
    # Ensure consistent dtype (mixed-precision may produce float16 logits)
    student_logits = tf.cast(student_logits, tf.float32)

    num_outputs = student_logits.shape[-1]
    if num_outputs == 1:
        labels = tf.cast(labels, tf.float32)
        if len(labels.shape) < len(student_logits.shape):
            labels = tf.expand_dims(labels, -1)
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                labels, student_logits, from_logits=True,
            )
        )
    else:
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, student_logits, from_logits=True,
            )
        )


# ====================================================================== #
#  4.  KNOWLEDGE DISTILLATION ENGINE  (RAM-optimised)                     #
# ====================================================================== #

class KnowledgeDistiller:
    """
    Performs server-side knowledge distillation from a weighted ensemble
    of client models (teacher) into the aggregated global model (student).

    RAM-optimised: streams teacher logits per-batch instead of
    pre-computing all into numpy arrays.
    """

    def __init__(
        self,
        global_model: tf.keras.Model,
        teacher: TeacherEnsemble,
        config: Optional[DistillationConfig] = None,
    ) -> None:
        self.global_model = global_model
        self.teacher = teacher
        self.config = config or DistillationConfig()
        self.optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)

    # ------------------------------------------------------------------ #
    #  Build a "logit model" view of the global student                   #
    # ------------------------------------------------------------------ #

    def _build_student_logit_model(self) -> tf.keras.Model:
        """
        Return an independent version of the global model that outputs
        raw logits.  Must NOT share the teacher's cached model object,
        otherwise teacher set_weights() calls corrupt the student.
        """
        student = TeacherEnsemble._rebuild_with_linear_output(self.global_model)
        student.set_weights(self.global_model.get_weights())
        return student

    # ------------------------------------------------------------------ #
    #  Single training step                                               #
    # ------------------------------------------------------------------ #

    @tf.function
    def _train_step_kd_only(
        self,
        x_batch: tf.Tensor,
        teacher_logits: tf.Tensor,
        student_model: tf.keras.Model,
        temperature: float,
    ) -> tf.Tensor:
        """Pure distillation step (no supervised term)."""
        with tf.GradientTape() as tape:
            student_logits = student_model(x_batch, training=True)
            loss = distillation_loss(teacher_logits, student_logits, temperature)
        grads = tape.gradient(loss, student_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, student_model.trainable_variables)
        )
        return loss

    @tf.function
    def _train_step_combined(
        self,
        x_proxy: tf.Tensor,
        teacher_logits: tf.Tensor,
        x_sup: tf.Tensor,
        y_sup: tf.Tensor,
        student_model: tf.keras.Model,
        temperature: float,
        lam: float,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Combined distillation + supervised step."""
        with tf.GradientTape() as tape:
            stud_logits_proxy = student_model(x_proxy, training=True)
            l_kd = distillation_loss(teacher_logits, stud_logits_proxy, temperature)

            stud_logits_sup = student_model(x_sup, training=True)
            l_sup = supervised_loss(stud_logits_sup, y_sup)

            l_total = lam * l_kd + (1.0 - lam) * l_sup

        grads = tape.gradient(l_total, student_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, student_model.trainable_variables)
        )
        return l_total, l_kd, l_sup
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  Full distillation loop  (streams teacher logits per-batch)          #
    # ------------------------------------------------------------------ #

    def distill(
        self,
        proxy_data: tf.data.Dataset,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full distillation loop.

        Teacher logits are computed per-batch during training instead
        of pre-computing all into numpy arrays.
        """
        cfg = self.config
        T = cfg.temperature
        lam = cfg.lam

        history: Dict[str, List[float]] = {
            "epoch": [],
            "loss_total": [],
            "loss_kd": [],
            "loss_sup": [],
        }

        # Build proxy dataset (batched)
        proxy_ds = proxy_data.batch(cfg.batch_size).prefetch(1)

        # Prepare supervised data iterator (if available)
        if supervised_data is not None:
            sup_ds = (
                supervised_data
                .shuffle(buffer_size=1000)
                .batch(cfg.batch_size)
                .repeat()
                .prefetch(1)
            )
            sup_iter = iter(sup_ds)
        else:
            sup_iter = None

        # Build student logit model (reuse teacher's cached clone)
        student = self._build_student_logit_model()

        # --- Distillation epochs -------------------------------------- #
        for epoch in range(1, cfg.epochs + 1):
            epoch_loss_total = []
            epoch_loss_kd = []
            epoch_loss_sup = []

            for batch in proxy_ds:
                if isinstance(batch, (list, tuple)):
                    x_proxy_batch = batch[0]
                else:
                    x_proxy_batch = batch

                # Compute teacher logits on-the-fly (no precompute)
                teach_logits_batch = self.teacher.compute_teacher_logits_batch(
                    x_proxy_batch, temperature=T,
                )

                if sup_iter is not None:
                    try:
                        x_sup_batch, y_sup_batch = next(sup_iter)
                    except StopIteration:
                        sup_iter = iter(sup_ds)
                        x_sup_batch, y_sup_batch = next(sup_iter)

                    l_total, l_kd, l_sup = self._train_step_combined(
                        x_proxy_batch, teach_logits_batch,
                        x_sup_batch, y_sup_batch,
                        student, T, lam,
                    )
                    epoch_loss_total.append(float(l_total))
                    epoch_loss_kd.append(float(l_kd))
                    epoch_loss_sup.append(float(l_sup))
                else:
                    l_kd = self._train_step_kd_only(
                        x_proxy_batch, teach_logits_batch,
                        student, T,
                    )
                    epoch_loss_total.append(float(l_kd))
                    epoch_loss_kd.append(float(l_kd))
                    epoch_loss_sup.append(0.0)
            mean_total = float(np.mean(epoch_loss_total))
            mean_total = float(np.mean(epoch_loss_total))
            mean_kd    = float(np.mean(epoch_loss_kd))
            mean_sup   = float(np.mean(epoch_loss_sup))

            history["epoch"].append(epoch)
            history["loss_total"].append(mean_total)
            history["loss_kd"].append(mean_kd)
            history["loss_sup"].append(mean_sup)

            logger.debug(
                "Distillation epoch %d/%d — L_total=%.5f  L_KD=%.5f  L_sup=%.5f",
                epoch, cfg.epochs, mean_total, mean_kd, mean_sup,
            )

        # --- Copy distilled weights back to global model -------------- #
        self.global_model.set_weights(student.get_weights())
        # Cleanup
        # Cleanup
        del student
        gc.collect()

        return history

# ====================================================================== #
# ====================================================================== #
#  5.  CONVENIENCE: one-call distillation after a federated round         #
# ====================================================================== #

def run_distillation_round(
    global_model: tf.keras.Model,
    client_weights: Dict[str, List[np.ndarray]],
    contribution_weights: Dict[str, float],
    proxy_data: tf.data.Dataset,
    supervised_data: Optional[tf.data.Dataset] = None,
    config: Optional[DistillationConfig] = None,
) -> Dict[str, List[float]]:
    """
    One-liner helper that creates the teacher ensemble, distiller,
    and runs the distillation loop.
    """
    config = config or DistillationConfig()

    teacher = TeacherEnsemble(
        global_model=global_model,
        client_weights=client_weights,
        contribution_weights=contribution_weights,
    )
    distiller = KnowledgeDistiller(
        global_model=global_model,
        teacher=teacher,
        config=config,
    )
    result = distiller.distill(proxy_data, supervised_data)

    # Free teacher ensemble
    del teacher, distiller
    gc.collect()

    return result
