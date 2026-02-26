"""
Evaluation Metrics & Report Generation
=======================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Evaluates the global model after the federated learning cycle on:

* **Accuracy**
* **F1 Score** (macro, weighted, and per-class)
* **ROC-AUC**
* **Inference Latency** (mean, std, p95 over repeated batches)
* **Model Size** (parameter count + on-disk file size)

Generates structured reports (JSON + human-readable text) into a
dedicated ``reports/`` folder to keep the workspace organised.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default report output directory (sibling of this file)
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = _MODULE_DIR / "reports"


# ====================================================================== #
#  1.  DATA STRUCTURES                                                    #
# ====================================================================== #

@dataclass
class ClassificationMetrics:
    """Container for all classification-related evaluation metrics."""
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    f1_per_class: Dict[str, float] = field(default_factory=dict)
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: Optional[List[List[int]]] = None
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Container for inference-latency measurements."""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    num_batches: int = 0
    batch_size: int = 0
    total_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSizeMetrics:
    """Container for model-size information."""
    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    file_size_bytes: int = 0
    file_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Full evaluation report aggregating all metric categories."""
    model_name: str = ""
    timestamp: str = ""
    federated_round: Optional[int] = None
    classification: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    model_size: ModelSizeMetrics = field(default_factory=ModelSizeMetrics)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "federated_round": self.federated_round,
            "classification": self.classification.to_dict(),
            "latency": self.latency.to_dict(),
            "model_size": self.model_size.to_dict(),
            "extra": self.extra,
        }


# ====================================================================== #
#  2.  METRIC COMPUTATIONS                                                #
# ====================================================================== #

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """
    Compute accuracy, F1 (macro/weighted/per-class), precision, recall,
    ROC-AUC, and confusion matrix from ground-truth labels and predicted
    probabilities.

    Parameters
    ----------
    y_true : np.ndarray, shape ``(N,)``
        Ground-truth binary labels (0 or 1).
    y_pred_probs : np.ndarray, shape ``(N,)`` or ``(N, 1)``
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold for converting probabilities to hard labels.
    class_names : list[str] | None
        Human-readable names for class 0 and class 1.
        Defaults to ``["Real", "Fake"]``.

    Returns
    -------
    ClassificationMetrics
    """
    if class_names is None:
        class_names = ["Real", "Fake"]

    y_true = np.asarray(y_true).ravel().astype(int)
    y_probs = np.asarray(y_pred_probs).ravel().astype(float)
    y_pred = (y_probs >= threshold).astype(int)
    n = len(y_true)

    # --- Confusion matrix --------------------------------------------- #
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    cm = [[tn, fp], [fn, tp]]

    # --- Accuracy ----------------------------------------------------- #
    accuracy = (tp + tn) / max(n, 1)

    # --- Per-class precision / recall / F1 ---------------------------- #
    def _prf(tp_c: int, fp_c: int, fn_c: int):
        prec = tp_c / max(tp_c + fp_c, 1)
        rec = tp_c / max(tp_c + fn_c, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return prec, rec, f1

    prec_0, rec_0, f1_0 = _prf(tn, fn, fp)   # class 0 = Real
    prec_1, rec_1, f1_1 = _prf(tp, fp, fn)   # class 1 = Fake

    support_0 = int(np.sum(y_true == 0))
    support_1 = int(np.sum(y_true == 1))

    f1_macro = (f1_0 + f1_1) / 2.0
    f1_weighted = (f1_0 * support_0 + f1_1 * support_1) / max(n, 1)
    precision_macro = (prec_0 + prec_1) / 2.0
    recall_macro = (rec_0 + rec_1) / 2.0

    f1_per_class = {
        class_names[0]: round(f1_0, 6),
        class_names[1]: round(f1_1, 6),
    }

    # --- ROC-AUC ------------------------------------------------------ #
    roc_auc = _compute_roc_auc(y_true, y_probs)

    return ClassificationMetrics(
        accuracy=round(accuracy, 6),
        f1_macro=round(f1_macro, 6),
        f1_weighted=round(f1_weighted, 6),
        f1_per_class=f1_per_class,
        precision_macro=round(precision_macro, 6),
        recall_macro=round(recall_macro, 6),
        roc_auc=round(roc_auc, 6),
        confusion_matrix=cm,
        num_samples=n,
    )


def _compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute ROC-AUC using the trapezoidal rule (no sklearn dependency).

    Returns 0.0 when only one class is present.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    # Sort by descending score
    desc = np.argsort(-y_scores)
    y_sorted = y_true[desc]
    scores_sorted = y_scores[desc]

    num_pos = np.sum(y_true == 1)
    num_neg = np.sum(y_true == 0)

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    # Walk through thresholds (distinct score values)
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        # Only compute a point when the score changes or at the end
        if i == len(y_sorted) - 1 or scores_sorted[i] != scores_sorted[i + 1]:
            tpr = tp / max(num_pos, 1)
            fpr = fp / max(num_neg, 1)
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_fpr = fpr
            prev_tpr = tpr

    return float(auc)


# ====================================================================== #
#  3.  INFERENCE LATENCY                                                  #
# ====================================================================== #

def measure_inference_latency(
    model: tf.keras.Model,
    test_data: tf.data.Dataset,
    batch_size: int = 32,
    warmup_batches: int = 3,
    max_batches: Optional[int] = None,
) -> LatencyMetrics:
    """
    Measure per-batch inference latency.

    Parameters
    ----------
    model : tf.keras.Model
    test_data : tf.data.Dataset
        Yields ``(images,)`` or ``(images, labels)``.
    batch_size : int
    warmup_batches : int
        Number of initial batches to discard (JIT warm-up).
    max_batches : int | None
        Cap the number of measured batches (``None`` = all).

    Returns
    -------
    LatencyMetrics
    """
    batched = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    times_ms: List[float] = []
    total_samples = 0
    batch_count = 0

    for batch in batched:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_count += 1

        if batch_count <= warmup_batches:
            model(x, training=False)   # warm-up, discard timing
            continue

        t0 = time.perf_counter()
        model(x, training=False)
        t1 = time.perf_counter()

        times_ms.append((t1 - t0) * 1000.0)
        total_samples += int(x.shape[0])

        if max_batches is not None and len(times_ms) >= max_batches:
            break

    if not times_ms:
        logger.warning("No batches measured — dataset may be too small for warm-up.")
        return LatencyMetrics(batch_size=batch_size)

    arr = np.array(times_ms)
    return LatencyMetrics(
        mean_ms=round(float(arr.mean()), 4),
        std_ms=round(float(arr.std()), 4),
        min_ms=round(float(arr.min()), 4),
        max_ms=round(float(arr.max()), 4),
        p50_ms=round(float(np.percentile(arr, 50)), 4),
        p95_ms=round(float(np.percentile(arr, 95)), 4),
        p99_ms=round(float(np.percentile(arr, 99)), 4),
        num_batches=len(times_ms),
        batch_size=batch_size,
        total_samples=total_samples,
    )


# ====================================================================== #
#  4.  MODEL SIZE                                                         #
# ====================================================================== #

def measure_model_size(
    model: tf.keras.Model,
    save_path: Optional[str] = None,
) -> ModelSizeMetrics:
    """
    Count parameters and measure on-disk file size of the model.

    Parameters
    ----------
    model : tf.keras.Model
    save_path : str | None
        If ``None``, a temporary file is used and deleted afterwards.

    Returns
    -------
    ModelSizeMetrics
    """
    total = int(model.count_params())
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = total - trainable

    # On-disk size
    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()
        save_path = tmp.name
        cleanup = True
    else:
        cleanup = False

    try:
        model.save(save_path)
        file_bytes = os.path.getsize(save_path)
    finally:
        if cleanup and os.path.exists(save_path):
            os.unlink(save_path)

    return ModelSizeMetrics(
        total_params=total,
        trainable_params=trainable,
        non_trainable_params=non_trainable,
        file_size_bytes=file_bytes,
        file_size_mb=round(file_bytes / (1024 * 1024), 4),
    )


# ====================================================================== #
#  5.  FULL EVALUATOR                                                     #
# ====================================================================== #

class FederatedModelEvaluator:
    """
    One-stop evaluator that runs all metrics and produces reports.

    Parameters
    ----------
    model : tf.keras.Model
        The global model to evaluate.
    model_name : str
        Human-readable label (used in report filenames).
    reports_dir : str | Path
        Directory where reports are saved.
    class_names : list[str]
        Names for class 0 / class 1.  Defaults to ``["Real", "Fake"]``.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: str = "effnet_global",
        reports_dir: str | Path = DEFAULT_REPORTS_DIR,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.reports_dir = Path(reports_dir)
        self.class_names = class_names or ["Real", "Fake"]

        # Ensure the reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Run full evaluation                                                #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        test_data: tf.data.Dataset,
        batch_size: int = 32,
        threshold: float = 0.5,
        federated_round: Optional[int] = None,
        warmup_batches: int = 3,
        latency_max_batches: Optional[int] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """
        Run **all** evaluations and return a structured report.

        Parameters
        ----------
        test_data : tf.data.Dataset
            Yields ``(images, labels)``.
        batch_size : int
        threshold : float
            Classification decision threshold.
        federated_round : int | None
            If provided, included in the report metadata.
        warmup_batches : int
            Warm-up batches for latency measurement.
        latency_max_batches : int | None
            Cap measured batches for latency.
        extra_info : dict | None
            Arbitrary metadata to attach to the report.

        Returns
        -------
        EvaluationReport
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Starting evaluation for '%s' …", self.model_name)

        # --- 1. Predictions ------------------------------------------- #
        y_true_list: List[np.ndarray] = []
        y_prob_list: List[np.ndarray] = []

        batched = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        for batch in batched:
            x, y = batch[0], batch[1]
            preds = self.model(x, training=False)
            y_true_list.append(y.numpy())
            y_prob_list.append(preds.numpy())

        y_true = np.concatenate(y_true_list).ravel()
        y_probs = np.concatenate(y_prob_list).ravel()

        # --- 2. Classification metrics -------------------------------- #
        cls_metrics = compute_classification_metrics(
            y_true, y_probs,
            threshold=threshold,
            class_names=self.class_names,
        )
        logger.info(
            "Classification — Acc: %.4f | F1-macro: %.4f | ROC-AUC: %.4f",
            cls_metrics.accuracy, cls_metrics.f1_macro, cls_metrics.roc_auc,
        )

        # --- 3. Inference latency ------------------------------------- #
        lat_metrics = measure_inference_latency(
            self.model, test_data,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            max_batches=latency_max_batches,
        )
        logger.info(
            "Latency — mean: %.2f ms | p95: %.2f ms | p99: %.2f ms",
            lat_metrics.mean_ms, lat_metrics.p95_ms, lat_metrics.p99_ms,
        )

        # --- 4. Model size -------------------------------------------- #
        size_metrics = measure_model_size(self.model)
        logger.info(
            "Model size — params: %s | disk: %.2f MB",
            f"{size_metrics.total_params:,}", size_metrics.file_size_mb,
        )

        report = EvaluationReport(
            model_name=self.model_name,
            timestamp=ts,
            federated_round=federated_round,
            classification=cls_metrics,
            latency=lat_metrics,
            model_size=size_metrics,
            extra=extra_info or {},
        )

        return report

    # ------------------------------------------------------------------ #
    #  Report persistence                                                 #
    # ------------------------------------------------------------------ #

    def save_report(
        self,
        report: EvaluationReport,
        tag: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Save the report as both JSON and a human-readable text file.

        Parameters
        ----------
        report : EvaluationReport
        tag : str | None
            Optional suffix for the filename (e.g. ``"round_05"``).

        Returns
        -------
        (json_path, txt_path) : tuple[Path, Path]
        """
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.model_name}_{ts_slug}"
        if tag:
            base += f"_{tag}"

        json_path = self.reports_dir / f"{base}.json"
        txt_path = self.reports_dir / f"{base}.txt"

        # --- JSON ----------------------------------------------------- #
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        # --- Human-readable text -------------------------------------- #
        txt_path.write_text(
            self._format_text_report(report),
            encoding="utf-8",
        )

        logger.info("Reports saved → %s  &  %s", json_path.name, txt_path.name)
        return json_path, txt_path

    # ------------------------------------------------------------------ #
    #  Text report formatter                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_text_report(r: EvaluationReport) -> str:
        """Render a pretty-printed text report."""
        sep = "=" * 62
        cls = r.classification
        lat = r.latency
        sz = r.model_size

        lines = [
            sep,
            "  EVALUATION REPORT",
            sep,
            f"  Model:            {r.model_name}",
            f"  Timestamp:        {r.timestamp}",
        ]
        if r.federated_round is not None:
            lines.append(f"  Federated Round:  {r.federated_round}")
        lines.append("")

        # Classification
        lines += [
            "-" * 62,
            "  CLASSIFICATION METRICS",
            "-" * 62,
            f"  Samples evaluated:  {cls.num_samples:,}",
            f"  Accuracy:           {cls.accuracy:.4f}",
            f"  F1 (macro):         {cls.f1_macro:.4f}",
            f"  F1 (weighted):      {cls.f1_weighted:.4f}",
            f"  Precision (macro):  {cls.precision_macro:.4f}",
            f"  Recall (macro):     {cls.recall_macro:.4f}",
            f"  ROC-AUC:            {cls.roc_auc:.4f}",
            "",
            "  Per-class F1:",
        ]
        for name, f1 in cls.f1_per_class.items():
            lines.append(f"    {name:<16} {f1:.4f}")

        if cls.confusion_matrix is not None:
            cm = cls.confusion_matrix
            lines += [
                "",
                "  Confusion Matrix:     Pred=0   Pred=1",
                f"    Actual=0 (Real)    {cm[0][0]:>7,}  {cm[0][1]:>7,}",
                f"    Actual=1 (Fake)    {cm[1][0]:>7,}  {cm[1][1]:>7,}",
            ]

        # Latency
        lines += [
            "",
            "-" * 62,
            "  INFERENCE LATENCY",
            "-" * 62,
            f"  Batches measured:   {lat.num_batches}  (batch_size={lat.batch_size})",
            f"  Total samples:      {lat.total_samples:,}",
            f"  Mean:               {lat.mean_ms:.2f} ms",
            f"  Std:                {lat.std_ms:.2f} ms",
            f"  Min:                {lat.min_ms:.2f} ms",
            f"  Max:                {lat.max_ms:.2f} ms",
            f"  P50 (median):       {lat.p50_ms:.2f} ms",
            f"  P95:                {lat.p95_ms:.2f} ms",
            f"  P99:                {lat.p99_ms:.2f} ms",
        ]

        # Model size
        lines += [
            "",
            "-" * 62,
            "  MODEL SIZE",
            "-" * 62,
            f"  Total params:       {sz.total_params:,}",
            f"  Trainable params:   {sz.trainable_params:,}",
            f"  Non-trainable:      {sz.non_trainable_params:,}",
            f"  File size:          {sz.file_size_mb:.2f} MB  ({sz.file_size_bytes:,} bytes)",
        ]

        # Extra
        if r.extra:
            lines += [
                "",
                "-" * 62,
                "  ADDITIONAL INFO",
                "-" * 62,
            ]
            for k, v in r.extra.items():
                lines.append(f"  {k}: {v}")

        lines += ["", sep, "  END OF REPORT", sep, ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Comparative report across rounds                                   #
    # ------------------------------------------------------------------ #

    def save_comparison_report(
        self,
        reports: List[EvaluationReport],
        filename: str = "comparison",
    ) -> Tuple[Path, Path]:
        """
        Generate a comparison table across multiple evaluation reports
        (e.g. one per federated round).

        Returns ``(json_path, txt_path)``.
        """
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{filename}_{ts_slug}"
        json_path = self.reports_dir / f"{base}.json"
        txt_path = self.reports_dir / f"{base}.txt"

        # JSON
        json_data = [r.to_dict() for r in reports]
        json_path.write_text(
            json.dumps(json_data, indent=2, default=str),
            encoding="utf-8",
        )

        # Text table
        header = (
            f"{'Round':>6} | {'Acc':>8} | {'F1-mac':>8} | {'ROC-AUC':>8} | "
            f"{'Lat(ms)':>8} | {'P95(ms)':>8} | {'Size(MB)':>9}"
        )
        sep = "-" * len(header)
        lines = [
            "=" * len(header),
            "  COMPARISON REPORT",
            "=" * len(header),
            "",
            header,
            sep,
        ]
        for r in reports:
            rnd = r.federated_round if r.federated_round is not None else "—"
            lines.append(
                f"{rnd:>6} | {r.classification.accuracy:>8.4f} | "
                f"{r.classification.f1_macro:>8.4f} | "
                f"{r.classification.roc_auc:>8.4f} | "
                f"{r.latency.mean_ms:>8.2f} | "
                f"{r.latency.p95_ms:>8.2f} | "
                f"{r.model_size.file_size_mb:>9.2f}"
            )
        lines += [sep, ""]

        txt_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Comparison report saved → %s  &  %s", json_path.name, txt_path.name)
        return json_path, txt_path


# ====================================================================== #
#  6.  CONVENIENCE: evaluate + save in one call                           #
# ====================================================================== #

def evaluate_and_report(
    model: tf.keras.Model,
    test_data: tf.data.Dataset,
    model_name: str = "effnet_global",
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    batch_size: int = 32,
    threshold: float = 0.5,
    federated_round: Optional[int] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    tag: Optional[str] = None,
) -> EvaluationReport:
    """
    One-liner: evaluate the model and write JSON + text reports.

    Returns the ``EvaluationReport`` dataclass for programmatic use.
    """
    evaluator = FederatedModelEvaluator(
        model=model,
        model_name=model_name,
        reports_dir=reports_dir,
    )
    report = evaluator.evaluate(
        test_data,
        batch_size=batch_size,
        threshold=threshold,
        federated_round=federated_round,
        extra_info=extra_info,
    )
    evaluator.save_report(report, tag=tag)
    return report


# ====================================================================== #
#  DEMO / SMOKE-TEST                                                      #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  Evaluation Metrics & Report Generation — Demo  ===\n")

    np.random.seed(42)
    tf.random.set_seed(42)

    # ---- 1. Build a tiny model --------------------------------------- #
    INPUT_DIM = 16
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # ---- 2. Synthetic test data -------------------------------------- #
    NUM_SAMPLES = 500
    x_test = np.random.randn(NUM_SAMPLES, INPUT_DIM).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(NUM_SAMPLES,)).astype(np.float32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # ---- 3. Run evaluation ------------------------------------------- #
    evaluator = FederatedModelEvaluator(
        model=model,
        model_name="demo_effnet",
        reports_dir=DEFAULT_REPORTS_DIR,
    )

    report = evaluator.evaluate(
        test_data=test_ds,
        batch_size=64,
        federated_round=5,
        extra_info={"dataset": "synthetic", "notes": "smoke-test"},
    )

    # ---- 4. Print text report to console ----------------------------- #
    print(evaluator._format_text_report(report))

    # ---- 5. Save reports to disk ------------------------------------- #
    json_path, txt_path = evaluator.save_report(report, tag="round_05")
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")

    # ---- 6. Simulate multi-round comparison -------------------------- #
    reports: List[EvaluationReport] = []
    for rnd in range(1, 4):
        # Slightly perturb weights to simulate different rounds
        for w in model.trainable_weights:
            w.assign_add(tf.random.normal(w.shape, stddev=0.01))
        r = evaluator.evaluate(test_ds, batch_size=64, federated_round=rnd)
        reports.append(r)

    cmp_json, cmp_txt = evaluator.save_comparison_report(reports)
    print(f"\nComparison JSON: {cmp_json}")
    print(f"Comparison text: {cmp_txt}")

    print("\nDone.")
