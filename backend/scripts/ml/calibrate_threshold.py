"""
Plot PR curves and find optimal decision threshold for precision target.

This script loads validation predictions from a trained model checkpoint,
computes precision-recall curves, and identifies the threshold that achieves
a target precision level (default: 0.75).

Usage:
    python backend/scripts/ml/calibrate_threshold.py \
        --predictions-csv path/to/val_predictions.csv \
        --target-precision 0.75 \
        --output-dir docs/modeling/calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate decision threshold from validation predictions")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="CSV with columns: y_true, y_pred_proba (validation set predictions)",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.75,
        help="Minimum acceptable precision for positive predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/modeling/calibration"),
        help="Directory to save PR curve plots and calibration report",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Optional run name to include in output filenames",
    )
    return parser.parse_args()


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load y_true and y_pred_proba from CSV."""
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "y_pred_proba" not in df.columns:
        raise ValueError(f"CSV must contain 'y_true' and 'y_pred_proba' columns. Found: {df.columns.tolist()}")
    y_true = df["y_true"].values
    y_pred_proba = df["y_pred_proba"].values
    return y_true, y_pred_proba


def find_threshold_for_precision(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    target_precision: float,
) -> tuple[float, float, float] | None:
    """
    Find the threshold that achieves at least target_precision.
    Returns (threshold, precision, recall) or None if target not achievable.
    """
    # precision_recall_curve returns thresholds for precision[:-1], so align arrays
    valid_mask = precision[:-1] >= target_precision
    if not np.any(valid_mask):
        return None
    
    # Find highest recall among valid thresholds
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(recall[:-1][valid_mask])]
    
    return thresholds[best_idx], precision[best_idx], recall[best_idx]


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    avg_precision: float,
    optimal_threshold: tuple[float, float, float] | None,
    output_path: Path,
) -> None:
    """Plot precision-recall curve with optimal threshold marker."""
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR curve (AP={avg_precision:.3f})")
    
    if optimal_threshold:
        thresh, prec, rec = optimal_threshold
        plt.scatter([rec], [prec], color="red", s=100, zorder=5, label=f"Threshold={thresh:.3f}")
        plt.axhline(y=prec, color="red", linestyle="--", alpha=0.5)
        plt.axvline(x=rec, color="red", linestyle="--", alpha=0.5)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve with Calibrated Threshold", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved PR curve to {output_path}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Plot ROC curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def save_calibration_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save calibration report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved calibration report to {output_path}")


def main() -> None:
    args = parse_args()
    
    # Load predictions
    y_true, y_pred_proba = load_predictions(args.predictions_csv)
    
    # Compute metrics
    avg_precision = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find optimal threshold
    optimal = find_threshold_for_precision(precision, recall, thresholds, args.target_precision)
    
    # Prepare output paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f"_{args.run_name}" if args.run_name else ""
    pr_plot_path = args.output_dir / f"pr_curve{run_suffix}.png"
    roc_plot_path = args.output_dir / f"roc_curve{run_suffix}.png"
    report_path = args.output_dir / f"calibration_report{run_suffix}.json"
    
    # Plot curves
    plot_pr_curve(precision, recall, avg_precision, optimal, pr_plot_path)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plot_roc_curve(fpr, tpr, roc_auc, roc_plot_path)
    
    # Build report
    report: dict[str, Any] = {
        "predictions_csv": str(args.predictions_csv),
        "target_precision": args.target_precision,
        "avg_precision_score": float(avg_precision),
        "roc_auc_score": float(roc_auc),
        "num_samples": len(y_true),
        "num_positives": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
    
    if optimal:
        threshold, prec, rec = optimal
        report["calibrated_threshold"] = float(threshold)
        report["achieved_precision"] = float(prec)
        report["achieved_recall"] = float(rec)
        report["f1_score"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"\nCalibration successful:")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Precision: {prec:.4f} (target: {args.target_precision:.4f})")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1: {report['f1_score']:.4f}")
    else:
        report["calibrated_threshold"] = None
        report["achieved_precision"] = None
        report["achieved_recall"] = None
        report["warning"] = f"Could not achieve target precision of {args.target_precision}"
        print(f"\nWarning: Could not find threshold achieving precision >= {args.target_precision}")
        print(f"Maximum achievable precision: {np.max(precision[:-1]):.4f}")
    
    # Save report
    save_calibration_report(report, report_path)
    
    print(f"\nOverall metrics:")
    print(f"  PR-AUC: {avg_precision:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
