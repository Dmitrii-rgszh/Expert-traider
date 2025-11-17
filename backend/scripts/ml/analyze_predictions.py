"""Quick script to analyze predictions and understand metrics."""
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_auc_score
import numpy as np

# Load predictions
val_preds = pd.read_csv('docs/modeling/predictions/tech_v2_predictions_val.csv')
test_preds = pd.read_csv('docs/modeling/predictions/tech_v2_predictions_test.csv')

print("=== VALIDATION SET ===")
print(f"Total samples: {len(val_preds)}")
print(f"Positive samples: {val_preds['y_true'].sum()} ({100*val_preds['y_true'].mean():.2f}%)")
print(f"PR-AUC (average_precision_score): {average_precision_score(val_preds['y_true'], val_preds['y_pred_proba']):.4f}")
print(f"ROC-AUC: {roc_auc_score(val_preds['y_true'], val_preds['y_pred_proba']):.4f}")

# Find precision at different recall levels
precisions, recalls, thresholds = precision_recall_curve(val_preds['y_true'], val_preds['y_pred_proba'])
print(f"\nMax achievable precision: {precisions.max():.4f} at recall {recalls[np.argmax(precisions)]:.4f}")

# Try to find thresholds for different precision targets
for target_p in [0.3, 0.5, 0.7, 0.75]:
    valid_idx = precisions >= target_p
    if valid_idx.any():
        best_recall = recalls[valid_idx].max()
        best_precision = precisions[valid_idx][recalls[valid_idx] == best_recall][0]
        thresh_idx = np.where((precisions >= target_p) & (recalls == best_recall))[0][0]
        if thresh_idx < len(thresholds):
            print(f"Precision >= {target_p}: achievable at recall={best_recall:.4f}, threshold={thresholds[thresh_idx]:.4f}")
        else:
            print(f"Precision >= {target_p}: achievable at recall={best_recall:.4f}, threshold=0.0 (classify all as positive)")
    else:
        print(f"Precision >= {target_p}: NOT achievable")

print("\n=== TEST SET ===")
print(f"Total samples: {len(test_preds)}")
print(f"Positive samples: {test_preds['y_true'].sum()} ({100*test_preds['y_true'].mean():.2f}%)")
print(f"PR-AUC (average_precision_score): {average_precision_score(test_preds['y_true'], test_preds['y_pred_proba']):.4f}")
print(f"ROC-AUC: {roc_auc_score(test_preds['y_true'], test_preds['y_pred_proba']):.4f}")

# Test set precision analysis
precisions_test, recalls_test, thresholds_test = precision_recall_curve(test_preds['y_true'], test_preds['y_pred_proba'])
print(f"\nMax achievable precision: {precisions_test.max():.4f} at recall {recalls_test[np.argmax(precisions_test)]:.4f}")

print("\n=== CLASS IMBALANCE IMPACT ===")
print(f"Val/Test positive rate ratio: {val_preds['y_true'].mean() / test_preds['y_true'].mean():.2f}x")
print(f"This means test set has {val_preds['y_true'].mean() / test_preds['y_true'].mean():.1f}x fewer trading opportunities")
print("\nConclusion: The model learned on val with 1.82% positives but test has only 0.51% positives.")
print("This temporal distribution shift makes calibration on val invalid for test set.")
