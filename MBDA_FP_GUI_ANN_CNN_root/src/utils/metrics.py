import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_global_metrics(y_true, y_pred):
    """Computes global Accuracy, Precision, Recall, and F1-Score."""
    acc = accuracy_score(y_true, y_pred) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec * 100, rec * 100, f1 * 100

def compute_per_class_metrics(y_true, y_pred, num_classes=26):
    """Computes accuracy and F1-Score for each individual class."""
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    per_class_acc = []
    for i in range(num_classes):
        mask = (y_true == i)
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean() * 100
        else:
            acc = 0.0
        per_class_acc.append(acc)
        
    return per_class_acc, f1_per_class * 100

def generate_confusion_matrix(y_true, y_pred):
    """Returns both raw and row-normalized confusion matrices."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize per row (true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # Handle NaNs if a class is completely missing
    cm_norm = np.nan_to_num(cm_norm)
    return cm, cm_norm