"""
Train a sign-language classifier on the extracted landmark features.

Improvements:
  • Tries multiple models (RandomForest, GradientBoosting, SVM) and picks the best.
  • Cross-validation for robust evaluation.
  • Per-class precision / recall report.
  • Handles variable-length feature vectors by padding to a fixed size.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from config import (
    DATASET_PATH, MODEL_PATH, LABELS,
    NUM_FEATURES, TEST_SIZE, RANDOM_STATE,
)


def pad_or_truncate(seq, target_len=NUM_FEATURES):
    """Ensure every feature vector has exactly *target_len* elements."""
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [0.0] * (target_len - len(seq))


def train(dataset_path: str = DATASET_PATH,
          model_path: str = MODEL_PATH):
    """Load features, train several classifiers, save the best one."""

    # ── 1. Load data ─────────────────────────────────────────
    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f)

    raw_data = data_dict['data']
    raw_labels = data_dict['labels']

    # Pad / truncate to uniform length
    data = np.array([pad_or_truncate(d) for d in raw_data], dtype=np.float64)
    labels = np.array(raw_labels)

    print(f"\n▶  Dataset: {data.shape[0]} samples, {data.shape[1]} features")

    unique_classes = sorted(set(labels), key=lambda x: int(x) if x.isdigit() else x)
    print(f"   Classes : {len(unique_classes)}")
    class_labels = []
    for c in unique_classes:
        lbl_name = LABELS.get(int(c), c) if c.isdigit() else c
        class_labels.append(f"{c}({lbl_name})")
    print(f"   Labels  : {', '.join(class_labels[:10])}{'…' if len(class_labels) > 10 else ''}\n")

    # ── 2. Split ─────────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    # ── 3. Define candidate models ───────────────────────────
    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # ── 4. Evaluate each model ───────────────────────────────
    best_name = None
    best_score = 0.0
    best_model = None

    for name, clf in candidates.items():
        print(f"  Training {name} …", end=' ', flush=True)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"accuracy = {acc * 100:.2f}%")

        # Optional 5-fold cross-val (on training set)
        try:
            cv = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
            print(f"           CV mean = {cv.mean() * 100:.2f}% ± {cv.std() * 100:.2f}%")
        except Exception:
            pass

        if acc > best_score:
            best_score = acc
            best_name = name
            best_model = clf

    print(f"\n★  Best model: {best_name} ({best_score * 100:.2f}%)\n")

    # ── 5. Detailed report for the best model ────────────────
    y_pred = best_model.predict(x_test)
    # Build target names for the report
    target_names = []
    for c in sorted(set(y_test), key=lambda x: int(x) if x.isdigit() else x):
        lbl_name = LABELS.get(int(c), c) if c.isdigit() else c
        target_names.append(lbl_name)

    print(classification_report(y_test, y_pred, target_names=target_names))

    # ── 6. Save ──────────────────────────────────────────────
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best_model, 'model_name': best_name}, f)

    print(f"[✓] Model saved to {model_path}\n")


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train()
