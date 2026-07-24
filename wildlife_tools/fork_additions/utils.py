import os
import torch
import numpy as np
import pandas as pd
import plotext as plt

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def print_info(msg):
    print(f"INFO: {msg}")


def print_warning(msg):
    print(f"WARNING: {msg}")


def cuda_available():
    return torch.cuda.is_available() and torch.version.cuda is not None


def warn_confused_pairs(confusion_matrix: np.ndarray, class_names=None, threshold=0.25):
    """
    Analyze a multi-class confusion matrix and warn about pairs with high mutual confusion.

    Args:
        confusion_matrix: numpy array of shape (n_classes, n_classes)
                         where element [i, j] represents true class i predicted as class j
        class_names: list of class names (optional). If None, uses class indices.
        threshold: confusion rate threshold above which to warn.

    Returns:
        list: List of tuples (class_i, class_j)
    """
    n_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if len(class_names) != n_classes:
        raise ValueError(f"Number of class names ({len(class_names)}) must match matrix size ({n_classes})")

    warned_pairs = []
    msg = "Some identifiants are often confused by the model.\n"

    for i in range(n_classes):
        total_i = confusion_matrix[i, :].sum()
        for j in range(i + 1, n_classes):

            i_predicted_as_j = confusion_matrix[i, j]
            j_predicted_as_i = confusion_matrix[j, i]

            total_j = confusion_matrix[j, :].sum()

            if total_i > 0 and total_j > 0:
                total_confusion = i_predicted_as_j + j_predicted_as_i
                confusion_rate = total_confusion / (total_i + total_j)

                i_to_j_rate = i_predicted_as_j / total_i
                j_to_i_rate = j_predicted_as_i / total_j

                if confusion_rate > threshold:
                    msg += f"\t-'{class_names[i]}' is confused as '{class_names[j]}' {i_to_j_rate:.1%} of the time, while '{class_names[j]}' is confused as '{class_names[i]}' {j_to_i_rate:.1%} of the time.\n"
                    warned_pairs.append((i, j))

    if warned_pairs:
        msg += """\n Your options are the following:
            \t1) If you can visually differienciate between those subjects yourself, you need to add nore data to your dataset.
            \t2) If you can not visually differienciate between those subjects yourself, you need to change your re-identification strategy."""
        print_warning(msg)

    return warned_pairs


def print_counts(counts: list, labels: list, phase: str = ""):
    plt.bar([str(i) for i in labels], [int(c) for c in counts])
    plt.xlabel("Identity")
    plt.ylabel("Count")
    plt.title(f"Identity Distribution ({phase})")
    plt.show()
    plt.clear_figure()


def enlarge_and_clip_bbox(bbox, factor, frame_shape):
    """
    Enlarge a bbox around its center by a fractional factor and clip to frame bounds.

    Args:
        bbox: tuple/list (x, y, w, h) in pixel coords.
        factor: float - enlargement fraction (e.g., 0.1 means 10% larger on each side).
        frame_shape: frame shape with (H, W) as the first two dims.

    Returns:
        tuple (x, y, w, h) of ints, clipped to [0, W] x [0, H].
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[0], frame_shape[1]

    half_dw = w * factor / 2.0
    half_dh = h * factor / 2.0

    x1 = max(0, int(x - half_dw))
    y1 = max(0, int(y - half_dh))
    x2 = min(frame_w, int(x + w + half_dw))
    y2 = min(frame_h, int(y + h + half_dh))

    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def calculate_overlaps(b1, b2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes in xywh format.

    Args:
        b1: tuple/list (x, y, w, h) - first bounding box
        b2: tuple/list (x, y, w, h) - second bounding box

    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    b1_x2, b1_y2 = x1 + w1, y1 + h1
    b2_x2, b2_y2 = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = w1 * h1
    b2_area = w2 * h2
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def bboxes_have_score_column(root: str, *phases: str) -> bool:
    """Check whether any bbox CSV under the given phases has a 'score' column, without loading full data."""
    for phase in phases:
        bboxes_dir = os.path.join(root, "bboxes", phase)
        if not os.path.isdir(bboxes_dir):
            continue
        for file in os.listdir(bboxes_dir):
            if file.endswith(".csv"):
                header = pd.read_csv(os.path.join(bboxes_dir, file), nrows=0)
                if "score" in header.columns:
                    return True
    return False
