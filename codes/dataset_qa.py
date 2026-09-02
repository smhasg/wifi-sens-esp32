#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_qa.py

Quality Assurance for paired WiFi CSI + Camera Pose dataset.

Expected NPZ keys:
    csi_amplitude      [N, 128]
    iq                 [N, 256]
    pose               [N, 17, 2]
    pose_confidence    [N, 17]
    timestamp          [N]
    rssi               [N]
    channel            [N]
    pose_age_ms        [N]
    csi_arrival_time   [N]
    camera_time        [N]

The script:
    1. Finds and loads chunk_*.npz files
    2. Validates shapes / NaN / Inf
    3. Analyzes CSI amplitude and subcarriers
    4. Finds dead / mostly-zero subcarriers
    5. Finds exact and near-duplicate CSI samples
    6. Analyzes pose confidence / visibility / diversity
    7. Measures body center, bounding box and temporal movement
    8. Analyzes synchronization
    9. Estimates sample rate
   10. Measures CSI-motion <-> Pose-motion relationship
   11. Searches temporal lag between CSI and camera
   12. Saves CSV reports + PNG plots + TXT summary

Author: Dataset QA pipeline
"""

from pathlib import Path
import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

# You can point this either to:
#   wifi_pose_dataset
#
# or directly to one session:
#   wifi_pose_dataset/session_2026...
#
DATASET_DIR = Path("wifi_pose_dataset")

FILE_PATTERN = "chunk_*.npz"

OUTPUT_DIR = Path("dataset_qa_output")

EXPECTED_SUBCARRIERS = 128
EXPECTED_KEYPOINTS = 17

# Keypoint is treated as visible when confidence >= this
POSE_CONF_THRESHOLD = 0.20

# Strong-confidence metric
POSE_GOOD_CONF_THRESHOLD = 0.50

# A subcarrier is "mostly zero" if more than this fraction is exactly zero
MOSTLY_ZERO_THRESHOLD = 0.95

# A subcarrier is considered very low variance below this threshold
LOW_VARIANCE_EPS = 1e-6

# Near duplicate threshold for NORMALIZED consecutive CSI distance.
# This is descriptive only; do not automatically delete samples based on this.
NEAR_DUP_THRESHOLD = 0.01

# Lag analysis
MAX_LAG_SAMPLES = 30

# Plot size limits
MAX_HEATMAP_SAMPLES = 1500

# Rolling average for temporal plots
ROLLING_WINDOW = 15


# ============================================================
# SKELETON
# COCO 17-keypoint topology
# ============================================================

SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),

    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),

    (5, 11),
    (6, 12),
    (11, 12),

    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# ============================================================
# UTILITIES
# ============================================================

def separator(title=None):
    print()
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)


def safe_float_array(x):
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception:
        return np.asarray(x)


def rolling_mean(x, window=15):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return x

    if window <= 1:
        return x.copy()

    s = pd.Series(x)
    return s.rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean().to_numpy()


def safe_corr(a, b):
    """
    Pearson correlation without scipy.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def percentile_text(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return "No valid data"

    return (
        f"min={np.min(x):.4f}, "
        f"p25={np.percentile(x, 25):.4f}, "
        f"median={np.median(x):.4f}, "
        f"p75={np.percentile(x, 75):.4f}, "
        f"p95={np.percentile(x, 95):.4f}, "
        f"max={np.max(x):.4f}"
    )


def normalize_rows(x):
    """
    Normalize each sample independently for shape-based comparison.
    """
    x = np.asarray(x, dtype=float)

    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)

    std[std < 1e-8] = 1.0

    return (x - mean) / std


def save_plot(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


# ============================================================
# LOAD DATASET
# ============================================================

def find_chunk_files():
    files = sorted(DATASET_DIR.rglob(FILE_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{FILE_PATTERN}' found under:\n"
            f"{DATASET_DIR.resolve()}"
        )

    return files


def inspect_chunks(files):
    records = []

    separator("CHUNK FILES")

    total = 0

    for f in files:
        try:
            with np.load(f, allow_pickle=False) as d:
                keys = list(d.keys())

                if "csi_amplitude" in d:
                    n = len(d["csi_amplitude"])
                else:
                    n = -1

                total += max(n, 0)

                record = {
                    "file": str(f),
                    "samples": n,
                    "keys": ",".join(keys),
                }

                records.append(record)

                print(
                    f"{f.name:<30} "
                    f"samples={n:<6} "
                    f"keys={len(keys)}"
                )

        except Exception as e:
            print(f"[ERROR] {f}: {e}")

            records.append({
                "file": str(f),
                "samples": -1,
                "keys": "",
                "error": str(e),
            })

    print()
    print(f"Files        : {len(files)}")
    print(f"Total samples: {total}")

    df = pd.DataFrame(records)
    df.to_csv(
        OUTPUT_DIR / "chunk_report.csv",
        index=False
    )


def load_dataset(files):
    wanted_keys = [
        "csi_amplitude",
        "iq",
        "pose",
        "pose_confidence",
        "timestamp",
        "rssi",
        "channel",
        "pose_age_ms",
        "csi_arrival_time",
        "camera_time",
    ]

    storage = {
        key: []
        for key in wanted_keys
    }

    source_file = []
    source_index = []

    separator("LOADING DATASET")

    for file_index, f in enumerate(files):

        print(f"[LOAD] {f}")

        with np.load(f, allow_pickle=False) as d:

            if "csi_amplitude" not in d:
                print(
                    f"[WARN] Skipping {f.name}: "
                    "missing csi_amplitude"
                )
                continue

            n = len(d["csi_amplitude"])

            for key in wanted_keys:
                if key in d:
                    storage[key].append(np.asarray(d[key]))

            source_file.extend([f.name] * n)
            source_index.extend(range(n))

    data = {}

    for key, arrays in storage.items():

        if not arrays:
            data[key] = None
            continue

        try:
            data[key] = np.concatenate(arrays, axis=0)
        except Exception as e:
            print(
                f"[WARN] Could not concatenate key '{key}': {e}"
            )
            data[key] = None

    data["source_file"] = np.asarray(source_file)
    data["source_index"] = np.asarray(source_index)

    return data


# ============================================================
# GENERAL VALIDATION
# ============================================================

def validate_dataset(data):
    separator("GENERAL DATASET VALIDATION")

    expected = {
        "csi_amplitude": (None, EXPECTED_SUBCARRIERS),
        "iq": (None, EXPECTED_SUBCARRIERS * 2),
        "pose": (None, EXPECTED_KEYPOINTS, 2),
        "pose_confidence": (None, EXPECTED_KEYPOINTS),
    }

    report = []

    sample_counts = []

    for key, value in data.items():

        if key in ("source_file", "source_index"):
            continue

        if value is None:
            print(f"{key:<22}: MISSING")

            report.append({
                "key": key,
                "present": False,
            })

            continue

        arr = np.asarray(value)

        sample_counts.append(arr.shape[0])

        if np.issubdtype(arr.dtype, np.number):
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
        else:
            nan_count = 0
            inf_count = 0

        print(
            f"{key:<22}: "
            f"shape={str(arr.shape):<22} "
            f"dtype={str(arr.dtype):<10} "
            f"NaN={nan_count:<6} "
            f"Inf={inf_count}"
        )

        report.append({
            "key": key,
            "present": True,
            "shape": str(arr.shape),
            "dtype": str(arr.dtype),
            "nan_count": nan_count,
            "inf_count": inf_count,
        })

    if sample_counts:

        unique_counts = sorted(set(sample_counts))

        print()
        print("Sample count values:", unique_counts)

        if len(unique_counts) != 1:
            print(
                "[WARNING] Keys do NOT contain the same number "
                "of samples!"
            )
        else:
            print("[OK] Sample counts are consistent.")

    csi = data.get("csi_amplitude")

    if csi is not None:

        if (
            csi.ndim == 2
            and csi.shape[1] == EXPECTED_SUBCARRIERS
        ):
            print(
                f"[OK] CSI shape matches expected "
                f"[N, {EXPECTED_SUBCARRIERS}]"
            )
        else:
            print(
                f"[WARNING] Unexpected CSI shape: {csi.shape}"
            )

    pose = data.get("pose")

    if pose is not None:

        if (
            pose.ndim == 3
            and pose.shape[1:] == (EXPECTED_KEYPOINTS, 2)
        ):
            print(
                f"[OK] Pose shape matches expected "
                f"[N, {EXPECTED_KEYPOINTS}, 2]"
            )
        else:
            print(
                f"[WARNING] Unexpected pose shape: {pose.shape}"
            )

    pd.DataFrame(report).to_csv(
        OUTPUT_DIR / "general_validation.csv",
        index=False
    )


# ============================================================
# CSI ANALYSIS
# ============================================================

def analyze_csi(data):

    csi = data.get("csi_amplitude")

    if csi is None:
        return {}

    csi = np.asarray(csi, dtype=np.float64)

    separator("CSI QUALITY")

    print(f"Samples      : {csi.shape[0]}")
    print(f"Subcarriers  : {csi.shape[1]}")

    finite = csi[np.isfinite(csi)]

    print()
    print("Amplitude statistics")
    print("-" * 50)

    print(f"min    : {np.min(finite):.6f}")
    print(f"max    : {np.max(finite):.6f}")
    print(f"mean   : {np.mean(finite):.6f}")
    print(f"median : {np.median(finite):.6f}")
    print(f"std    : {np.std(finite):.6f}")

    # --------------------------------------------------------
    # Per-subcarrier statistics
    # --------------------------------------------------------

    sc_mean = np.nanmean(csi, axis=0)
    sc_std = np.nanstd(csi, axis=0)

    zero_rate = np.mean(
        np.isclose(csi, 0.0, atol=1e-12),
        axis=0
    )

    mostly_zero = zero_rate >= MOSTLY_ZERO_THRESHOLD

    low_variance = sc_std <= LOW_VARIANCE_EPS

    print()
    print("Subcarrier quality")
    print("-" * 50)

    print(
        f"Mostly-zero subcarriers "
        f"(>={MOSTLY_ZERO_THRESHOLD * 100:.0f}% zero): "
        f"{np.sum(mostly_zero)} / {csi.shape[1]}"
    )

    print(
        f"Near-zero variance subcarriers: "
        f"{np.sum(low_variance)} / {csi.shape[1]}"
    )

    if np.any(mostly_zero):
        print(
            "Mostly-zero indices:",
            np.where(mostly_zero)[0].tolist()
        )

    if np.any(low_variance):
        print(
            "Low-variance indices:",
            np.where(low_variance)[0].tolist()
        )

    subcarrier_df = pd.DataFrame({
        "subcarrier": np.arange(csi.shape[1]),
        "mean": sc_mean,
        "std": sc_std,
        "zero_rate": zero_rate,
        "mostly_zero": mostly_zero,
        "low_variance": low_variance,
    })

    subcarrier_df.to_csv(
        OUTPUT_DIR / "subcarrier_quality.csv",
        index=False
    )

    # --------------------------------------------------------
    # Sample-level statistics
    # --------------------------------------------------------

    sample_mean = np.nanmean(csi, axis=1)
    sample_std = np.nanstd(csi, axis=1)

    sample_energy = np.nanmean(
        csi ** 2,
        axis=1
    )

    sample_zero_fraction = np.mean(
        np.isclose(csi, 0.0, atol=1e-12),
        axis=1
    )

    sample_df = pd.DataFrame({
        "sample": np.arange(len(csi)),
        "csi_mean": sample_mean,
        "csi_std": sample_std,
        "csi_energy": sample_energy,
        "zero_fraction": sample_zero_fraction,
    })

    sample_df.to_csv(
        OUTPUT_DIR / "csi_sample_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Exact duplicate CSI rows
    # --------------------------------------------------------

    print()
    print("Duplicate analysis")
    print("-" * 50)

    contiguous_csi = np.ascontiguousarray(
        csi.astype(np.float32)
    )

    row_view = contiguous_csi.view(
        np.dtype(
            (
                np.void,
                contiguous_csi.dtype.itemsize
                * contiguous_csi.shape[1]
            )
        )
    ).reshape(-1)

    _, unique_idx, counts = np.unique(
        row_view,
        return_index=True,
        return_counts=True
    )

    n_unique = len(unique_idx)
    exact_duplicate_samples = len(csi) - n_unique

    duplicate_rate = (
        exact_duplicate_samples / len(csi)
        if len(csi)
        else 0
    )

    print(f"Unique CSI samples       : {n_unique}")
    print(f"Exact duplicate samples  : {exact_duplicate_samples}")
    print(f"Exact duplicate rate     : {duplicate_rate*100:.3f}%")

    # --------------------------------------------------------
    # Consecutive differences
    # --------------------------------------------------------

    if len(csi) > 1:

        normalized = normalize_rows(csi)

        normalized_diff = np.sqrt(
            np.mean(
                np.diff(normalized, axis=0) ** 2,
                axis=1
            )
        )

        raw_diff = np.sqrt(
            np.mean(
                np.diff(csi, axis=0) ** 2,
                axis=1
            )
        )

        near_duplicate = (
            normalized_diff < NEAR_DUP_THRESHOLD
        )

        print()
        print("Consecutive CSI difference")
        print("-" * 50)

        print(
            "Raw RMS difference       :",
            percentile_text(raw_diff)
        )

        print(
            "Normalized RMS difference:",
            percentile_text(normalized_diff)
        )

        print(
            f"Near-identical consecutive pairs "
            f"(< {NEAR_DUP_THRESHOLD}): "
            f"{np.sum(near_duplicate)} / "
            f"{len(normalized_diff)} "
            f"({np.mean(near_duplicate)*100:.2f}%)"
        )

    else:

        normalized_diff = np.array([])
        raw_diff = np.array([])

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plt.figure(figsize=(14, 5))
    plt.plot(sc_mean)
    plt.xlabel("Subcarrier")
    plt.ylabel("Mean amplitude")
    plt.title("Mean CSI Amplitude per Subcarrier")
    plt.grid(alpha=0.25)
    save_plot("01_subcarrier_mean.png")

    plt.figure(figsize=(14, 5))
    plt.plot(sc_std)
    plt.xlabel("Subcarrier")
    plt.ylabel("Standard deviation")
    plt.title("CSI Temporal Variation per Subcarrier")
    plt.grid(alpha=0.25)
    save_plot("02_subcarrier_std.png")

    plt.figure(figsize=(14, 5))
    plt.bar(
        np.arange(len(zero_rate)),
        zero_rate
    )
    plt.xlabel("Subcarrier")
    plt.ylabel("Zero fraction")
    plt.title("Zero Rate per CSI Subcarrier")
    plt.ylim(0, 1.05)
    save_plot("03_subcarrier_zero_rate.png")

    heat_count = min(
        MAX_HEATMAP_SAMPLES,
        len(csi)
    )

    plt.figure(figsize=(15, 7))
    plt.imshow(
        csi[:heat_count].T,
        aspect="auto",
        origin="lower"
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Sample")
    plt.ylabel("Subcarrier")
    plt.title(
        f"CSI Temporal Heatmap - First {heat_count} Samples"
    )
    save_plot("04_csi_heatmap.png")

    plt.figure(figsize=(14, 5))
    plt.plot(sample_energy, alpha=0.4)
    plt.plot(
        rolling_mean(
            sample_energy,
            ROLLING_WINDOW
        )
    )
    plt.xlabel("Sample")
    plt.ylabel("Mean squared amplitude")
    plt.title("CSI Energy over Time")
    plt.grid(alpha=0.25)
    save_plot("05_csi_energy.png")

    if len(normalized_diff):

        plt.figure(figsize=(14, 5))
        plt.plot(normalized_diff)
        plt.axhline(
            NEAR_DUP_THRESHOLD,
            linestyle="--"
        )
        plt.xlabel("Consecutive sample")
        plt.ylabel("Normalized RMS difference")
        plt.title(
            "CSI Consecutive-Sample Difference"
        )
        plt.grid(alpha=0.25)
        save_plot("06_csi_consecutive_difference.png")

    return {
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "sample_energy": sample_energy,
        "sample_zero_fraction": sample_zero_fraction,

        "subcarrier_mean": sc_mean,
        "subcarrier_std": sc_std,
        "subcarrier_zero_rate": zero_rate,

        "raw_consecutive_diff": raw_diff,
        "normalized_consecutive_diff": normalized_diff,

        "n_unique": n_unique,
        "exact_duplicates": exact_duplicate_samples,
        "duplicate_rate": duplicate_rate,
    }


# ============================================================
# POSE ANALYSIS
# ============================================================

def analyze_pose(data):

    pose = data.get("pose")
    confidence = data.get("pose_confidence")

    if pose is None or confidence is None:
        return {}

    pose = np.asarray(
        pose,
        dtype=np.float64
    )

    confidence = np.asarray(
        confidence,
        dtype=np.float64
    )

    separator("POSE QUALITY")

    valid_coordinate = (
        np.isfinite(pose[:, :, 0])
        & np.isfinite(pose[:, :, 1])
    )

    visible = (
        confidence >= POSE_CONF_THRESHOLD
    ) & valid_coordinate

    good = (
        confidence >= POSE_GOOD_CONF_THRESHOLD
    ) & valid_coordinate

    visible_count = np.sum(
        visible,
        axis=1
    )

    good_count = np.sum(
        good,
        axis=1
    )

    mean_conf = np.nanmean(
        confidence,
        axis=1
    )

    print(f"Pose samples: {len(pose)}")

    print()
    print("Confidence")
    print("-" * 50)

    print(
        f"Global mean   : "
        f"{np.nanmean(confidence):.4f}"
    )

    print(
        f"Global median : "
        f"{np.nanmedian(confidence):.4f}"
    )

    print(
        f"Visible KP >= {POSE_CONF_THRESHOLD}: "
        f"{np.mean(visible)*100:.2f}%"
    )

    print(
        f"Good KP >= {POSE_GOOD_CONF_THRESHOLD}: "
        f"{np.mean(good)*100:.2f}%"
    )

    print(
        "Visible keypoints/sample:",
        percentile_text(visible_count)
    )

    # --------------------------------------------------------
    # Keypoint confidence
    # --------------------------------------------------------

    kp_rows = []

    print()
    print("Per-keypoint")
    print("-" * 78)

    for i, name in enumerate(KEYPOINT_NAMES):

        kp_mean = np.nanmean(
            confidence[:, i]
        )

        kp_median = np.nanmedian(
            confidence[:, i]
        )

        visibility = np.mean(
            visible[:, i]
        )

        good_rate = np.mean(
            good[:, i]
        )

        print(
            f"{i:02d} {name:<16} "
            f"mean={kp_mean:.4f}  "
            f"median={kp_median:.4f}  "
            f"visible={visibility*100:6.2f}%  "
            f"conf>=0.5={good_rate*100:6.2f}%"
        )

        kp_rows.append({
            "index": i,
            "keypoint": name,
            "mean_confidence": kp_mean,
            "median_confidence": kp_median,
            "visibility_rate": visibility,
            "good_confidence_rate": good_rate,
        })

    pd.DataFrame(
        kp_rows
    ).to_csv(
        OUTPUT_DIR / "pose_keypoint_quality.csv",
        index=False
    )

    # --------------------------------------------------------
    # Body center
    #
    # shoulders + hips:
    # left shoulder 5
    # right shoulder 6
    # left hip 11
    # right hip 12
    # --------------------------------------------------------

    torso_indices = [5, 6, 11, 12]

    body_center = np.full(
        (len(pose), 2),
        np.nan
    )

    for i in range(len(pose)):

        valid_torso = [
            k
            for k in torso_indices
            if visible[i, k]
        ]

        if valid_torso:
            body_center[i] = np.mean(
                pose[i, valid_torso, :],
                axis=0
            )

    # --------------------------------------------------------
    # Bounding box using visible keypoints
    # --------------------------------------------------------

    bbox_width = np.full(
        len(pose),
        np.nan
    )

    bbox_height = np.full(
        len(pose),
        np.nan
    )

    bbox_area = np.full(
        len(pose),
        np.nan
    )

    for i in range(len(pose)):

        pts = pose[i][visible[i]]

        if len(pts) < 2:
            continue

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])

        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        w = xmax - xmin
        h = ymax - ymin

        bbox_width[i] = w
        bbox_height[i] = h
        bbox_area[i] = w * h

    print()
    print("Pose spatial diversity")
    print("-" * 50)

    print(
        "Body center X:",
        percentile_text(body_center[:, 0])
    )

    print(
        "Body center Y:",
        percentile_text(body_center[:, 1])
    )

    print(
        "BBox width:",
        percentile_text(bbox_width)
    )

    print(
        "BBox height:",
        percentile_text(bbox_height)
    )

    print(
        "BBox area:",
        percentile_text(bbox_area)
    )

    # --------------------------------------------------------
    # Pose motion
    # --------------------------------------------------------

    pose_motion = np.full(
        len(pose),
        np.nan
    )

    pose_motion[0] = 0.0

    for i in range(1, len(pose)):

        valid = (
            visible[i]
            & visible[i - 1]
        )

        if np.sum(valid) < 3:
            continue

        delta = (
            pose[i, valid]
            - pose[i - 1, valid]
        )

        distances = np.sqrt(
            np.sum(
                delta ** 2,
                axis=1
            )
        )

        pose_motion[i] = np.mean(
            distances
        )

    center_motion = np.full(
        len(pose),
        np.nan
    )

    center_motion[0] = 0.0

    valid_center = (
        np.isfinite(body_center[:, 0])
        & np.isfinite(body_center[:, 1])
    )

    for i in range(1, len(body_center)):

        if (
            valid_center[i]
            and valid_center[i - 1]
        ):

            center_motion[i] = np.linalg.norm(
                body_center[i]
                - body_center[i - 1]
            )

    print()
    print("Motion")
    print("-" * 50)

    print(
        "Pose keypoint motion:",
        percentile_text(pose_motion)
    )

    print(
        "Body-center motion:",
        percentile_text(center_motion)
    )

    # --------------------------------------------------------
    # Consecutive pose similarity
    # --------------------------------------------------------

    almost_static = (
        np.isfinite(pose_motion)
        & (pose_motion < 0.002)
    )

    print(
        f"Very-low-motion frames (<0.002): "
        f"{np.mean(almost_static)*100:.2f}%"
    )

    # --------------------------------------------------------
    # Pose report
    # --------------------------------------------------------

    pose_df = pd.DataFrame({
        "sample": np.arange(len(pose)),
        "mean_confidence": mean_conf,
        "visible_keypoints": visible_count,
        "good_keypoints": good_count,

        "body_center_x": body_center[:, 0],
        "body_center_y": body_center[:, 1],

        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,

        "pose_motion": pose_motion,
        "center_motion": center_motion,
    })

    pose_df.to_csv(
        OUTPUT_DIR / "pose_sample_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    kp_mean_conf = np.nanmean(
        confidence,
        axis=0
    )

    plt.figure(figsize=(14, 6))
    plt.bar(
        np.arange(EXPECTED_KEYPOINTS),
        kp_mean_conf
    )
    plt.xticks(
        np.arange(EXPECTED_KEYPOINTS),
        KEYPOINT_NAMES,
        rotation=55,
        ha="right"
    )
    plt.ylim(0, 1)
    plt.ylabel("Mean confidence")
    plt.title(
        "YOLO Pose Confidence per Keypoint"
    )
    save_plot("07_pose_keypoint_confidence.png")

    plt.figure(figsize=(8, 8))

    valid = (
        np.isfinite(body_center[:, 0])
        & np.isfinite(body_center[:, 1])
    )

    plt.scatter(
        body_center[valid, 0],
        body_center[valid, 1],
        s=10,
        alpha=0.4
    )

    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title("Body Center Distribution")
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.grid(alpha=0.25)
    save_plot("08_body_center_distribution.png")

    plt.figure(figsize=(14, 5))
    plt.plot(
        pose_motion,
        alpha=0.35
    )
    plt.plot(
        rolling_mean(
            pose_motion,
            ROLLING_WINDOW
        )
    )
    plt.xlabel("Sample")
    plt.ylabel("Mean keypoint displacement")
    plt.title("Camera Pose Motion")
    plt.grid(alpha=0.25)
    save_plot("09_pose_motion.png")

    plt.figure(figsize=(14, 5))
    plt.plot(
        bbox_area,
        alpha=0.45
    )
    plt.plot(
        rolling_mean(
            bbox_area,
            ROLLING_WINDOW
        )
    )
    plt.xlabel("Sample")
    plt.ylabel("Normalized bbox area")
    plt.title("Person Bounding Box Area")
    plt.grid(alpha=0.25)
    save_plot("10_pose_bbox_area.png")

    return {
        "visible": visible,
        "visible_count": visible_count,
        "mean_confidence": mean_conf,

        "body_center": body_center,

        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,

        "pose_motion": pose_motion,
        "center_motion": center_motion,
    }


# ============================================================
# SYNCHRONIZATION
# ============================================================

def analyze_sync(data):

    pose_age = data.get("pose_age_ms")

    if pose_age is None:
        return {}

    pose_age = np.asarray(
        pose_age,
        dtype=float
    )

    separator("CSI <-> CAMERA SYNCHRONIZATION")

    valid = pose_age[
        np.isfinite(pose_age)
    ]

    if len(valid) == 0:
        print("No valid synchronization values.")
        return {}

    print(f"Samples: {len(valid)}")
    print()
    print(
        f"min     : {np.min(valid):.3f} ms"
    )
    print(
        f"mean    : {np.mean(valid):.3f} ms"
    )
    print(
        f"median  : {np.median(valid):.3f} ms"
    )
    print(
        f"p90     : {np.percentile(valid,90):.3f} ms"
    )
    print(
        f"p95     : {np.percentile(valid,95):.3f} ms"
    )
    print(
        f"p99     : {np.percentile(valid,99):.3f} ms"
    )
    print(
        f"max     : {np.max(valid):.3f} ms"
    )

    print()

    thresholds = [
        10,
        20,
        30,
        50,
        75,
        100,
        200,
    ]

    sync_rows = []

    for threshold in thresholds:

        fraction = np.mean(
            valid <= threshold
        )

        print(
            f"<= {threshold:3d} ms : "
            f"{fraction*100:6.2f}%"
        )

        sync_rows.append({
            "threshold_ms": threshold,
            "fraction": fraction,
            "percent": fraction * 100,
        })

    pd.DataFrame(
        sync_rows
    ).to_csv(
        OUTPUT_DIR / "synchronization_thresholds.csv",
        index=False
    )

    plt.figure(figsize=(14, 5))
    plt.plot(
        pose_age,
        alpha=0.6
    )
    plt.axhline(
        50,
        linestyle="--",
        label="50 ms"
    )
    plt.axhline(
        100,
        linestyle="--",
        label="100 ms"
    )
    plt.xlabel("Sample")
    plt.ylabel("CSI-Pose age (ms)")
    plt.title("CSI / Camera Synchronization")
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot("11_synchronization.png")

    plt.figure(figsize=(9, 5))
    plt.hist(
        valid,
        bins=40
    )
    plt.xlabel("Synchronization age (ms)")
    plt.ylabel("Count")
    plt.title("Synchronization Delay Distribution")
    save_plot("12_sync_histogram.png")

    return {
        "mean_ms": np.mean(valid),
        "median_ms": np.median(valid),
        "p95_ms": np.percentile(valid, 95),
        "max_ms": np.max(valid),
    }


# ============================================================
# SAMPLE RATE / TIMING
# ============================================================

def analyze_timing(data):

    separator("TEMPORAL SAMPLING")

    results = {}

    for name in [
        "csi_arrival_time",
        "camera_time",
    ]:

        t = data.get(name)

        if t is None:
            continue

        t = np.asarray(
            t,
            dtype=float
        )

        t = t[np.isfinite(t)]

        if len(t) < 3:
            continue

        dt = np.diff(t)

        # Ignore negative and zero jumps
        valid_dt = dt[
            (dt > 0)
            & np.isfinite(dt)
        ]

        if len(valid_dt) == 0:
            continue

        median_dt = np.median(valid_dt)
        mean_dt = np.mean(valid_dt)

        median_hz = (
            1.0 / median_dt
            if median_dt > 0
            else np.nan
        )

        mean_hz = (
            1.0 / mean_dt
            if mean_dt > 0
            else np.nan
        )

        print()
        print(name)
        print("-" * 50)

        print(
            f"median dt  : {median_dt*1000:.3f} ms"
        )

        print(
            f"mean dt    : {mean_dt*1000:.3f} ms"
        )

        print(
            f"median FPS : {median_hz:.2f} Hz"
        )

        print(
            f"mean FPS   : {mean_hz:.2f} Hz"
        )

        print(
            f"p95 dt     : "
            f"{np.percentile(valid_dt,95)*1000:.3f} ms"
        )

        results[name] = {
            "median_dt": median_dt,
            "median_hz": median_hz,
            "mean_dt": mean_dt,
            "mean_hz": mean_hz,
        }

    return results


# ============================================================
# CSI <-> POSE RELATIONSHIP
# ============================================================

def analyze_relationship(
    data,
    csi_stats,
    pose_stats
):

    if not csi_stats or not pose_stats:
        return {}

    separator("CSI <-> POSE RELATIONSHIP")

    csi = np.asarray(
        data["csi_amplitude"],
        dtype=float
    )

    pose_motion = np.asarray(
        pose_stats["pose_motion"],
        dtype=float
    )

    center_motion = np.asarray(
        pose_stats["center_motion"],
        dtype=float
    )

    bbox_area = np.asarray(
        pose_stats["bbox_area"],
        dtype=float
    )

    body_center = np.asarray(
        pose_stats["body_center"],
        dtype=float
    )

    # --------------------------------------------------------
    # CSI temporal-motion features
    # --------------------------------------------------------

    csi_motion = np.zeros(
        len(csi),
        dtype=float
    )

    if len(csi) > 1:

        # Raw temporal change
        csi_motion[1:] = np.sqrt(
            np.mean(
                np.diff(csi, axis=0) ** 2,
                axis=1
            )
        )

    normalized = normalize_rows(csi)

    csi_shape_motion = np.zeros(
        len(csi),
        dtype=float
    )

    if len(csi) > 1:

        csi_shape_motion[1:] = np.sqrt(
            np.mean(
                np.diff(normalized, axis=0) ** 2,
                axis=1
            )
        )

    csi_energy = csi_stats[
        "sample_energy"
    ]

    csi_mean = csi_stats[
        "sample_mean"
    ]

    csi_std = csi_stats[
        "sample_std"
    ]

    relationships = {
        "CSI motion vs Pose motion":
            safe_corr(
                csi_motion,
                pose_motion
            ),

        "CSI shape motion vs Pose motion":
            safe_corr(
                csi_shape_motion,
                pose_motion
            ),

        "CSI motion vs Body-center motion":
            safe_corr(
                csi_motion,
                center_motion
            ),

        "CSI energy vs Pose motion":
            safe_corr(
                csi_energy,
                pose_motion
            ),

        "CSI energy vs BBox area":
            safe_corr(
                csi_energy,
                bbox_area
            ),

        "CSI mean vs BBox area":
            safe_corr(
                csi_mean,
                bbox_area
            ),

        "CSI std vs BBox area":
            safe_corr(
                csi_std,
                bbox_area
            ),

        "CSI energy vs Body center X":
            safe_corr(
                csi_energy,
                body_center[:, 0]
            ),

        "CSI energy vs Body center Y":
            safe_corr(
                csi_energy,
                body_center[:, 1]
            ),
    }

    for name, corr in relationships.items():

        print(
            f"{name:<40}: "
            f"{corr: .4f}"
        )

    relationship_df = pd.DataFrame([
        {
            "relationship": name,
            "pearson_correlation": value,
            "abs_correlation": abs(value)
            if np.isfinite(value)
            else np.nan,
        }
        for name, value in relationships.items()
    ])

    relationship_df.to_csv(
        OUTPUT_DIR / "csi_pose_correlations.csv",
        index=False
    )

    # --------------------------------------------------------
    # CSI subcarrier <-> pose motion correlation
    # --------------------------------------------------------

    subcarrier_corr = []

    for sc in range(
        csi.shape[1]
    ):

        corr = safe_corr(
            csi[:, sc],
            pose_motion
        )

        subcarrier_corr.append(
            corr
        )

    subcarrier_corr = np.asarray(
        subcarrier_corr
    )

    temporal_subcarrier_motion = np.zeros_like(
        csi
    )

    temporal_subcarrier_motion[1:] = np.abs(
        np.diff(csi, axis=0)
    )

    subcarrier_motion_corr = []

    for sc in range(
        csi.shape[1]
    ):

        corr = safe_corr(
            temporal_subcarrier_motion[:, sc],
            pose_motion
        )

        subcarrier_motion_corr.append(
            corr
        )

    subcarrier_motion_corr = np.asarray(
        subcarrier_motion_corr
    )

    corr_df = pd.DataFrame({
        "subcarrier":
            np.arange(csi.shape[1]),

        "amplitude_vs_pose_motion":
            subcarrier_corr,

        "abs_csi_change_vs_pose_motion":
            subcarrier_motion_corr,
    })

    corr_df.to_csv(
        OUTPUT_DIR /
        "subcarrier_pose_correlations.csv",
        index=False
    )

    # strongest subcarriers

    valid_corr = np.where(
        np.isfinite(
            subcarrier_motion_corr
        ),
        np.abs(
            subcarrier_motion_corr
        ),
        -1
    )

    strongest = np.argsort(
        valid_corr
    )[::-1][:10]

    print()
    print(
        "Top 10 subcarriers whose CHANGE "
        "correlates with pose motion"
    )
    print("-" * 60)

    for sc in strongest:

        print(
            f"SC {sc:3d}: "
            f"r={subcarrier_motion_corr[sc]: .4f}"
        )

    # --------------------------------------------------------
    # Plot temporal motion
    # --------------------------------------------------------

    csi_plot = rolling_mean(
        csi_shape_motion,
        ROLLING_WINDOW
    )

    pose_plot = rolling_mean(
        pose_motion,
        ROLLING_WINDOW
    )

    # Normalize just for visual comparison
    def zscore(x):

        x = np.asarray(
            x,
            dtype=float
        )

        finite = np.isfinite(x)

        result = np.full_like(
            x,
            np.nan
        )

        if np.sum(finite) < 2:
            return result

        m = np.nanmean(x)
        s = np.nanstd(x)

        if s < 1e-12:
            return x - m

        result[finite] = (
            x[finite] - m
        ) / s

        return result

    plt.figure(figsize=(15, 6))
    plt.plot(
        zscore(csi_plot),
        label="CSI temporal change"
    )
    plt.plot(
        zscore(pose_plot),
        label="Camera pose motion"
    )
    plt.xlabel("Sample")
    plt.ylabel("Normalized value")
    plt.title(
        "CSI Temporal Change vs Camera Pose Motion"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot("13_csi_vs_pose_motion.png")

    plt.figure(figsize=(14, 5))
    plt.plot(
        subcarrier_motion_corr
    )
    plt.axhline(
        0,
        linewidth=1
    )
    plt.xlabel("Subcarrier")
    plt.ylabel("Pearson correlation")
    plt.title(
        "|CSI(t)-CSI(t-1)| vs Pose Motion "
        "Correlation per Subcarrier"
    )
    plt.grid(alpha=0.25)
    save_plot(
        "14_subcarrier_motion_correlation.png"
    )

    return {
        "csi_motion":
            csi_motion,

        "csi_shape_motion":
            csi_shape_motion,

        "relationships":
            relationships,

        "subcarrier_motion_corr":
            subcarrier_motion_corr,
    }


# ============================================================
# TEMPORAL LAG ANALYSIS
# ============================================================

def lag_correlation(
    x,
    y,
    max_lag
):
    """
    Positive lag:
        x is shifted forward relative to y.

    We mainly use this to see whether the strongest CSI/Pose
    agreement occurs around lag=0 or some constant offset.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    rows = []

    for lag in range(
        -max_lag,
        max_lag + 1
    ):

        if lag < 0:

            xx = x[-lag:]
            yy = y[:lag]

        elif lag > 0:

            xx = x[:-lag]
            yy = y[lag:]

        else:

            xx = x
            yy = y

        corr = safe_corr(
            xx,
            yy
        )

        rows.append(
            (lag, corr)
        )

    return rows


def analyze_lag(
    relationship_stats,
    pose_stats
):

    if not relationship_stats:
        return {}

    separator("TEMPORAL LAG ANALYSIS")

    csi_motion = relationship_stats[
        "csi_shape_motion"
    ]

    pose_motion = pose_stats[
        "pose_motion"
    ]

    # smooth before lag analysis
    csi_smooth = rolling_mean(
        csi_motion,
        ROLLING_WINDOW
    )

    pose_smooth = rolling_mean(
        pose_motion,
        ROLLING_WINDOW
    )

    rows = lag_correlation(
        csi_smooth,
        pose_smooth,
        MAX_LAG_SAMPLES
    )

    lag_df = pd.DataFrame(
        rows,
        columns=[
            "lag_samples",
            "correlation"
        ]
    )

    lag_df["abs_correlation"] = np.abs(
        lag_df["correlation"]
    )

    lag_df.to_csv(
        OUTPUT_DIR /
        "temporal_lag_analysis.csv",
        index=False
    )

    valid = lag_df[
        np.isfinite(
            lag_df["correlation"]
        )
    ]

    if len(valid):

        strongest = valid.loc[
            valid["abs_correlation"].idxmax()
        ]

        best_lag = int(
            strongest["lag_samples"]
        )

        best_corr = float(
            strongest["correlation"]
        )

        zero_corr_row = valid[
            valid["lag_samples"] == 0
        ]

        zero_corr = (
            float(
                zero_corr_row[
                    "correlation"
                ].iloc[0]
            )
            if len(zero_corr_row)
            else np.nan
        )

        print(
            f"Correlation at lag 0 : "
            f"{zero_corr:.4f}"
        )

        print(
            f"Strongest lag        : "
            f"{best_lag:+d} samples"
        )

        print(
            f"Correlation there    : "
            f"{best_corr:.4f}"
        )

        if abs(best_lag) <= 2:

            print(
                "[GOOD] Strongest relationship is "
                "very close to temporal alignment."
            )

        else:

            print(
                "[INFO] Relationship peaks away from "
                "lag=0. This does NOT automatically mean "
                "synchronization is wrong; human motion / "
                "CSI response and smoothing can create lag."
            )

    else:

        best_lag = None
        best_corr = np.nan
        zero_corr = np.nan

    plt.figure(figsize=(12, 5))
    plt.plot(
        lag_df["lag_samples"],
        lag_df["correlation"],
        marker="o"
    )
    plt.axvline(
        0,
        linestyle="--"
    )
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")
    plt.title(
        "CSI Motion vs Camera Pose Motion - Lag Correlation"
    )
    plt.grid(alpha=0.25)
    save_plot("15_temporal_lag.png")

    return {
        "best_lag": best_lag,
        "best_corr": best_corr,
        "zero_corr": zero_corr,
    }


# ============================================================
# RSSI ANALYSIS
# ============================================================

def analyze_rssi(
    data,
    csi_stats
):

    rssi = data.get("rssi")

    if rssi is None:
        return {}

    rssi = np.asarray(
        rssi,
        dtype=float
    )

    separator("RSSI")

    print(
        "RSSI:",
        percentile_text(rssi)
    )

    result = {
        "mean": np.nanmean(rssi),
        "std": np.nanstd(rssi),
    }

    if csi_stats:

        energy = csi_stats[
            "sample_energy"
        ]

        corr = safe_corr(
            rssi,
            energy
        )

        print()
        print(
            f"RSSI vs CSI energy correlation: "
            f"{corr:.4f}"
        )

        result[
            "rssi_energy_corr"
        ] = corr

    plt.figure(figsize=(14, 5))
    plt.plot(
        rssi,
        alpha=0.5
    )
    plt.plot(
        rolling_mean(
            rssi,
            ROLLING_WINDOW
        )
    )
    plt.xlabel("Sample")
    plt.ylabel("RSSI (dBm)")
    plt.title("RSSI over Time")
    plt.grid(alpha=0.25)
    save_plot("16_rssi.png")

    return result


# ============================================================
# SESSION / GAP ANALYSIS
# ============================================================

def analyze_recording_gaps(data):

    t = data.get(
        "camera_time"
    )

    if t is None:
        return

    t = np.asarray(
        t,
        dtype=float
    )

    if len(t) < 2:
        return

    dt = np.diff(t)

    positive = dt[
        (dt > 0)
        & np.isfinite(dt)
    ]

    if len(positive) == 0:
        return

    median_dt = np.median(
        positive
    )

    # Large gap heuristic:
    # at least 1 second or 10x normal frame interval
    gap_threshold = max(
        1.0,
        median_dt * 10.0
    )

    gap_indices = np.where(
        dt > gap_threshold
    )[0]

    separator("RECORDING GAP ANALYSIS")

    print(
        f"Median interval : "
        f"{median_dt*1000:.3f} ms"
    )

    print(
        f"Gap threshold   : "
        f"{gap_threshold:.3f} sec"
    )

    print(
        f"Large gaps      : "
        f"{len(gap_indices)}"
    )

    rows = []

    for idx in gap_indices:

        rows.append({
            "before_sample": int(idx),
            "after_sample": int(idx + 1),
            "gap_seconds": float(dt[idx]),
        })

        print(
            f"sample {idx} -> {idx+1}: "
            f"{dt[idx]:.3f} sec"
        )

    pd.DataFrame(
        rows,
        columns=[
            "before_sample",
            "after_sample",
            "gap_seconds"
        ]
    ).to_csv(
        OUTPUT_DIR /
        "recording_gaps.csv",
        index=False
    )


# ============================================================
# MASTER SAMPLE MANIFEST
# ============================================================

def save_manifest(
    data,
    csi_stats,
    pose_stats
):

    n = len(
        data["csi_amplitude"]
    )

    manifest = pd.DataFrame({
        "global_index":
            np.arange(n),

        "source_file":
            data["source_file"],

        "source_index":
            data["source_index"],
    })

    optional = {
        "rssi":
            data.get("rssi"),

        "channel":
            data.get("channel"),

        "pose_age_ms":
            data.get("pose_age_ms"),

        "csi_arrival_time":
            data.get("csi_arrival_time"),

        "camera_time":
            data.get("camera_time"),
    }

    for name, arr in optional.items():

        if (
            arr is not None
            and len(arr) == n
        ):
            manifest[name] = arr

    if csi_stats:

        manifest[
            "csi_mean"
        ] = csi_stats["sample_mean"]

        manifest[
            "csi_std"
        ] = csi_stats["sample_std"]

        manifest[
            "csi_energy"
        ] = csi_stats["sample_energy"]

        manifest[
            "csi_zero_fraction"
        ] = csi_stats[
            "sample_zero_fraction"
        ]

    if pose_stats:

        manifest[
            "pose_mean_confidence"
        ] = pose_stats[
            "mean_confidence"
        ]

        manifest[
            "visible_keypoints"
        ] = pose_stats[
            "visible_count"
        ]

        manifest[
            "body_center_x"
        ] = pose_stats[
            "body_center"
        ][:, 0]

        manifest[
            "body_center_y"
        ] = pose_stats[
            "body_center"
        ][:, 1]

        manifest[
            "bbox_area"
        ] = pose_stats[
            "bbox_area"
        ]

        manifest[
            "pose_motion"
        ] = pose_stats[
            "pose_motion"
        ]

    manifest.to_csv(
        OUTPUT_DIR /
        "dataset_manifest.csv",
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def create_summary(
    data,
    csi_stats,
    pose_stats,
    sync_stats,
    timing_stats,
    relationship_stats,
    lag_stats,
    rssi_stats,
):

    separator("FINAL QA SUMMARY")

    n = len(
        data["csi_amplitude"]
    )

    summary = {}

    summary[
        "dataset_directory"
    ] = str(
        DATASET_DIR.resolve()
    )

    summary[
        "total_samples"
    ] = int(n)

    summary[
        "number_of_chunks"
    ] = int(
        len(
            np.unique(
                data["source_file"]
            )
        )
    )

    if csi_stats:

        summary[
            "csi_exact_duplicates"
        ] = int(
            csi_stats[
                "exact_duplicates"
            ]
        )

        summary[
            "csi_duplicate_rate"
        ] = float(
            csi_stats[
                "duplicate_rate"
            ]
        )

        summary[
            "mostly_zero_subcarriers"
        ] = int(
            np.sum(
                csi_stats[
                    "subcarrier_zero_rate"
                ]
                >= MOSTLY_ZERO_THRESHOLD
            )
        )

    if pose_stats:

        summary[
            "pose_global_mean_confidence"
        ] = float(
            np.nanmean(
                pose_stats[
                    "mean_confidence"
                ]
            )
        )

        summary[
            "mean_visible_keypoints"
        ] = float(
            np.nanmean(
                pose_stats[
                    "visible_count"
                ]
            )
        )

        summary[
            "median_pose_motion"
        ] = float(
            np.nanmedian(
                pose_stats[
                    "pose_motion"
                ]
            )
        )

    if sync_stats:

        summary.update({
            "sync_mean_ms":
                float(
                    sync_stats[
                        "mean_ms"
                    ]
                ),

            "sync_median_ms":
                float(
                    sync_stats[
                        "median_ms"
                    ]
                ),

            "sync_p95_ms":
                float(
                    sync_stats[
                        "p95_ms"
                    ]
                ),

            "sync_max_ms":
                float(
                    sync_stats[
                        "max_ms"
                    ]
                ),
        })

    if rssi_stats:

        summary[
            "rssi_mean"
        ] = float(
            rssi_stats["mean"]
        )

        summary[
            "rssi_std"
        ] = float(
            rssi_stats["std"]
        )

        if (
            "rssi_energy_corr"
            in rssi_stats
        ):

            summary[
                "rssi_csi_energy_correlation"
            ] = float(
                rssi_stats[
                    "rssi_energy_corr"
                ]
            )

    if lag_stats:

        summary[
            "csi_pose_zero_lag_correlation"
        ] = (
            float(
                lag_stats[
                    "zero_corr"
                ]
            )
            if np.isfinite(
                lag_stats[
                    "zero_corr"
                ]
            )
            else None
        )

        summary[
            "csi_pose_best_lag_samples"
        ] = lag_stats[
            "best_lag"
        ]

        summary[
            "csi_pose_best_lag_correlation"
        ] = (
            float(
                lag_stats[
                    "best_corr"
                ]
            )
            if np.isfinite(
                lag_stats[
                    "best_corr"
                ]
            )
            else None
        )

    summary[
        "timing"
    ] = timing_stats

    # JSON
    with open(
        OUTPUT_DIR /
        "qa_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    # Text version
    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "WiFi CSI + Camera Pose Dataset QA Summary"
    )

    lines.append(
        "=" * 78
    )

    for key, value in summary.items():

        lines.append(
            f"{key}: {value}"
        )

    with open(
        OUTPUT_DIR /
        "qa_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print()
    print(
        f"Samples              : {n}"
    )

    if csi_stats:

        print(
            f"CSI duplicates       : "
            f"{csi_stats['duplicate_rate']*100:.3f}%"
        )

    if pose_stats:

        print(
            f"Mean visible KPs      : "
            f"{np.nanmean(pose_stats['visible_count']):.2f}"
            f"/{EXPECTED_KEYPOINTS}"
        )

        print(
            f"Mean pose confidence  : "
            f"{np.nanmean(pose_stats['mean_confidence']):.4f}"
        )

    if sync_stats:

        print(
            f"Sync median           : "
            f"{sync_stats['median_ms']:.2f} ms"
        )

        print(
            f"Sync p95              : "
            f"{sync_stats['p95_ms']:.2f} ms"
        )

    if lag_stats:

        print(
            f"CSI/Pose best lag     : "
            f"{lag_stats['best_lag']} samples"
        )

        print(
            f"CSI/Pose lag corr     : "
            f"{lag_stats['best_corr']:.4f}"
        )

    print()
    print(
        f"All QA outputs saved to:"
    )
    print(
        OUTPUT_DIR.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    separator(
        "WiFi CSI + CAMERA POSE DATASET QA"
    )

    print(
        f"Dataset directory:"
    )
    print(
        DATASET_DIR.resolve()
    )

    print()
    print(
        f"Output directory:"
    )
    print(
        OUTPUT_DIR.resolve()
    )

    # --------------------------------------------------------
    # 1. Find chunks
    # --------------------------------------------------------

    files = find_chunk_files()

    inspect_chunks(
        files
    )

    # --------------------------------------------------------
    # 2. Load
    # --------------------------------------------------------

    data = load_dataset(
        files
    )

    if data.get(
        "csi_amplitude"
    ) is None:

        raise RuntimeError(
            "No CSI amplitude data was loaded."
        )

    # --------------------------------------------------------
    # 3. General validation
    # --------------------------------------------------------

    validate_dataset(
        data
    )

    # --------------------------------------------------------
    # 4. CSI
    # --------------------------------------------------------

    csi_stats = analyze_csi(
        data
    )

    # --------------------------------------------------------
    # 5. Pose
    # --------------------------------------------------------

    pose_stats = analyze_pose(
        data
    )

    # --------------------------------------------------------
    # 6. Synchronization
    # --------------------------------------------------------

    sync_stats = analyze_sync(
        data
    )

    # --------------------------------------------------------
    # 7. Timing
    # --------------------------------------------------------

    timing_stats = analyze_timing(
        data
    )

    # --------------------------------------------------------
    # 8. RSSI
    # --------------------------------------------------------

    rssi_stats = analyze_rssi(
        data,
        csi_stats
    )

    # --------------------------------------------------------
    # 9. CSI / Pose
    # --------------------------------------------------------

    relationship_stats = (
        analyze_relationship(
            data,
            csi_stats,
            pose_stats
        )
    )

    # --------------------------------------------------------
    # 10. Lag
    # --------------------------------------------------------

    lag_stats = analyze_lag(
        relationship_stats,
        pose_stats
    )

    # --------------------------------------------------------
    # 11. Recording gaps
    # --------------------------------------------------------

    analyze_recording_gaps(
        data
    )

    # --------------------------------------------------------
    # 12. Dataset manifest
    # --------------------------------------------------------

    save_manifest(
        data,
        csi_stats,
        pose_stats
    )

    # --------------------------------------------------------
    # 13. Summary
    # --------------------------------------------------------

    create_summary(
        data=data,
        csi_stats=csi_stats,
        pose_stats=pose_stats,
        sync_stats=sync_stats,
        timing_stats=timing_stats,
        relationship_stats=relationship_stats,
        lag_stats=lag_stats,
        rssi_stats=rssi_stats,
    )

    separator("DONE")

    print(
        "Dataset QA completed successfully."
    )


if __name__ == "__main__":
    main()