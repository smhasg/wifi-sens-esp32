#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

POSE_CONF_THRESHOLD = 0.20

# A CSI sample is considered fully usable when at least
# this many keypoints have valid coordinates/confidence.
MIN_VALID_KEYPOINTS = 8


# ============================================================
# HELPERS
# ============================================================

def percentile_safe(x, q):

    x = np.asarray(x)

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return np.percentile(x, q)


def print_stat(name, value):

    print(
        f"{name:35s}: {value}"
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(aligned_dir):

    aligned_dir = Path(
        aligned_dir
    )

    chunks = sorted(
        aligned_dir.glob(
            "chunk_*.npz"
        )
    )

    if not chunks:

        raise FileNotFoundError(
            f"No aligned chunks found in:\n"
            f"{aligned_dir}"
        )

    return chunks


# ============================================================
# QA
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="QA for aligned CSI + pose V3 dataset"
    )

    parser.add_argument(
        "aligned_dir",
        type=str,
        help="Path to aligned directory"
    )

    args = parser.parse_args()

    aligned_dir = Path(
        args.aligned_dir
    )

    chunks = load_chunks(
        aligned_dir
    )

    print("=" * 100)
    print("ALIGNED DATASET QA")
    print("=" * 100)

    print(
        f"Directory : {aligned_dir}"
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    # ========================================================
    # GLOBAL ACCUMULATORS
    # ========================================================

    total_samples = 0
    total_valid = 0

    total_nan_csi = 0
    total_inf_csi = 0

    total_nan_pose = 0
    total_nan_conf = 0

    total_duplicate = 0

    all_csi_energy = []
    all_rssi = []

    all_csi_time = []
    all_pose_age = []

    all_valid_kp_count = []

    method_counter = {}

    per_keypoint_conf = [[] for _ in range(17)]

    per_keypoint_valid = np.zeros(
        17,
        dtype=np.int64
    )

    # ========================================================
    # PROCESS CHUNKS
    # ========================================================

    for chunk_idx, chunk_file in enumerate(chunks):

        print()
        print("-" * 100)
        print(
            f"[{chunk_idx + 1}/{len(chunks)}] "
            f"{chunk_file.name}"
        )
        print("-" * 100)

        data = np.load(
            chunk_file,
            allow_pickle=True
        )

        required = [
            "csi_amplitude",
            "pose",
            "pose_confidence",
            "csi_time",
            "pose_time",
            "pose_valid",
            "pose_age_ms",
            "alignment_method",
        ]

        missing = [
            x
            for x in required
            if x not in data
        ]

        if missing:

            print(
                "ERROR: missing fields:",
                missing
            )

            continue

        csi = np.asarray(
            data["csi_amplitude"],
            dtype=np.float32
        )

        pose = np.asarray(
            data["pose"],
            dtype=np.float32
        )

        conf = np.asarray(
            data["pose_confidence"],
            dtype=np.float32
        )

        csi_time = np.asarray(
            data["csi_time"],
            dtype=np.float64
        )

        pose_time = np.asarray(
            data["pose_time"],
            dtype=np.float64
        )

        pose_valid = np.asarray(
            data["pose_valid"]
        ).astype(bool)

        pose_age = np.asarray(
            data["pose_age_ms"],
            dtype=np.float64
        )

        methods = np.asarray(
            data["alignment_method"]
        )

        n = len(csi)

        print_stat(
            "samples",
            n
        )

        print_stat(
            "CSI shape",
            csi.shape
        )

        print_stat(
            "pose shape",
            pose.shape
        )

        # ====================================================
        # SHAPE QA
        # ====================================================

        if csi.ndim != 2:

            print(
                "WARNING: unexpected CSI dimensions"
            )

        if (
            pose.ndim != 3
            or pose.shape[1:] != (17, 2)
        ):

            print(
                "WARNING: unexpected pose dimensions"
            )

        if (
            conf.ndim != 2
            or conf.shape[1] != 17
        ):

            print(
                "WARNING: unexpected confidence dimensions"
            )

        # ====================================================
        # NAN / INF
        # ====================================================

        nan_csi = int(
            np.isnan(csi).sum()
        )

        inf_csi = int(
            np.isinf(csi).sum()
        )

        nan_pose = int(
            np.isnan(pose).sum()
        )

        nan_conf = int(
            np.isnan(conf).sum()
        )

        total_nan_csi += nan_csi
        total_inf_csi += inf_csi
        total_nan_pose += nan_pose
        total_nan_conf += nan_conf

        print_stat(
            "CSI NaN",
            nan_csi
        )

        print_stat(
            "CSI Inf",
            inf_csi
        )

        print_stat(
            "Pose NaN",
            nan_pose
        )

        print_stat(
            "Confidence NaN",
            nan_conf
        )

        # ====================================================
        # CSI STATISTICS
        # ====================================================

        finite_csi = csi[
            np.isfinite(csi)
        ]

        if len(finite_csi):

            print_stat(
                "CSI min",
                float(np.min(finite_csi))
            )

            print_stat(
                "CSI max",
                float(np.max(finite_csi))
            )

            print_stat(
                "CSI mean",
                float(np.mean(finite_csi))
            )

            print_stat(
                "CSI median",
                float(np.median(finite_csi))
            )

            print_stat(
                "CSI std",
                float(np.std(finite_csi))
            )

        # Per-sample CSI energy

        csi_energy = np.sqrt(
            np.mean(
                np.square(csi),
                axis=1
            )
        )

        all_csi_energy.extend(
            csi_energy.tolist()
        )

        # ====================================================
        # POSE KEYPOINT QUALITY
        # ====================================================

        kp_valid = (
            np.isfinite(pose).all(axis=2)
            & np.isfinite(conf)
            & (
                conf >= POSE_CONF_THRESHOLD
            )
        )

        valid_kp_count = (
            kp_valid.sum(axis=1)
        )

        all_valid_kp_count.extend(
            valid_kp_count.tolist()
        )

        strong_pose = (
            valid_kp_count
            >= MIN_VALID_KEYPOINTS
        )

        print_stat(
            "pose_valid samples",
            int(pose_valid.sum())
        )

        print_stat(
            "pose_valid %",
            f"{100.0 * pose_valid.mean():.2f}%"
        )

        print_stat(
            f">={MIN_VALID_KEYPOINTS} KP samples",
            int(strong_pose.sum())
        )

        print_stat(
            f">={MIN_VALID_KEYPOINTS} KP %",
            f"{100.0 * strong_pose.mean():.2f}%"
        )

        print_stat(
            "valid KP/sample median",
            float(np.median(valid_kp_count))
        )

        print_stat(
            "valid KP/sample p95",
            float(
                np.percentile(
                    valid_kp_count,
                    95
                )
            )
        )

        print_stat(
            "valid KP/sample max",
            int(np.max(valid_kp_count))
        )

        # ====================================================
        # PER KEYPOINT
        # ====================================================

        for k in range(17):

            valid_k = kp_valid[:, k]

            count = int(
                valid_k.sum()
            )

            per_keypoint_valid[k] += count

            if count:

                vals = conf[
                    valid_k,
                    k
                ]

                per_keypoint_conf[k].extend(
                    vals.tolist()
                )

        # ====================================================
        # ALIGNMENT
        # ====================================================

        valid_alignment = (
            pose_valid
        )

        total_valid += int(
            valid_alignment.sum()
        )

        # Pose age

        valid_age = pose_age[
            valid_alignment
            & np.isfinite(pose_age)
        ]

        if len(valid_age):

            all_pose_age.extend(
                valid_age.tolist()
            )

        print_stat(
            "pose age median",
            (
                f"{np.median(valid_age):.2f} ms"
                if len(valid_age)
                else "N/A"
            )
        )

        print_stat(
            "pose age p95",
            (
                f"{np.percentile(valid_age, 95):.2f} ms"
                if len(valid_age)
                else "N/A"
            )
        )

        print_stat(
            "pose age max",
            (
                f"{np.max(valid_age):.2f} ms"
                if len(valid_age)
                else "N/A"
            )
        )

        # ====================================================
        # ALIGNMENT METHODS
        # ====================================================

        unique_methods, counts = np.unique(
            methods,
            return_counts=True
        )

        for method, count in zip(
            unique_methods,
            counts
        ):

            method = str(method)

            method_counter[method] = (
                method_counter.get(
                    method,
                    0
                )
                + int(count)
            )

        # ====================================================
        # TIME
        # ====================================================

        finite_time = csi_time[
            np.isfinite(csi_time)
        ]

        if len(finite_time):

            all_csi_time.extend(
                finite_time.tolist()
            )

            if len(finite_time) > 1:

                dt = np.diff(
                    finite_time
                )

                dt = dt[
                    dt > 0
                ]

                if len(dt):

                    print_stat(
                        "CSI dt median",
                        f"{np.median(dt) * 1000:.2f} ms"
                    )

                    print_stat(
                        "CSI FPS median",
                        f"{1.0 / np.median(dt):.2f} Hz"
                    )

        # ====================================================
        # RSSI
        # ====================================================

        if "rssi" in data:

            rssi = np.asarray(
                data["rssi"]
            ).astype(
                np.float64
            )

            rssi = rssi[
                np.isfinite(rssi)
            ]

            all_rssi.extend(
                rssi.tolist()
            )

        # ====================================================
        # DUPLICATES
        # ====================================================

        if n > 1:

            # Hash rows after converting to bytes.
            # Exact duplicate CSI samples.
            unique_count = len(
                np.unique(
                    csi.view(
                        np.dtype(
                            (np.void,
                             csi.dtype.itemsize
                             * csi.shape[1])
                        )
                    )
                )
            )

            duplicate_count = (
                n - unique_count
            )

            total_duplicate += (
                duplicate_count
            )

            print_stat(
                "exact CSI duplicates",
                duplicate_count
            )

        total_samples += n

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("GLOBAL QA SUMMARY")
    print("=" * 100)

    print_stat(
        "Total chunks",
        len(chunks)
    )

    print_stat(
        "Total samples",
        total_samples
    )

    print()

    print_stat(
        "Total valid aligned",
        total_valid
    )

    print_stat(
        "Alignment ratio",
        f"{100.0 * total_valid / max(total_samples, 1):.2f}%"
    )

    # ========================================================
    # CSI QUALITY
    # ========================================================

    print()
    print("=" * 100)
    print("CSI QUALITY")
    print("=" * 100)

    if all_csi_energy:

        energy = np.asarray(
            all_csi_energy
        )

        print_stat(
            "energy mean",
            float(np.mean(energy))
        )

        print_stat(
            "energy median",
            float(np.median(energy))
        )

        print_stat(
            "energy std",
            float(np.std(energy))
        )

    print_stat(
        "CSI NaN total",
        total_nan_csi
    )

    print_stat(
        "CSI Inf total",
        total_inf_csi
    )

    print_stat(
        "Exact duplicate samples",
        total_duplicate
    )

    # ========================================================
    # POSE QUALITY
    # ========================================================

    print()
    print("=" * 100)
    print("POSE QUALITY")
    print("=" * 100)

    if all_valid_kp_count:

        vk = np.asarray(
            all_valid_kp_count
        )

        print_stat(
            "valid KP median",
            float(np.median(vk))
        )

        print_stat(
            "valid KP mean",
            float(np.mean(vk))
        )

        print_stat(
            "valid KP p25",
            float(np.percentile(vk, 25))
        )

        print_stat(
            "valid KP p75",
            float(np.percentile(vk, 75))
        )

        print_stat(
            "valid KP p95",
            float(np.percentile(vk, 95))
        )

        print_stat(
            "valid KP max",
            int(np.max(vk))
        )

    print()
    print("Per-keypoint quality:")

    for k in range(17):

        count = per_keypoint_valid[k]

        if per_keypoint_conf[k]:

            mean_conf = np.mean(
                per_keypoint_conf[k]
            )

            median_conf = np.median(
                per_keypoint_conf[k]
            )

        else:

            mean_conf = np.nan
            median_conf = np.nan

        ratio = (
            count / total_samples
            if total_samples
            else 0.0
        )

        print(
            f"KP {k:02d} | "
            f"valid={count:7d} | "
            f"ratio={ratio * 100:6.2f}% | "
            f"mean_conf={mean_conf:.3f} | "
            f"median_conf={median_conf:.3f}"
        )

    # ========================================================
    # ALIGNMENT AGE
    # ========================================================

    print()
    print("=" * 100)
    print("ALIGNMENT TIMING")
    print("=" * 100)

    if all_pose_age:

        ages = np.asarray(
            all_pose_age
        )

        print_stat(
            "age median",
            f"{np.median(ages):.2f} ms"
        )

        print_stat(
            "age mean",
            f"{np.mean(ages):.2f} ms"
        )

        print_stat(
            "age p90",
            f"{np.percentile(ages, 90):.2f} ms"
        )

        print_stat(
            "age p95",
            f"{np.percentile(ages, 95):.2f} ms"
        )

        print_stat(
            "age p99",
            f"{np.percentile(ages, 99):.2f} ms"
        )

        print_stat(
            "age max",
            f"{np.max(ages):.2f} ms"
        )

    # ========================================================
    # METHODS
    # ========================================================

    print()
    print("=" * 100)
    print("ALIGNMENT METHODS")
    print("=" * 100)

    for method, count in sorted(
        method_counter.items()
    ):

        ratio = (
            count / total_samples
            if total_samples
            else 0.0
        )

        print(
            f"{method:20s}: "
            f"{count:8d} "
            f"({ratio * 100:6.2f}%)"
        )

    # ========================================================
    # TIME / FPS
    # ========================================================

    print()
    print("=" * 100)
    print("CSI TIMELINE")
    print("=" * 100)

    if len(all_csi_time) > 1:

        t = np.asarray(
            all_csi_time
        )

        t = np.sort(t)

        dt = np.diff(t)

        dt = dt[
            dt > 0
        ]

        if len(dt):

            print_stat(
                "duration",
                f"{t[-1] - t[0]:.2f} sec"
            )

            print_stat(
                "dt median",
                f"{np.median(dt) * 1000:.2f} ms"
            )

            print_stat(
                "dt mean",
                f"{np.mean(dt) * 1000:.2f} ms"
            )

            print_stat(
                "dt p95",
                f"{np.percentile(dt, 95) * 1000:.2f} ms"
            )

            print_stat(
                "FPS median",
                f"{1.0 / np.median(dt):.2f} Hz"
            )

            print_stat(
                "FPS mean",
                f"{1.0 / np.mean(dt):.2f} Hz"
            )

    # ========================================================
    # RSSI
    # ========================================================

    print()
    print("=" * 100)
    print("RSSI")
    print("=" * 100)

    if all_rssi:

        rssi = np.asarray(
            all_rssi
        )

        print_stat(
            "min",
            float(np.min(rssi))
        )

        print_stat(
            "p25",
            float(np.percentile(rssi, 25))
        )

        print_stat(
            "median",
            float(np.median(rssi))
        )

        print_stat(
            "p75",
            float(np.percentile(rssi, 75))
        )

        print_stat(
            "p95",
            float(np.percentile(rssi, 95))
        )

        print_stat(
            "max",
            float(np.max(rssi))
        )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    alignment_ratio = (
        total_valid
        / max(total_samples, 1)
    )

    print()
    print("=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)

    if total_nan_csi > 0:
        print(
            "[WARNING] CSI contains NaN values."
        )
    else:
        print(
            "[OK] CSI contains no NaN."
        )

    if total_inf_csi > 0:
        print(
            "[WARNING] CSI contains Inf values."
        )
    else:
        print(
            "[OK] CSI contains no Inf."
        )

    if total_duplicate > 0:
        print(
            f"[WARNING] "
            f"{total_duplicate} exact CSI duplicates."
        )
    else:
        print(
            "[OK] No exact CSI duplicates."
        )

    if alignment_ratio >= 0.80:

        print(
            f"[GOOD] Alignment ratio = "
            f"{alignment_ratio * 100:.2f}%"
        )

    elif alignment_ratio >= 0.60:

        print(
            f"[ACCEPTABLE] Alignment ratio = "
            f"{alignment_ratio * 100:.2f}%"
        )

    elif alignment_ratio >= 0.40:

        print(
            f"[WEAK] Alignment ratio = "
            f"{alignment_ratio * 100:.2f}%"
        )

    else:

        print(
            f"[BAD] Alignment ratio = "
            f"{alignment_ratio * 100:.2f}%"
        )

    print()

    if alignment_ratio >= 0.60:

        print(
            ">>> Dataset is potentially ready "
            "for baseline CSI -> Pose training."
        )

    else:

        print(
            ">>> Dataset is NOT ready for serious "
            "pose training yet."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()