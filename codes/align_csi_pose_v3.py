#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import math
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

POSE_CONF_THRESHOLD = 0.20

# Maximum time gap allowed for interpolation.
# Example:
#   previous pose = t0
#   next pose     = t1
#
# If CSI time is inside this interval and the interval is not
# too large, interpolate.
MAX_INTERPOLATION_GAP_MS = 1000.0

# If interpolation is impossible, allow nearest pose only
# when it is close enough.
MAX_NEAREST_POSE_AGE_MS = 300.0

# A pose is considered valid only if this many keypoints
# have confidence >= POSE_CONF_THRESHOLD.
MIN_VALID_KEYPOINTS = 8

CHUNK_SIZE = 5000


# ============================================================
# UTILS
# ============================================================

def load_pose_timeline(session_dir):
    pose_file = (
        Path(session_dir)
        / "pose"
        / "pose_timeline.npz"
    )

    if not pose_file.exists():
        raise FileNotFoundError(
            f"Pose timeline not found:\n{pose_file}"
        )

    print("=" * 80)
    print("Loading pose timeline")
    print("=" * 80)
    print(pose_file)

    data = np.load(
        pose_file,
        allow_pickle=True
    )

    required = [
        "camera_time",
        "pose",
        "pose_confidence",
    ]

    for key in required:
        if key not in data:
            raise KeyError(
                f"Required field '{key}' not found in {pose_file}"
            )

    camera_time = np.asarray(
        data["camera_time"],
        dtype=np.float64
    )

    pose = np.asarray(
        data["pose"],
        dtype=np.float32
    )

    pose_confidence = np.asarray(
        data["pose_confidence"],
        dtype=np.float32
    )

    print(f"camera_time shape      : {camera_time.shape}")
    print(f"pose shape             : {pose.shape}")
    print(f"pose_confidence shape  : {pose_confidence.shape}")

    if pose.ndim != 3 or pose.shape[1:] != (17, 2):
        raise ValueError(
            f"Unexpected pose shape: {pose.shape}"
        )

    if pose_confidence.ndim != 2 or pose_confidence.shape[1] != 17:
        raise ValueError(
            f"Unexpected pose_confidence shape: "
            f"{pose_confidence.shape}"
        )

    if len(camera_time) != len(pose):
        raise ValueError(
            "camera_time and pose lengths do not match"
        )

    if len(camera_time) != len(pose_confidence):
        raise ValueError(
            "camera_time and pose_confidence lengths do not match"
        )

    # --------------------------------------------------------
    # Remove NaN camera timestamps
    # --------------------------------------------------------

    finite_time = np.isfinite(camera_time)

    camera_time = camera_time[finite_time]
    pose = pose[finite_time]
    pose_confidence = pose_confidence[finite_time]

    # --------------------------------------------------------
    # Sort by time
    # --------------------------------------------------------

    order = np.argsort(camera_time)

    camera_time = camera_time[order]
    pose = pose[order]
    pose_confidence = pose_confidence[order]

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # Keep first occurrence
    # --------------------------------------------------------

    if len(camera_time) > 1:

        keep = np.ones(len(camera_time), dtype=bool)

        keep[1:] = (
            np.diff(camera_time) > 0
        )

        camera_time = camera_time[keep]
        pose = pose[keep]
        pose_confidence = pose_confidence[keep]

    print()
    print(f"Valid pose timeline frames : {len(camera_time)}")

    if len(camera_time) == 0:
        raise RuntimeError(
            "No valid pose timestamps found."
        )

    # --------------------------------------------------------
    # Pose quality
    # --------------------------------------------------------

    valid_kp_count = (
        np.isfinite(pose).all(axis=2)
        & (pose_confidence >= POSE_CONF_THRESHOLD)
    ).sum(axis=1)

    pose_frame_valid = (
        valid_kp_count >= MIN_VALID_KEYPOINTS
    )

    print(
        f"Frames with >= {MIN_VALID_KEYPOINTS} valid "
        f"keypoints: "
        f"{pose_frame_valid.sum()} / {len(pose_frame_valid)} "
        f"({pose_frame_valid.mean() * 100:.2f}%)"
    )

    print(
        f"Pose time range: "
        f"{camera_time[0]:.6f} -> "
        f"{camera_time[-1]:.6f}"
    )

    return (
        camera_time,
        pose,
        pose_confidence,
        valid_kp_count,
        pose_frame_valid,
    )


# ============================================================
# POSE INTERPOLATION
# ============================================================

def interpolate_pose(
    t,
    pose_time,
    pose,
    pose_conf,
):
    """
    Interpolate pose at timestamp t.

    Returns:
        pose_out
        conf_out
        valid
        method
        age_ms
    """

    n = len(pose_time)

    if n == 0:
        return (
            np.full((17, 2), np.nan, dtype=np.float32),
            np.full(17, np.nan, dtype=np.float32),
            False,
            "none",
            np.nan,
        )

    # --------------------------------------------------------
    # Position in pose timeline
    # --------------------------------------------------------

    idx = np.searchsorted(
        pose_time,
        t,
        side="left"
    )

    # --------------------------------------------------------
    # Before first pose
    # --------------------------------------------------------

    if idx == 0:

        age_ms = abs(t - pose_time[0]) * 1000.0

        if age_ms <= MAX_NEAREST_POSE_AGE_MS:

            return (
                pose[0].copy(),
                pose_conf[0].copy(),
                True,
                "next",
                age_ms,
            )

        return (
            np.full((17, 2), np.nan, dtype=np.float32),
            np.full(17, np.nan, dtype=np.float32),
            False,
            "none",
            age_ms,
        )

    # --------------------------------------------------------
    # After last pose
    # --------------------------------------------------------

    if idx >= n:

        age_ms = abs(t - pose_time[-1]) * 1000.0

        if age_ms <= MAX_NEAREST_POSE_AGE_MS:

            return (
                pose[-1].copy(),
                pose_conf[-1].copy(),
                True,
                "previous",
                age_ms,
            )

        return (
            np.full((17, 2), np.nan, dtype=np.float32),
            np.full(17, np.nan, dtype=np.float32),
            False,
            "none",
            age_ms,
        )

    # --------------------------------------------------------
    # Exact / surrounding frames
    # --------------------------------------------------------

    t1 = pose_time[idx]
    t0 = pose_time[idx - 1]

    p0 = pose[idx - 1]
    p1 = pose[idx]

    c0 = pose_conf[idx - 1]
    c1 = pose_conf[idx]

    dt = t1 - t0

    # Exact timestamp
    if dt <= 0:
        age_ms = abs(t - t0) * 1000.0

        if age_ms <= MAX_NEAREST_POSE_AGE_MS:

            return (
                p0.copy(),
                c0.copy(),
                True,
                "exact",
                age_ms,
            )

        return (
            np.full((17, 2), np.nan, dtype=np.float32),
            np.full(17, np.nan, dtype=np.float32),
            False,
            "none",
            age_ms,
        )

    gap_ms = dt * 1000.0

    # --------------------------------------------------------
    # Interpolation
    # --------------------------------------------------------

    if gap_ms <= MAX_INTERPOLATION_GAP_MS:

        alpha = (
            (t - t0)
            / (t1 - t0)
        )

        alpha = float(
            np.clip(alpha, 0.0, 1.0)
        )

        out_pose = np.full(
            (17, 2),
            np.nan,
            dtype=np.float32
        )

        out_conf = np.full(
            17,
            np.nan,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Per-keypoint interpolation
        # ----------------------------------------------------

        for k in range(17):

            p0_valid = (
                np.isfinite(p0[k]).all()
                and np.isfinite(c0[k])
                and c0[k] >= POSE_CONF_THRESHOLD
            )

            p1_valid = (
                np.isfinite(p1[k]).all()
                and np.isfinite(c1[k])
                and c1[k] >= POSE_CONF_THRESHOLD
            )

            if p0_valid and p1_valid:

                out_pose[k] = (
                    (1.0 - alpha) * p0[k]
                    + alpha * p1[k]
                )

                # Conservative confidence:
                # use minimum confidence of two endpoints.
                out_conf[k] = min(
                    c0[k],
                    c1[k]
                )

            elif p0_valid:

                out_pose[k] = p0[k]
                out_conf[k] = c0[k]

            elif p1_valid:

                out_pose[k] = p1[k]
                out_conf[k] = c1[k]

        valid_kp = (
            np.isfinite(out_pose).all(axis=1)
            & np.isfinite(out_conf)
            & (out_conf >= POSE_CONF_THRESHOLD)
        )

        valid_count = valid_kp.sum()

        if valid_count >= MIN_VALID_KEYPOINTS:

            # Distance to nearest endpoint
            nearest_age = min(
                abs(t - t0),
                abs(t - t1)
            ) * 1000.0

            return (
                out_pose,
                out_conf,
                True,
                "interpolated",
                nearest_age,
            )

    # --------------------------------------------------------
    # Fallback: nearest
    # --------------------------------------------------------

    age0 = abs(t - t0)
    age1 = abs(t - t1)

    if age0 <= age1:

        nearest_idx = idx - 1
        age_ms = age0 * 1000.0
        method = "previous"

    else:

        nearest_idx = idx
        age_ms = age1 * 1000.0
        method = "next"

    if age_ms <= MAX_NEAREST_POSE_AGE_MS:

        valid_kp = (
            np.isfinite(pose[nearest_idx]).all(axis=1)
            & np.isfinite(pose_conf[nearest_idx])
            & (
                pose_conf[nearest_idx]
                >= POSE_CONF_THRESHOLD
            )
        )

        if valid_kp.sum() >= MIN_VALID_KEYPOINTS:

            return (
                pose[nearest_idx].copy(),
                pose_conf[nearest_idx].copy(),
                True,
                method,
                age_ms,
            )

    return (
        np.full((17, 2), np.nan, dtype=np.float32),
        np.full(17, np.nan, dtype=np.float32),
        False,
        "none",
        min(age0, age1) * 1000.0,
    )


# ============================================================
# CSI CHUNK LOADING
# ============================================================

def get_csi_chunks(session_dir):

    csi_dir = (
        Path(session_dir)
        / "csi"
    )

    chunks = sorted(
        csi_dir.glob("chunk_*.npz")
    )

    if not chunks:
        raise FileNotFoundError(
            f"No CSI chunks found in:\n{csi_dir}"
        )

    return chunks


# ============================================================
# ALIGN ONE CHUNK
# ============================================================

def align_chunk(
    csi_file,
    pose_time,
    pose,
    pose_conf,
):
    print()
    print("-" * 80)
    print(f"Processing: {csi_file.name}")
    print("-" * 80)

    data = np.load(
        csi_file,
        allow_pickle=True
    )

    if "csi_arrival_time" not in data:
        raise KeyError(
            f"'csi_arrival_time' missing from {csi_file}"
        )

    csi_time = np.asarray(
        data["csi_arrival_time"],
        dtype=np.float64
    )

    n = len(csi_time)

    poses = np.full(
        (n, 17, 2),
        np.nan,
        dtype=np.float32
    )

    confs = np.full(
        (n, 17),
        np.nan,
        dtype=np.float32
    )

    pose_time_out = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    pose_valid = np.zeros(
        n,
        dtype=np.uint8
    )

    pose_age_ms = np.full(
        n,
        np.nan,
        dtype=np.float32
    )

    alignment_method = np.empty(
        n,
        dtype="<U16"
    )

    alignment_method[:] = "none"

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    for i, t in enumerate(csi_time):

        if not np.isfinite(t):
            continue

        (
            p,
            c,
            valid,
            method,
            age_ms,
        ) = interpolate_pose(
            t,
            pose_time,
            pose,
            pose_conf,
        )

        poses[i] = p
        confs[i] = c

        pose_valid[i] = 1 if valid else 0

        pose_age_ms[i] = (
            age_ms
            if np.isfinite(age_ms)
            else np.nan
        )

        alignment_method[i] = method

        # Find the actual nearest/interpolated reference time.
        idx = np.searchsorted(
            pose_time,
            t,
            side="left"
        )

        if idx <= 0:
            pose_time_out[i] = pose_time[0]

        elif idx >= len(pose_time):
            pose_time_out[i] = pose_time[-1]

        else:
            d0 = abs(t - pose_time[idx - 1])
            d1 = abs(t - pose_time[idx])

            if d0 <= d1:
                pose_time_out[i] = pose_time[idx - 1]
            else:
                pose_time_out[i] = pose_time[idx]

    # --------------------------------------------------------
    # Copy original CSI fields
    # --------------------------------------------------------

    output = {}

    for key in data.files:

        output[key] = data[key]

    # --------------------------------------------------------
    # Add aligned fields
    # --------------------------------------------------------

    output["pose"] = poses
    output["pose_confidence"] = confs

    output["csi_time"] = csi_time
    output["pose_time"] = pose_time_out

    output["pose_valid"] = pose_valid

    output["pose_age_ms"] = pose_age_ms

    output["alignment_method"] = alignment_method

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    valid_count = int(
        pose_valid.sum()
    )

    print(f"CSI samples       : {n}")
    print(
        f"Aligned valid     : {valid_count}"
        f" ({100.0 * valid_count / max(n, 1):.2f}%)"
    )

    if valid_count:

        valid_age = pose_age_ms[
            pose_valid.astype(bool)
        ]

        print(
            f"Pose age median   : "
            f"{np.nanmedian(valid_age):.2f} ms"
        )

        print(
            f"Pose age p95      : "
            f"{np.nanpercentile(valid_age, 95):.2f} ms"
        )

    unique_methods, counts = np.unique(
        alignment_method,
        return_counts=True
    )

    print("Alignment methods:")

    for method, count in zip(
        unique_methods,
        counts
    ):
        print(
            f"  {method:15s}: "
            f"{count:7d} "
            f"({100.0 * count / max(n, 1):6.2f}%)"
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Align V3 CSI timeline with offline "
            "camera pose timeline."
        )
    )

    parser.add_argument(
        "session_dir",
        type=str,
        help="Path to V3 session directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output aligned directory"
    )

    args = parser.parse_args()

    session_dir = Path(
        args.session_dir
    )

    if not session_dir.exists():
        raise FileNotFoundError(
            f"Session directory does not exist:\n"
            f"{session_dir}"
        )

    if args.output:

        output_dir = Path(
            args.output
        )

    else:

        output_dir = (
            session_dir
            / "aligned"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load pose
    # --------------------------------------------------------

    (
        pose_time,
        pose,
        pose_conf,
        valid_kp_count,
        pose_frame_valid,
    ) = load_pose_timeline(
        session_dir
    )

    # --------------------------------------------------------
    # CSI chunks
    # --------------------------------------------------------

    chunks = get_csi_chunks(
        session_dir
    )

    print()
    print("=" * 80)
    print("ALIGNMENT")
    print("=" * 80)

    print(
        f"CSI chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Global counters
    # --------------------------------------------------------

    total_samples = 0
    total_valid = 0

    method_counter = {}

    valid_ages = []

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    for chunk_index, csi_file in enumerate(chunks):

        aligned = align_chunk(
            csi_file,
            pose_time,
            pose,
            pose_conf,
        )

        output_file = (
            output_dir
            / csi_file.name
        )

        np.savez_compressed(
            output_file,
            **aligned
        )

        n = len(
            aligned["csi_time"]
        )

        valid = int(
            aligned["pose_valid"].sum()
        )

        total_samples += n
        total_valid += valid

        methods, counts = np.unique(
            aligned["alignment_method"],
            return_counts=True
        )

        for method, count in zip(
            methods,
            counts
        ):
            method_counter[method] = (
                method_counter.get(method, 0)
                + int(count)
            )

        valid_mask = (
            aligned["pose_valid"].astype(bool)
        )

        if valid_mask.any():

            valid_ages.extend(
                aligned["pose_age_ms"][
                    valid_mask
                ].tolist()
            )

        print(
            f"Saved -> {output_file}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "source_session": str(
            session_dir
        ),

        "pose_conf_threshold":
            POSE_CONF_THRESHOLD,

        "min_valid_keypoints":
            MIN_VALID_KEYPOINTS,

        "max_interpolation_gap_ms":
            MAX_INTERPOLATION_GAP_MS,

        "max_nearest_pose_age_ms":
            MAX_NEAREST_POSE_AGE_MS,

        "total_csi_samples":
            total_samples,

        "total_valid_aligned_samples":
            total_valid,

        "alignment_ratio":
            (
                total_valid / total_samples
                if total_samples
                else 0.0
            ),

        "alignment_methods":
            method_counter,

        "pose_frames":
            len(pose_time),

        "pose_valid_frames":
            int(pose_frame_valid.sum()),

        "pose_time_start":
            float(pose_time[0]),

        "pose_time_end":
            float(pose_time[-1]),
    }

    if valid_ages:

        metadata["pose_age_median_ms"] = float(
            np.median(valid_ages)
        )

        metadata["pose_age_p95_ms"] = float(
            np.percentile(
                valid_ages,
                95
            )
        )

        metadata["pose_age_max_ms"] = float(
            np.max(valid_ages)
        )

    metadata_file = (
        output_dir
        / "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ALIGNMENT COMPLETE")
    print("=" * 80)

    print(
        f"Total CSI samples : {total_samples}"
    )

    print(
        f"Valid aligned     : {total_valid}"
    )

    print(
        f"Alignment ratio   : "
        f"{100.0 * total_valid / max(total_samples, 1):.2f}%"
    )

    print()
    print("Methods:")

    for method, count in sorted(
        method_counter.items()
    ):

        print(
            f"  {method:15s}: "
            f"{count:8d} "
            f"({100.0 * count / max(total_samples, 1):6.2f}%)"
        )

    if valid_ages:

        print()
        print(
            f"Pose age median : "
            f"{np.median(valid_ages):.2f} ms"
        )

        print(
            f"Pose age p95    : "
            f"{np.percentile(valid_ages, 95):.2f} ms"
        )

        print(
            f"Pose age max    : "
            f"{np.max(valid_ages):.2f} ms"
        )

    print()
    print(
        f"Output directory:\n{output_dir}"
    )


if __name__ == "__main__":
    main()