import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("wifi_pose_dataset")

CHUNK_FILES = sorted(
    DATASET_DIR.rglob("chunk_*.npz")
)

# How many samples to visualize
NUM_SAMPLES = 12

# Number of CSI samples for temporal plot
TEMPORAL_SAMPLES = 300

# ============================================================
# POSE SKELETON
# ============================================================

POSE_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


POSE_NAMES = [
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
# LOAD DATA
# ============================================================

def load_dataset():

    if not CHUNK_FILES:

        raise RuntimeError(
            "No chunk_*.npz files found."
        )

    print("=" * 70)
    print("FOUND CHUNKS")
    print("=" * 70)

    for f in CHUNK_FILES:
        print(f)

    print()

    arrays = {}

    keys = [
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

    for key in keys:
        arrays[key] = []

    total = 0

    for file in CHUNK_FILES:

        print(
            f"[LOAD] {file}"
        )

        data = np.load(
            file,
            allow_pickle=False,
        )

        print(
            f"       samples = "
            f"{len(data['pose'])}"
        )

        for key in keys:

            if key in data:
                arrays[key].append(
                    data[key]
                )

        total += len(data["pose"])

    result = {}

    for key in keys:

        if arrays[key]:

            result[key] = np.concatenate(
                arrays[key],
                axis=0,
            )

    print()
    print(
        f"[OK] Total samples: {total}"
    )

    return result


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(data):

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for key, value in data.items():

        print(
            f"{key:20s} "
            f"shape={str(value.shape):18s} "
            f"dtype={value.dtype}"
        )

    print()

    # --------------------------------------------------------
    # CSI
    # --------------------------------------------------------

    csi = data["csi_amplitude"]

    print("CSI")
    print("-" * 70)

    print(
        f"min       : {csi.min():.4f}"
    )

    print(
        f"max       : {csi.max():.4f}"
    )

    print(
        f"mean      : {csi.mean():.4f}"
    )

    print(
        f"std       : {csi.std():.4f}"
    )

    # --------------------------------------------------------
    # RSSI
    # --------------------------------------------------------

    rssi = data["rssi"]

    print()
    print("RSSI")
    print("-" * 70)

    print(
        f"min       : {rssi.min()}"
    )

    print(
        f"max       : {rssi.max()}"
    )

    print(
        f"mean      : {rssi.mean():.2f}"
    )

    print(
        f"std       : {rssi.std():.2f}"
    )

    # --------------------------------------------------------
    # POSE CONFIDENCE
    # --------------------------------------------------------

    conf = data["pose_confidence"]

    print()
    print("POSE CONFIDENCE")
    print("-" * 70)

    print(
        f"min       : {conf.min():.4f}"
    )

    print(
        f"max       : {conf.max():.4f}"
    )

    print(
        f"mean      : {conf.mean():.4f}"
    )

    print(
        f"median    : {np.median(conf):.4f}"
    )

    # --------------------------------------------------------
    # PAIR AGE
    # --------------------------------------------------------

    age = data["pose_age_ms"]

    print()
    print("CSI ↔ CAMERA SYNCHRONIZATION")
    print("-" * 70)

    print(
        f"min       : {age.min():.3f} ms"
    )

    print(
        f"max       : {age.max():.3f} ms"
    )

    print(
        f"mean      : {age.mean():.3f} ms"
    )

    print(
        f"median    : {np.median(age):.3f} ms"
    )

    for threshold in [
        10,
        20,
        50,
        100,
        200,
    ]:

        percentage = (
            np.mean(age <= threshold)
            * 100
        )

        print(
            f"<= {threshold:3d} ms : "
            f"{percentage:6.2f}%"
        )


# ============================================================
# PLOT POSE SAMPLES
# ============================================================

def plot_pose_samples(data):

    pose = data["pose"]

    conf = data["pose_confidence"]

    n = len(pose)

    count = min(
        NUM_SAMPLES,
        n,
    )

    # Pick evenly distributed samples
    indices = np.linspace(
        0,
        n - 1,
        count,
        dtype=int,
    )

    fig = plt.figure(
        figsize=(16, 10)
    )

    for plot_idx, idx in enumerate(indices):

        ax = fig.add_subplot(
            3,
            4,
            plot_idx + 1,
        )

        p = pose[idx]

        c = conf[idx]

        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        for a, b in POSE_EDGES:

            if (
                c[a] < 0.2
                or c[b] < 0.2
            ):
                continue

            ax.plot(
                [p[a, 0], p[b, 0]],
                [p[a, 1], p[b, 1]],
                linewidth=2,
            )

        # ----------------------------------------------------
        # Keypoints
        # ----------------------------------------------------

        valid = c >= 0.2

        ax.scatter(
            p[valid, 0],
            p[valid, 1],
            s=30,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

        ax.set_title(
            f"Sample {idx}"
        )

        ax.set_aspect(
            "equal"
        )

        ax.grid(
            alpha=0.2
        )

    plt.suptitle(
        "YOLO Pose Samples"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CSI SAMPLE PLOTS
# ============================================================

def plot_csi_samples(data):

    csi = data["csi_amplitude"]

    n = len(csi)

    count = min(
        6,
        n,
    )

    indices = np.linspace(
        0,
        n - 1,
        count,
        dtype=int,
    )

    fig = plt.figure(
        figsize=(14, 9)
    )

    for plot_idx, idx in enumerate(indices):

        ax = fig.add_subplot(
            3,
            2,
            plot_idx + 1,
        )

        ax.plot(
            csi[idx]
        )

        ax.set_title(
            f"CSI Sample {idx}"
        )

        ax.set_xlabel(
            "Subcarrier"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.grid(
            alpha=0.2
        )

    plt.suptitle(
        "CSI Amplitude Samples"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CSI HEATMAP
# ============================================================

def plot_csi_heatmap(data):

    csi = data["csi_amplitude"]

    n = min(
        TEMPORAL_SAMPLES,
        len(csi),
    )

    subset = csi[:n]

    plt.figure(
        figsize=(15, 7)
    )

    plt.imshow(
        subset.T,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="CSI amplitude"
    )

    plt.xlabel(
        "Time / sample"
    )

    plt.ylabel(
        "Subcarrier"
    )

    plt.title(
        f"CSI Temporal Heatmap "
        f"(first {n} samples)"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# POSE CENTER TRAJECTORY
# ============================================================

def plot_pose_center(data):

    pose = data["pose"]

    # Use shoulders + hips as body center.
    body_indices = [
        5,
        6,
        11,
        12,
    ]

    centers = np.mean(
        pose[:, body_indices, :],
        axis=1,
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        centers[:, 0],
        label="X",
    )

    plt.plot(
        centers[:, 1],
        label="Y",
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Normalized position"
    )

    plt.title(
        "Body Center Position Over Time"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# POSE MOTION
# ============================================================

def plot_pose_motion(data):

    pose = data["pose"]

    body_indices = [
        5,
        6,
        11,
        12,
    ]

    centers = np.mean(
        pose[:, body_indices, :],
        axis=1,
    )

    velocity = np.diff(
        centers,
        axis=0,
    )

    speed = np.linalg.norm(
        velocity,
        axis=1,
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        speed
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Frame-to-frame motion"
    )

    plt.title(
        "Camera Pose Motion Over Time"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CSI ENERGY
# ============================================================

def plot_csi_energy(data):

    csi = data["csi_amplitude"]

    energy = np.mean(
        csi ** 2,
        axis=1,
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        energy
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Mean CSI power"
    )

    plt.title(
        "CSI Energy Over Time"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CSI vs POSE MOTION
# ============================================================

def plot_csi_vs_motion(data):

    csi = data["csi_amplitude"]

    pose = data["pose"]

    # CSI energy
    csi_energy = np.mean(
        csi ** 2,
        axis=1,
    )

    # Body center
    body_indices = [
        5,
        6,
        11,
        12,
    ]

    centers = np.mean(
        pose[:, body_indices, :],
        axis=1,
    )

    velocity = np.diff(
        centers,
        axis=0,
    )

    speed = np.linalg.norm(
        velocity,
        axis=1,
    )

    # Align lengths
    csi_energy = csi_energy[1:]

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        speed,
        csi_energy,
        alpha=0.4,
        s=10,
    )

    plt.xlabel(
        "Camera pose motion"
    )

    plt.ylabel(
        "CSI energy"
    )

    plt.title(
        "CSI Energy vs Camera Motion"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# KEYPOINT CONFIDENCE
# ============================================================

def plot_keypoint_confidence(data):

    conf = data["pose_confidence"]

    mean_conf = np.mean(
        conf,
        axis=0,
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.bar(
        np.arange(17),
        mean_conf,
    )

    plt.xticks(
        np.arange(17),
        POSE_NAMES,
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Mean confidence"
    )

    plt.title(
        "Average YOLO Confidence per Keypoint"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# SYNCHRONIZATION
# ============================================================

def plot_sync(data):

    age = data["pose_age_ms"]

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        age,
        linewidth=0.8,
    )

    plt.axhline(
        20,
        linestyle="--",
        label="20 ms",
    )

    plt.axhline(
        50,
        linestyle="--",
        label="50 ms",
    )

    plt.axhline(
        100,
        linestyle="--",
        label="100 ms",
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "CSI ↔ Camera difference (ms)"
    )

    plt.title(
        "CSI / Camera Synchronization"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )
    print(
        "CSI + POSE DATASET EXPLORER"
    )
    print(
        "=" * 70
    )

    data = load_dataset()

    print_summary(
        data
    )

    print()
    print(
        "[1/8] Pose samples..."
    )

    plot_pose_samples(
        data
    )

    print(
        "[2/8] CSI samples..."
    )

    plot_csi_samples(
        data
    )

    print(
        "[3/8] CSI heatmap..."
    )

    plot_csi_heatmap(
        data
    )

    print(
        "[4/8] Body center..."
    )

    plot_pose_center(
        data
    )

    print(
        "[5/8] Pose motion..."
    )

    plot_pose_motion(
        data
    )

    print(
        "[6/8] CSI energy..."
    )

    plot_csi_energy(
        data
    )

    print(
        "[7/8] CSI vs pose motion..."
    )

    plot_csi_vs_motion(
        data
    )

    print(
        "[8/8] Confidence + synchronization..."
    )

    plot_keypoint_confidence(
        data
    )

    plot_sync(
        data
    )

    print()
    print(
        "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()