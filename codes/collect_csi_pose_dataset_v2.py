#!/usr/bin/env python3

import cv2
import json
import math
import time
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import serial
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

# -------------------------
# Serial / CSI
# -------------------------
COM_PORT = "COM9"
BAUD_RATE = 921600

EXPECTED_SUBCARRIERS = 128

# Maximum number of CSI packets kept in RAM
CSI_BUFFER_SIZE = 10000

# -------------------------
# Camera
# -------------------------
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

YOLO_MODEL = "yolov8n-pose.pt"

# -------------------------
# Synchronization
# -------------------------

# A CSI sample is allowed to use a camera pose only if
# the pose timestamp is within this distance.
MAX_POSE_AGE_MS = 150.0

# Camera pose history.
POSE_BUFFER_SIZE = 300

# -------------------------
# Dataset
# -------------------------

OUTPUT_ROOT = Path("wifi_pose_dataset")

# Save every N CSI samples.
CHUNK_SIZE = 5000

# Only accept exactly 128 subcarriers.
DROP_WRONG_CSI_LENGTH = True

# -------------------------
# Visualization
# -------------------------

SHOW_CAMERA = True

# P = pause/resume recording
# Q / ESC = quit


# ============================================================
# YOLO COCO 17 KEYPOINTS
# ============================================================

SKELETON_EDGES = [
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


# ============================================================
# GLOBAL STATE
# ============================================================

stop_event = threading.Event()

pause_event = threading.Event()

csi_lock = threading.Lock()

pose_lock = threading.Lock()


# CSI packets waiting to be written.
#
# Each item:
#
# {
#     "arrival_time": float,
#     "timestamp": int,
#     "rssi": int,
#     "channel": int,
#     "iq": np.ndarray shape (256,)
# }
#
csi_buffer = deque(maxlen=CSI_BUFFER_SIZE)


# Camera pose history.
#
# Each item:
#
# {
#     "camera_time": float,
#     "pose": np.ndarray shape (17,2),
#     "confidence": np.ndarray shape (17,)
# }
#
pose_buffer = deque(maxlen=POSE_BUFFER_SIZE)


# ============================================================
# CSI PARSER
# ============================================================

def parse_csi_line(line):
    """
    Parse CSI serial line.

    Expected format:

        timestamp,rssi,channel,data_len,I,Q,I,Q,...

    Returns dict or None.
    """

    try:

        line = line.strip()

        if not line:
            return None

        parts = line.split(",")

        if len(parts) < 5:
            return None

        timestamp = int(parts[0])
        rssi = int(parts[1])
        channel = int(parts[2])
        data_len = int(parts[3])

        if data_len <= 0:
            return None

        raw = parts[4:4 + data_len]

        if len(raw) < data_len:
            return None

        iq = []

        for value in raw:

            v = int(value)

            # int8 range
            v = max(-128, min(127, v))

            iq.append(v)

        # Must contain complete I/Q pairs.
        if len(iq) % 2 != 0:
            iq = iq[:-1]

        if len(iq) == 0:
            return None

        iq = np.asarray(iq, dtype=np.int16)

        n_subcarriers = len(iq) // 2

        return {
            "timestamp": timestamp,
            "rssi": rssi,
            "channel": channel,
            "iq": iq,
            "n_subcarriers": n_subcarriers,
        }

    except Exception:
        return None


# ============================================================
# IQ -> AMPLITUDE
# ============================================================

def iq_to_amplitude(iq):
    """
    Convert:

        [I0,Q0,I1,Q1,...]

    to:

        [A0,A1,...]

    """

    iq = np.asarray(iq)

    I = iq[0::2].astype(np.float32)
    Q = iq[1::2].astype(np.float32)

    amplitude = np.sqrt(I * I + Q * Q)

    return amplitude.astype(np.float32)


# ============================================================
# CSI READER THREAD
# ============================================================

class CSIReader(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)

        self.packet_count = 0
        self.valid_count = 0
        self.bad_count = 0

        self.start_time = None

        self.serial = None

    def run(self):

        print()
        print("=" * 80)
        print("CSI READER")
        print("=" * 80)

        try:

            self.serial = serial.Serial(
                COM_PORT,
                BAUD_RATE,
                timeout=0.05,
            )

            print(f"[CSI] Connected to {COM_PORT}")
            print(f"[CSI] Baudrate: {BAUD_RATE}")

        except Exception as e:

            print(f"[CSI] ERROR opening serial: {e}")

            stop_event.set()

            return

        self.start_time = time.monotonic()

        while not stop_event.is_set():

            try:

                raw_line = self.serial.readline()

                if not raw_line:
                    continue

                self.packet_count += 1

                try:
                    line = raw_line.decode(
                        "utf-8",
                        errors="ignore"
                    )
                except Exception:
                    self.bad_count += 1
                    continue

                packet = parse_csi_line(line)

                if packet is None:

                    self.bad_count += 1

                    continue

                n_subcarriers = packet["n_subcarriers"]

                if (
                    DROP_WRONG_CSI_LENGTH
                    and n_subcarriers != EXPECTED_SUBCARRIERS
                ):

                    self.bad_count += 1

                    continue

                packet["arrival_time"] = time.monotonic()

                with csi_lock:

                    csi_buffer.append(packet)

                self.valid_count += 1

            except Exception as e:

                print(f"[CSI] Read error: {e}")

                time.sleep(0.01)

        if self.serial is not None:

            try:
                self.serial.close()
            except Exception:
                pass

        print("[CSI] Reader stopped.")

    def stats(self):

        elapsed = time.monotonic() - self.start_time \
            if self.start_time else 0

        rate = (
            self.valid_count / elapsed
            if elapsed > 0
            else 0
        )

        return {
            "total_lines": self.packet_count,
            "valid_packets": self.valid_count,
            "bad_packets": self.bad_count,
            "elapsed_sec": elapsed,
            "rate_hz": rate,
        }


# ============================================================
# POSE BUFFER
# ============================================================

def add_pose(camera_time, pose, confidence):

    with pose_lock:

        pose_buffer.append(
            {
                "camera_time": camera_time,
                "pose": pose.copy(),
                "confidence": confidence.copy(),
            }
        )


def find_nearest_pose(target_time):

    with pose_lock:

        if not pose_buffer:
            return None

        best = None
        best_dt = float("inf")

        # Search newest -> oldest.
        for item in reversed(pose_buffer):

            dt = abs(
                item["camera_time"] - target_time
            )

            if dt < best_dt:

                best_dt = dt
                best = item

            # Since buffer is chronological,
            # once timestamps are sufficiently old
            # there is no reason to continue.
            if (
                item["camera_time"] < target_time
                and dt > best_dt
            ):
                break

        if best is None:
            return None

        age_ms = best_dt * 1000.0

        if age_ms > MAX_POSE_AGE_MS:

            return None

        return {
            "pose": best["pose"].copy(),
            "confidence": best["confidence"].copy(),
            "age_ms": age_ms,
            "camera_time": best["camera_time"],
        }


# ============================================================
# DATASET WRITER
# ============================================================

class DatasetWriter:

    def __init__(self, session_dir):

        self.session_dir = Path(session_dir)

        self.chunks_dir = (
            self.session_dir / "chunks"
        )

        self.chunks_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.csi_amplitude = []
        self.iq = []

        self.pose = []
        self.pose_confidence = []

        self.timestamp = []
        self.rssi = []
        self.channel = []

        self.csi_arrival_time = []
        self.camera_time = []

        self.pose_age_ms = []

        self.chunk_id = 0

        self.total_saved = 0
        self.total_unlabeled = 0

        self.first_csi_time = None
        self.last_csi_time = None

        self.first_camera_time = None
        self.last_camera_time = None

    # --------------------------------------------------------
    # ADD SAMPLE
    # --------------------------------------------------------

    def add(
        self,
        packet,
        pose_info,
    ):

        amplitude = iq_to_amplitude(
            packet["iq"]
        )

        if len(amplitude) != EXPECTED_SUBCARRIERS:
            return False

        self.csi_amplitude.append(
            amplitude
        )

        self.iq.append(
            packet["iq"].copy()
        )

        self.timestamp.append(
            packet["timestamp"]
        )

        self.rssi.append(
            packet["rssi"]
        )

        self.channel.append(
            packet["channel"]
        )

        self.csi_arrival_time.append(
            packet["arrival_time"]
        )

        if pose_info is None:

            # No valid synchronized pose.
            #
            # We do NOT want to silently throw away CSI.
            # Instead mark the pose as NaN.
            #
            self.pose.append(
                np.full(
                    (17, 2),
                    np.nan,
                    dtype=np.float32
                )
            )

            self.pose_confidence.append(
                np.full(
                    17,
                    np.nan,
                    dtype=np.float32
                )
            )

            self.camera_time.append(
                np.nan
            )

            self.pose_age_ms.append(
                np.nan
            )

            self.total_unlabeled += 1

        else:

            self.pose.append(
                pose_info["pose"]
            )

            self.pose_confidence.append(
                pose_info["confidence"]
            )

            self.camera_time.append(
                pose_info["camera_time"]
            )

            self.pose_age_ms.append(
                pose_info["age_ms"]
            )

            if self.first_camera_time is None:

                self.first_camera_time = (
                    pose_info["camera_time"]
                )

            self.last_camera_time = (
                pose_info["camera_time"]
            )

        if self.first_csi_time is None:

            self.first_csi_time = (
                packet["arrival_time"]
            )

        self.last_csi_time = (
            packet["arrival_time"]
        )

        self.total_saved += 1

        if len(self.csi_amplitude) >= CHUNK_SIZE:

            self.flush()

        return True

    # --------------------------------------------------------
    # FLUSH
    # --------------------------------------------------------

    def flush(self):

        if not self.csi_amplitude:
            return

        filename = (
            self.chunks_dir
            / f"chunk_{self.chunk_id:05d}.npz"
        )

        data = {

            "csi_amplitude":
                np.asarray(
                    self.csi_amplitude,
                    dtype=np.float32
                ),

            "iq":
                np.asarray(
                    self.iq,
                    dtype=np.int16
                ),

            "pose":
                np.asarray(
                    self.pose,
                    dtype=np.float32
                ),

            "pose_confidence":
                np.asarray(
                    self.pose_confidence,
                    dtype=np.float32
                ),

            "timestamp":
                np.asarray(
                    self.timestamp,
                    dtype=np.int64
                ),

            "rssi":
                np.asarray(
                    self.rssi,
                    dtype=np.int16
                ),

            "channel":
                np.asarray(
                    self.channel,
                    dtype=np.int16
                ),

            "csi_arrival_time":
                np.asarray(
                    self.csi_arrival_time,
                    dtype=np.float64
                ),

            "camera_time":
                np.asarray(
                    self.camera_time,
                    dtype=np.float64
                ),

            "pose_age_ms":
                np.asarray(
                    self.pose_age_ms,
                    dtype=np.float64
                ),
        }

        np.savez_compressed(
            filename,
            **data
        )

        print()
        print(
            f"[SAVE] {filename.name} "
            f"-> {len(self.csi_amplitude)} samples"
        )

        self.chunk_id += 1

        self.csi_amplitude.clear()
        self.iq.clear()

        self.pose.clear()
        self.pose_confidence.clear()

        self.timestamp.clear()
        self.rssi.clear()
        self.channel.clear()

        self.csi_arrival_time.clear()
        self.camera_time.clear()

        self.pose_age_ms.clear()

    # --------------------------------------------------------
    # FINALIZE
    # --------------------------------------------------------

    def close(self):

        self.flush()

        elapsed = 0

        if (
            self.first_csi_time is not None
            and self.last_csi_time is not None
        ):

            elapsed = (
                self.last_csi_time
                - self.first_csi_time
            )

        summary = {

            "total_saved_samples":
                self.total_saved,

            "total_unlabeled_csi":
                self.total_unlabeled,

            "chunks":
                self.chunk_id,

            "duration_sec":
                elapsed,

            "effective_csi_rate_hz":
                (
                    self.total_saved / elapsed
                    if elapsed > 0
                    else 0
                ),

            "expected_subcarriers":
                EXPECTED_SUBCARRIERS,

            "camera_index":
                CAMERA_INDEX,

            "camera_backend":
                "CAP_DSHOW",

            "serial_port":
                COM_PORT,

            "baud_rate":
                BAUD_RATE,

            "max_pose_age_ms":
                MAX_POSE_AGE_MS,

        }

        with open(
            self.session_dir / "summary.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

        return summary


# ============================================================
# DRAW POSE
# ============================================================

def draw_pose(
    frame,
    keypoints,
    confidence,
):

    h, w = frame.shape[:2]

    # Draw skeleton
    for a, b in SKELETON_EDGES:

        if (
            confidence[a] < 0.2
            or confidence[b] < 0.2
        ):
            continue

        xa = int(keypoints[a, 0] * w)
        ya = int(keypoints[a, 1] * h)

        xb = int(keypoints[b, 0] * w)
        yb = int(keypoints[b, 1] * h)

        cv2.line(
            frame,
            (xa, ya),
            (xb, yb),
            (0, 255, 0),
            2
        )

    # Draw keypoints
    for i in range(17):

        if confidence[i] < 0.2:
            continue

        x = int(keypoints[i, 0] * w)
        y = int(keypoints[i, 1] * h)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# ============================================================
# CAMERA / YOLO
# ============================================================

def process_camera(
    model,
    writer,
):

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera index "
            f"{CAMERA_INDEX}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print()
    print("=" * 80)
    print("CAMERA")
    print("=" * 80)

    print(
        f"[CAM] index      = {CAMERA_INDEX}"
    )

    print(
        f"[CAM] resolution  = "
        f"{actual_width}x{actual_height}"
    )

    print(
        f"[CAM] reported FPS = "
        f"{actual_fps:.2f}"
    )

    print()

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:

            print(
                "[CAM] Failed to read frame."
            )

            time.sleep(0.01)

            continue

        # Detect black frame.
        if (
            frame.max() == 0
            or float(frame.std()) < 1.0
        ):

            print(
                "[CAM] WARNING: black frame"
            )

            if SHOW_CAMERA:

                cv2.imshow(
                    "CSI Pose Collector",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    stop_event.set()
                    break

            continue

        camera_time = time.monotonic()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            frame,
            verbose=False
        )

        best_pose = None
        best_conf = None

        if results:

            result = results[0]

            if (
                result.keypoints is not None
                and len(result.keypoints) > 0
            ):

                # Select person with highest mean confidence.
                best_idx = None
                best_score = -1

                for i in range(
                    len(result.keypoints)
                ):

                    kp_conf = (
                        result.keypoints.conf[i]
                        .cpu()
                        .numpy()
                    )

                    score = float(
                        np.nanmean(kp_conf)
                    )

                    if score > best_score:

                        best_score = score
                        best_idx = i

                if best_idx is not None:

                    xy = (
                        result.keypoints.xy[
                            best_idx
                        ]
                        .cpu()
                        .numpy()
                    )

                    conf = (
                        result.keypoints.conf[
                            best_idx
                        ]
                        .cpu()
                        .numpy()
                    )

                    # Normalize to [0,1].
                    pose = np.zeros(
                        (17, 2),
                        dtype=np.float32
                    )

                    pose[:, 0] = (
                        xy[:, 0] / frame.shape[1]
                    )

                    pose[:, 1] = (
                        xy[:, 1] / frame.shape[0]
                    )

                    pose = np.clip(
                        pose,
                        0.0,
                        1.0
                    )

                    best_pose = pose
                    best_conf = conf.astype(
                        np.float32
                    )

                    add_pose(
                        camera_time,
                        best_pose,
                        best_conf
                    )

                    if SHOW_CAMERA:

                        draw_pose(
                            frame,
                            best_pose,
                            best_conf
                        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT collect CSI here.
        #
        # CSI is already continuously collected
        # by CSIReader.
        # ----------------------------------------------------

        if SHOW_CAMERA:

            # Display status.
            with csi_lock:
                csi_queue_size = len(csi_buffer)

            text = (
                f"CSI buffer: "
                f"{csi_queue_size}"
            )

            cv2.putText(
                frame,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "P: Pause/Resume   Q: Quit",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "CSI Pose Collector",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):

                if pause_event.is_set():

                    pause_event.clear()

                    print(
                        "[CONTROL] Recording RESUMED"
                    )

                else:

                    pause_event.set()

                    print(
                        "[CONTROL] Recording PAUSED"
                    )

            elif key in (ord("q"), 27):

                stop_event.set()

                break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# CONTINUOUS CSI DRAINER
# ============================================================

def continuous_writer_loop(writer):

    """
    This is the critical part.

    CSI is drained independently from camera/YOLO.

    Every CSI packet is processed immediately.

    For each CSI timestamp:

        CSI time
            ↓
        nearest camera pose
            ↓
        save

    """

    print()
    print("=" * 80)
    print("CONTINUOUS CSI WRITER")
    print("=" * 80)

    last_report = time.monotonic()

    last_saved_count = 0

    while not stop_event.is_set():

        packet = None

        with csi_lock:

            if csi_buffer:

                # FIFO:
                # oldest CSI packet first.
                packet = csi_buffer.popleft()

        if packet is None:

            time.sleep(0.001)

            continue

        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        if pause_event.is_set():

            # While paused, we intentionally discard
            # CSI packets from the live buffer.
            #
            # This prevents RAM from growing indefinitely.
            continue

        # ----------------------------------------------------
        # FIND CAMERA POSE
        # ----------------------------------------------------

        pose_info = find_nearest_pose(
            packet["arrival_time"]
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        writer.add(
            packet,
            pose_info
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        now = time.monotonic()

        if now - last_report >= 2.0:

            stats_saved = (
                writer.total_saved
                - last_saved_count
            )

            print(
                f"[DATA] saved={writer.total_saved:7d} "
                f"rate={stats_saved / (now-last_report):7.2f} Hz "
                f"unlabeled={writer.total_unlabeled:6d} "
                f"pose_buffer={len(pose_buffer):4d}"
            )

            last_saved_count = writer.total_saved
            last_report = now


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("WiFi CSI + CAMERA POSE DATASET COLLECTOR V2")
    print("=" * 80)

    print()
    print("Architecture:")
    print()
    print("    ESP32 CSI")
    print("       │")
    print("       ▼")
    print("   CSI Reader Thread")
    print("       │")
    print("       ▼")
    print("   CSI Ring Buffer")
    print("       │")
    print("       ▼")
    print(" Continuous Writer")
    print("       ▲")
    print("       │ nearest pose")
    print("       │")
    print(" Camera + YOLO")
    print()
    print("=" * 80)

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session_name = (
        "session_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    session_dir = (
        OUTPUT_ROOT / session_name
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"[DATASET] {session_dir}"
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "created_at":
            datetime.now().isoformat(),

        "serial": {
            "port": COM_PORT,
            "baud_rate": BAUD_RATE,
        },

        "camera": {
            "index": CAMERA_INDEX,
            "backend": "CAP_DSHOW",
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "requested_fps": CAMERA_FPS,
        },

        "yolo": {
            "model": YOLO_MODEL,
            "keypoints": 17,
            "format": "COCO17",
        },

        "csi": {
            "subcarriers":
                EXPECTED_SUBCARRIERS,

            "representation":
                "amplitude",

            "raw_iq": True,
        },

        "synchronization": {

            "method":
                "nearest_camera_pose",

            "max_pose_age_ms":
                MAX_POSE_AGE_MS,

            "clock":
                "time.monotonic",
        },

        "collector_version":
            "V2-continuous",

    }

    with open(
        session_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print()
    print(
        f"[YOLO] Loading {YOLO_MODEL} ..."
    )

    model = YOLO(
        YOLO_MODEL
    )

    print(
        "[YOLO] Model loaded."
    )

    # --------------------------------------------------------
    # WRITER
    # --------------------------------------------------------

    writer = DatasetWriter(
        session_dir
    )

    # --------------------------------------------------------
    # CSI THREAD
    # --------------------------------------------------------

    csi_reader = CSIReader()

    csi_reader.start()

    # --------------------------------------------------------
    # WRITER THREAD
    # --------------------------------------------------------

    writer_thread = threading.Thread(
        target=continuous_writer_loop,
        args=(writer,),
        daemon=True
    )

    writer_thread.start()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    try:

        process_camera(
            model,
            writer
        )

    except KeyboardInterrupt:

        print()
        print(
            "[MAIN] Ctrl+C received."
        )

    except Exception as e:

        print()
        print(
            f"[MAIN] ERROR: {e}"
        )

    finally:

        stop_event.set()

        print()
        print(
            "[MAIN] Stopping..."
        )

        csi_reader.join(
            timeout=3.0
        )

        writer_thread.join(
            timeout=3.0
        )

        summary = writer.close()

        # ----------------------------------------------------
        # CSI STATS
        # ----------------------------------------------------

        csi_stats = csi_reader.stats()

        summary["csi_reader"] = csi_stats

        with open(
            session_dir / "summary.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

        print()
        print("=" * 80)
        print("COLLECTION FINISHED")
        print("=" * 80)

        print()
        print(
            f"Dataset      : {session_dir}"
        )

        print(
            f"Samples      : "
            f"{summary['total_saved_samples']}"
        )

        print(
            f"Duration     : "
            f"{summary['duration_sec']:.2f} sec"
        )

        print(
            f"Effective CSI rate : "
            f"{summary['effective_csi_rate_hz']:.2f} Hz"
        )

        print(
            f"Unlabeled CSI: "
            f"{summary['total_unlabeled_csi']}"
        )

        print()
        print(
            "Run dataset_qa.py on this session."
        )

        print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()