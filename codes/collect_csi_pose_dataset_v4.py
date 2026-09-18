#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WiFi CSI + Camera Pose Dataset Collector V4

Pipeline:

    ESP32 RX
       |
       | Serial COM9
       v
    CSI Reader Thread
       |
       +--------------------+
                            |
                            v
                     CSI timeline
                            |
                            |
    Webcam -> YOLO Pose ----+
               |
               v
         Pose timeline

Important:
    CSI and Pose are NOT online-matched.

Both timelines are saved independently using the same
local monotonic clock so that they can be aligned later.

V4 improvements:
    - Explicit person_present label
    - Explicit pose_valid_mask
    - Pose confidence preserved
    - Raw YOLO coordinates preserved as normalized x/y
    - CSI saved independently from camera speed
    - Empty-room samples are preserved
    - Robust camera handling: index 1 + DirectShow
    - No OpenCV GUI calls from background threads
    - Metadata contains synchronization information
"""

import os
import sys
import cv2
import json
import time
import math
import queue
import signal
import threading
import traceback

import numpy as np

from collections import deque
from datetime import datetime

try:
    import serial
except ImportError:
    print("[ERROR] pyserial is not installed.")
    print("Install with:")
    print("    pip install pyserial")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed.")
    print("Install with:")
    print("    pip install ultralytics")
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================

COM_PORT = "COM9"
BAUD_RATE = 921600

CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

YOLO_MODEL = "yolov8n-pose.pt"

# YOLO inference confidence
YOLO_CONF = 0.25

# Minimum confidence for considering a keypoint usable
KEYPOINT_CONF_THRESHOLD = 0.20

# Save CSI chunks
CSI_CHUNK_SIZE = 5000

# Maximum number of CSI packets kept in memory
CSI_BUFFER_MAXLEN = 50000

# Pose timeline flush interval
POSE_FLUSH_INTERVAL = 5.0

# Output root
DATASET_ROOT = "wifi_pose_dataset"

# Display camera
SHOW_CAMERA = True

# Print CSI statistics every N packets
CSI_PRINT_EVERY = 1000


# ============================================================
# GLOBAL STATE
# ============================================================

stop_event = threading.Event()

csi_buffer = deque(maxlen=CSI_BUFFER_MAXLEN)

csi_lock = threading.Lock()

stats_lock = threading.Lock()

stats = {
    "csi_packets": 0,
    "csi_bad_packets": 0,
    "pose_frames": 0,
    "person_frames": 0,
    "empty_frames": 0,
}


# ============================================================
# SIGNAL HANDLER
# ============================================================

def signal_handler(sig, frame):
    print()
    print("[INFO] Stop requested...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# DIRECTORY
# ============================================================

def create_session_dir():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = os.path.join(
        DATASET_ROOT,
        f"session_{timestamp}"
    )

    csi_dir = os.path.join(session_dir, "csi")
    pose_dir = os.path.join(session_dir, "pose")

    os.makedirs(csi_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    return session_dir, csi_dir, pose_dir


# ============================================================
# CSI PARSER
# ============================================================

def parse_csi_line(line):
    """
    Expected:

    timestamp,rssi,channel,data_len,I,Q,I,Q,...

    Returns:
        dict or None
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

        if len(raw) != data_len:
            return None

        iq = []

        for x in raw:

            value = int(x)

            # ESP32 CSI values are int8-like.
            value = max(-128, min(127, value))

            iq.append(value)

        # I/Q must come in pairs.
        if len(iq) < 2:
            return None

        if len(iq) % 2 != 0:
            iq = iq[:-1]

        iq = np.asarray(iq, dtype=np.int16)

        return {
            "timestamp": timestamp,
            "rssi": rssi,
            "channel": channel,
            "iq": iq,
        }

    except Exception:
        return None


# ============================================================
# IQ -> AMPLITUDE
# ============================================================

def iq_to_amplitude(iq):
    """
    [I,Q,I,Q,...]

    ->

    [sqrt(I^2 + Q^2), ...]
    """

    iq = np.asarray(iq)

    if len(iq) < 2:
        return np.empty(0, dtype=np.float32)

    i = iq[0::2].astype(np.float32)
    q = iq[1::2].astype(np.float32)

    amplitude = np.sqrt(i * i + q * q)

    return amplitude.astype(np.float32)


# ============================================================
# CSI READER THREAD
# ============================================================

class CSIReader(threading.Thread):

    def __init__(self):

        super().__init__(
            daemon=True,
            name="CSIReader"
        )

        self.ser = None

        self.local_count = 0

    def open_serial(self):

        print()
        print("=" * 70)
        print("Opening CSI Serial")
        print("=" * 70)

        print(f"Port     : {COM_PORT}")
        print(f"Baudrate : {BAUD_RATE}")

        self.ser = serial.Serial(
            COM_PORT,
            BAUD_RATE,
            timeout=1
        )

        print("[OK] Serial opened")

    def run(self):

        try:

            self.open_serial()

        except Exception as e:

            print()
            print("[ERROR] Cannot open serial port:")
            print(e)

            stop_event.set()
            return

        while not stop_event.is_set():

            try:

                line = self.ser.readline()

                if not line:
                    continue

                try:
                    line = line.decode(
                        "utf-8",
                        errors="ignore"
                    )
                except Exception:
                    continue

                parsed = parse_csi_line(line)

                if parsed is None:

                    with stats_lock:
                        stats["csi_bad_packets"] += 1

                    continue

                iq = parsed["iq"]

                amplitude = iq_to_amplitude(iq)

                if len(amplitude) != 128:

                    with stats_lock:
                        stats["csi_bad_packets"] += 1

                    continue

                arrival_time = time.monotonic()

                packet = {
                    "timestamp": parsed["timestamp"],
                    "rssi": parsed["rssi"],
                    "channel": parsed["channel"],

                    "iq": iq,
                    "csi_amplitude": amplitude,

                    # IMPORTANT:
                    # same clock used by camera
                    "arrival_time": arrival_time,
                }

                with csi_lock:

                    csi_buffer.append(packet)

                self.local_count += 1

                with stats_lock:
                    stats["csi_packets"] += 1

                if self.local_count % CSI_PRINT_EVERY == 0:

                    print(
                        f"[CSI] packets={self.local_count}"
                    )

            except serial.SerialException as e:

                print()
                print("[ERROR] Serial error:")
                print(e)

                stop_event.set()
                break

            except Exception:

                traceback.print_exc()

        try:

            if self.ser is not None:
                self.ser.close()

        except Exception:
            pass

        print("[CSI] Reader stopped.")


# ============================================================
# POSE EXTRACTION
# ============================================================

def extract_pose(result):

    """
    Returns:

        person_present
        pose
        confidence
        valid_mask
        bbox

    pose:
        [17,2]

    confidence:
        [17]

    valid_mask:
        [17]

    Coordinates are normalized [0,1].
    """

    empty_pose = np.full(
        (17, 2),
        np.nan,
        dtype=np.float32
    )

    empty_conf = np.full(
        (17,),
        np.nan,
        dtype=np.float32
    )

    empty_mask = np.zeros(
        (17,),
        dtype=np.uint8
    )

    empty_bbox = np.full(
        (4,),
        np.nan,
        dtype=np.float32
    )

    try:

        if result is None:
            return (
                False,
                empty_pose,
                empty_conf,
                empty_mask,
                empty_bbox
            )

        if result.keypoints is None:
            return (
                False,
                empty_pose,
                empty_conf,
                empty_mask,
                empty_bbox
            )

        if len(result.keypoints) == 0:
            return (
                False,
                empty_pose,
                empty_conf,
                empty_mask,
                empty_bbox
            )

        # ----------------------------------------------------
        # Select best person
        # ----------------------------------------------------

        if result.boxes is not None and len(result.boxes) > 0:

            try:

                box_conf = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                best_idx = int(
                    np.argmax(box_conf)
                )

            except Exception:

                best_idx = 0

        else:

            best_idx = 0

        # ----------------------------------------------------
        # Keypoints
        # ----------------------------------------------------

        kp_xy = (
            result.keypoints.xyn[best_idx]
            .detach()
            .cpu()
            .numpy()
        )

        kp_xy = np.asarray(
            kp_xy,
            dtype=np.float32
        )

        # Ensure exactly 17
        if kp_xy.shape[0] != 17:

            return (
                False,
                empty_pose,
                empty_conf,
                empty_mask,
                empty_bbox
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        try:

            kp_data = (
                result.keypoints.data[best_idx]
                .detach()
                .cpu()
                .numpy()
            )

            if kp_data.shape[1] >= 3:

                confidence = kp_data[:, 2].astype(
                    np.float32
                )

            else:

                confidence = np.ones(
                    (17,),
                    dtype=np.float32
                )

        except Exception:

            confidence = np.ones(
                (17,),
                dtype=np.float32
            )

        # ----------------------------------------------------
        # Valid mask
        # ----------------------------------------------------

        valid_mask = (
            confidence >= KEYPOINT_CONF_THRESHOLD
        ).astype(np.uint8)

        pose = kp_xy.copy()

        # Invalid points become NaN
        pose[valid_mask == 0] = np.nan

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox = empty_bbox.copy()

        if result.boxes is not None:

            try:

                xyxy = (
                    result.boxes.xyxy[best_idx]
                    .detach()
                    .cpu()
                    .numpy()
                )

                frame_h, frame_w = result.orig_shape

                x1, y1, x2, y2 = xyxy

                bbox[:] = [
                    x1 / frame_w,
                    y1 / frame_h,
                    x2 / frame_w,
                    y2 / frame_h
                ]

            except Exception:
                pass

        return (
            True,
            pose,
            confidence,
            valid_mask,
            bbox
        )

    except Exception:

        traceback.print_exc()

        return (
            False,
            empty_pose,
            empty_conf,
            empty_mask,
            empty_bbox
        )


# ============================================================
# SAVE CSI CHUNK
# ============================================================

def save_csi_chunk(
    csi_dir,
    chunk_id,
    packets
):

    if not packets:
        return None

    amplitudes = np.stack([
        p["csi_amplitude"]
        for p in packets
    ]).astype(np.float32)

    iq = np.stack([
        p["iq"]
        for p in packets
    ]).astype(np.int16)

    timestamp = np.asarray([
        p["timestamp"]
        for p in packets
    ], dtype=np.int64)

    rssi = np.asarray([
        p["rssi"]
        for p in packets
    ], dtype=np.int16)

    channel = np.asarray([
        p["channel"]
        for p in packets
    ], dtype=np.int16)

    arrival_time = np.asarray([
        p["arrival_time"]
        for p in packets
    ], dtype=np.float64)

    path = os.path.join(
        csi_dir,
        f"chunk_{chunk_id:05d}.npz"
    )

    np.savez_compressed(
        path,

        csi_amplitude=amplitudes,
        iq=iq,

        timestamp=timestamp,
        rssi=rssi,
        channel=channel,

        csi_time=arrival_time,
    )

    return path


# ============================================================
# SAVE POSE TIMELINE
# ============================================================

def save_pose_timeline(
    pose_dir,
    poses
):

    if not poses:

        print("[WARN] No pose frames collected.")

        return None

    pose = np.stack([
        x["pose"]
        for x in poses
    ]).astype(np.float32)

    confidence = np.stack([
        x["confidence"]
        for x in poses
    ]).astype(np.float32)

    valid_mask = np.stack([
        x["valid_mask"]
        for x in poses
    ]).astype(np.uint8)

    person_present = np.asarray([
        x["person_present"]
        for x in poses
    ], dtype=np.uint8)

    bbox = np.stack([
        x["bbox"]
        for x in poses
    ]).astype(np.float32)

    camera_time = np.asarray([
        x["camera_time"]
        for x in poses
    ], dtype=np.float64)

    frame_id = np.asarray([
        x["frame_id"]
        for x in poses
    ], dtype=np.int64)

    path = os.path.join(
        pose_dir,
        "pose_timeline.npz"
    )

    np.savez_compressed(
        path,

        pose=pose,

        pose_confidence=confidence,

        pose_valid_mask=valid_mask,

        person_present=person_present,

        bbox=bbox,

        camera_time=camera_time,

        frame_id=frame_id,
    )

    return path


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    session_dir,
    start_time,
    end_time
):

    duration = end_time - start_time

    with stats_lock:

        s = dict(stats)

    csi_rate = (
        s["csi_packets"] / duration
        if duration > 0
        else 0
    )

    person_ratio = (
        s["person_frames"] / s["pose_frames"]
        if s["pose_frames"] > 0
        else 0
    )

    empty_ratio = (
        s["empty_frames"] / s["pose_frames"]
        if s["pose_frames"] > 0
        else 0
    )

    metadata = {

        "dataset_version": "V4",

        "created_at": datetime.now().isoformat(),

        "session_start_monotonic": start_time,

        "session_end_monotonic": end_time,

        "duration_seconds": duration,

        "camera": {

            "index": CAMERA_INDEX,

            "backend": "CAP_DSHOW",

            "width": CAMERA_WIDTH,

            "height": CAMERA_HEIGHT,

            "requested_fps": CAMERA_FPS,

            "yolo_model": YOLO_MODEL,

            "yolo_conf": YOLO_CONF,

            "keypoint_conf_threshold":
                KEYPOINT_CONF_THRESHOLD,
        },

        "csi": {

            "serial_port": COM_PORT,

            "baud_rate": BAUD_RATE,

            "n_subcarriers": 128,

            "iq_values_per_sample": 256,

            "chunk_size": CSI_CHUNK_SIZE,
        },

        "synchronization": {

            "clock": "time.monotonic",

            "online_matching": False,

            "alignment": (
                "performed_offline"
            ),
        },

        "statistics": {

            "csi_packets": s["csi_packets"],

            "csi_bad_packets":
                s["csi_bad_packets"],

            "pose_frames":
                s["pose_frames"],

            "person_frames":
                s["person_frames"],

            "empty_frames":
                s["empty_frames"],

            "csi_rate_hz":
                csi_rate,

            "person_detection_ratio":
                person_ratio,

            "empty_room_ratio":
                empty_ratio,
        },

        "labels": {

            "person_present": {

                "0": "empty_room",

                "1": "person_present",
            },

            "pose": {

                "format": "17x2",

                "coordinates":
                    "normalized_xy",

                "invalid_keypoints":
                    "NaN",

                "valid_mask":
                    "pose_valid_mask",
            },
        },

        "keypoints": [

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
        ],
    }

    path = os.path.join(
        session_dir,
        "metadata.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("WiFi CSI + Camera Pose Dataset Collector V4")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    (
        session_dir,
        csi_dir,
        pose_dir
    ) = create_session_dir()

    print(f"Session : {session_dir}")
    print()

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("[INFO] Loading YOLO pose model...")

    try:

        model = YOLO(YOLO_MODEL)

    except Exception as e:

        print("[ERROR] Cannot load YOLO:")
        print(e)

        return

    print("[OK] YOLO loaded")
    print()

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print("=" * 80)
    print("Opening Camera")
    print("=" * 80)

    print(f"Index   : {CAMERA_INDEX}")
    print("Backend : DirectShow")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND
    )

    if not cap.isOpened():

        print()
        print("[ERROR] Camera could not be opened.")

        print()
        print("Known working camera:")
        print("    index=1")
        print("    backend=CAP_DSHOW")

        return

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

    # --------------------------------------------------------
    # Verify first frame
    # --------------------------------------------------------

    ok, frame = cap.read()

    if not ok or frame is None:

        print("[ERROR] Camera opened but frame read failed.")

        cap.release()

        return

    print(
        f"[OK] Camera frame: "
        f"{frame.shape}"
    )

    print()

    # --------------------------------------------------------
    # Start CSI reader
    # --------------------------------------------------------

    csi_reader = CSIReader()

    csi_reader.start()

    # --------------------------------------------------------
    # Recording state
    # --------------------------------------------------------

    session_start = time.monotonic()

    last_pose_flush = session_start

    pose_records = []

    csi_chunk = []

    chunk_id = 0

    frame_id = 0

    last_stats_print = session_start

    print("=" * 80)
    print("RECORDING")
    print("=" * 80)

    print()
    print("Controls:")
    print("    Q = stop")
    print("    ESC = stop")
    print()
    print("IMPORTANT:")
    print("    Empty room is intentionally recorded.")
    print("    Person present = 1")
    print("    Empty room    = 0")
    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while not stop_event.is_set():

            loop_start = time.monotonic()

            # ------------------------------------------------
            # Drain CSI buffer
            # ------------------------------------------------

            with csi_lock:

                while csi_buffer:

                    packet = csi_buffer.popleft()

                    csi_chunk.append(packet)

            # ------------------------------------------------
            # Save CSI chunk
            # ------------------------------------------------

            if len(csi_chunk) >= CSI_CHUNK_SIZE:

                chunk_to_save = csi_chunk[:CSI_CHUNK_SIZE]

                del csi_chunk[:CSI_CHUNK_SIZE]

                path = save_csi_chunk(
                    csi_dir,
                    chunk_id,
                    chunk_to_save
                )

                if path:

                    print(
                        f"[SAVE CSI] "
                        f"chunk={chunk_id:05d} "
                        f"samples={len(chunk_to_save)}"
                    )

                    chunk_id += 1

            # ------------------------------------------------
            # Camera
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:

                print("[WARN] Camera read failed.")

                time.sleep(0.01)

                continue

            camera_time = time.monotonic()

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            try:

                results = model.predict(
                    frame,
                    conf=YOLO_CONF,
                    verbose=False
                )

                result = (
                    results[0]
                    if results
                    else None
                )

            except Exception:

                traceback.print_exc()

                result = None

            # ------------------------------------------------
            # Extract pose
            # ------------------------------------------------

            (
                person_present,
                pose,
                confidence,
                valid_mask,
                bbox
            ) = extract_pose(result)

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            with stats_lock:

                stats["pose_frames"] += 1

                if person_present:

                    stats["person_frames"] += 1

                else:

                    stats["empty_frames"] += 1

            # ------------------------------------------------
            # Save pose record
            # ------------------------------------------------

            pose_records.append({

                "pose": pose,

                "confidence": confidence,

                "valid_mask": valid_mask,

                "person_present":
                    int(person_present),

                "bbox": bbox,

                "camera_time":
                    camera_time,

                "frame_id":
                    frame_id,
            })

            frame_id += 1

            # ------------------------------------------------
            # Draw visualization
            # ------------------------------------------------

            if SHOW_CAMERA:

                display = frame.copy()

                if person_present:

                    try:

                        result.plot(
                            img=display,
                            boxes=True,
                            labels=False,
                            conf=False
                        )

                    except Exception:
                        pass

                    text = "PERSON"

                else:

                    text = "EMPTY ROOM"

                cv2.putText(
                    display,
                    text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0)
                    if person_present
                    else (0, 0, 255),
                    2
                )

                with stats_lock:

                    csi_count = stats["csi_packets"]

                    pose_count = stats["pose_frames"]

                    person_count = stats["person_frames"]

                cv2.putText(
                    display,
                    f"CSI: {csi_count}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Pose: {pose_count}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Person: {person_count}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "CSI + Pose Collector V4",
                    display
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27
                ):

                    stop_event.set()

                    break

            # ------------------------------------------------
            # Periodic statistics
            # ------------------------------------------------

            now = time.monotonic()

            if now - last_stats_print >= 10.0:

                elapsed = (
                    now - session_start
                )

                with stats_lock:

                    csi_count = stats["csi_packets"]

                    pose_count = stats["pose_frames"]

                    person_count = stats["person_frames"]

                    empty_count = stats["empty_frames"]

                    bad = stats["csi_bad_packets"]

                csi_rate = (
                    csi_count / elapsed
                    if elapsed > 0
                    else 0
                )

                person_ratio = (
                    person_count / pose_count
                    if pose_count > 0
                    else 0
                )

                print()
                print(
                    f"[STATUS] "
                    f"time={elapsed:.1f}s | "
                    f"CSI={csi_count} "
                    f"({csi_rate:.2f}Hz) | "
                    f"Pose={pose_count} | "
                    f"Person={person_count} "
                    f"({person_ratio*100:.1f}%) | "
                    f"Empty={empty_count} | "
                    f"BadCSI={bad}"
                )

                last_stats_print = now

            # ------------------------------------------------
            # Pose memory safety
            # ------------------------------------------------

            if (
                time.monotonic()
                - last_pose_flush
                >= POSE_FLUSH_INTERVAL
            ):

                # Pose records stay in memory for final save.
                # We only update flush timestamp here.
                last_pose_flush = time.monotonic()

    except KeyboardInterrupt:

        print()
        print("[INFO] KeyboardInterrupt")

        stop_event.set()

    except Exception:

        print()
        print("[ERROR] Main loop crashed.")

        traceback.print_exc()

        stop_event.set()

    finally:

        print()
        print("=" * 80)
        print("Stopping")
        print("=" * 80)

        stop_event.set()

        # ----------------------------------------------------
        # Give CSI thread time to finish
        # ----------------------------------------------------

        try:

            csi_reader.join(
                timeout=2.0
            )

        except Exception:
            pass

        # ----------------------------------------------------
        # Drain remaining CSI
        # ----------------------------------------------------

        with csi_lock:

            while csi_buffer:

                packet = csi_buffer.popleft()

                csi_chunk.append(packet)

        # ----------------------------------------------------
        # Save remaining CSI
        # ----------------------------------------------------

        if csi_chunk:

            path = save_csi_chunk(
                csi_dir,
                chunk_id,
                csi_chunk
            )

            if path:

                print(
                    f"[SAVE CSI FINAL] "
                    f"chunk={chunk_id:05d} "
                    f"samples={len(csi_chunk)}"
                )

        # ----------------------------------------------------
        # Camera release
        # ----------------------------------------------------

        try:

            cap.release()

        except Exception:
            pass

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass

        # ----------------------------------------------------
        # Save pose timeline
        # ----------------------------------------------------

        print()
        print("[INFO] Saving pose timeline...")

        pose_path = save_pose_timeline(
            pose_dir,
            pose_records
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        session_end = time.monotonic()

        metadata_path = save_metadata(
            session_dir,
            session_start,
            session_end
        )

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        with stats_lock:

            s = dict(stats)

        duration = (
            session_end - session_start
        )

        csi_rate = (
            s["csi_packets"] / duration
            if duration > 0
            else 0
        )

        person_ratio = (
            s["person_frames"] /
            s["pose_frames"]
            if s["pose_frames"] > 0
            else 0
        )

        print()
        print("=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        print()
        print(
            f"Session                 : "
            f"{os.path.basename(session_dir)}"
        )

        print(
            f"Duration                : "
            f"{duration:.2f} sec"
        )

        print(
            f"CSI packets             : "
            f"{s['csi_packets']}"
        )

        print(
            f"CSI bad packets         : "
            f"{s['csi_bad_packets']}"
        )

        print(
            f"CSI rate                : "
            f"{csi_rate:.2f} Hz"
        )

        print(
            f"Pose frames             : "
            f"{s['pose_frames']}"
        )

        print(
            f"Person frames           : "
            f"{s['person_frames']}"
        )

        print(
            f"Empty-room frames       : "
            f"{s['empty_frames']}"
        )

        print(
            f"Person ratio            : "
            f"{person_ratio * 100:.2f}%"
        )

        print()
        print(
            f"CSI directory           : "
            f"{csi_dir}"
        )

        print(
            f"Pose timeline           : "
            f"{pose_path}"
        )

        print(
            f"Metadata                : "
            f"{metadata_path}"
        )

        print()
        print("=" * 80)
        print("DONE")
        print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()