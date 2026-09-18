#!/usr/bin/env python3

import os
import cv2
import json
import time
import math
import signal
import threading
import traceback

import numpy as np
from collections import deque
from datetime import datetime

import serial
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

COM_PORT = "COM9"
BAUD_RATE = 921600

CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW

YOLO_MODEL = "yolov8n-pose.pt"

DATASET_ROOT = "wifi_pose_dataset"

CSI_CHUNK_SIZE = 5000

# Maximum number of CSI packets kept in RAM before writer consumes them
CSI_BUFFER_SIZE = 30000

# Pose timeline RAM size
POSE_BUFFER_SIZE = 10000

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Only one person is currently needed.
# If multiple people exist, we select the person
# with highest average keypoint confidence.
MIN_PERSON_CONF = 0.20

# ============================================================
# CSI PARSER
# ============================================================

CSI_MAGIC_V1 = 0xC5110001


def parse_csi_line(line):

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

        raw = parts[4:4 + data_len]

        if len(raw) < data_len:
            return None

        iq = []

        for x in raw:

            v = int(x)

            if v < -128:
                v = -128

            elif v > 127:
                v = 127

            iq.append(v)

        if len(iq) % 2 != 0:
            iq = iq[:-1]

        iq = np.asarray(iq, dtype=np.int16)

        n_subcarriers = len(iq) // 2

        if n_subcarriers <= 0:
            return None

        # I/Q -> amplitude

        i = iq[0::2].astype(np.float32)
        q = iq[1::2].astype(np.float32)

        amplitude = np.sqrt(i * i + q * q)

        return {
            "timestamp": timestamp,
            "rssi": rssi,
            "channel": channel,
            "iq": iq,
            "amplitude": amplitude,
            "n_subcarriers": n_subcarriers,
        }

    except Exception:
        return None


# ============================================================
# CSI READER THREAD
# ============================================================

class CSIReader:

    def __init__(self):

        self.running = False

        self.ser = None

        self.buffer = deque(maxlen=CSI_BUFFER_SIZE)

        self.lock = threading.Lock()

        self.thread = None

        self.total_packets = 0

        self.bad_packets = 0

        self.start_time = None

        self.last_packet_time = None

        self.bytes_received = 0

    def start(self):

        print()
        print("=" * 80)
        print("Opening CSI serial...")
        print("=" * 80)

        self.ser = serial.Serial(
            COM_PORT,
            BAUD_RATE,
            timeout=0.1
        )

        print(f"COM     : {COM_PORT}")
        print(f"Baud    : {BAUD_RATE}")

        self.running = True

        self.start_time = time.monotonic()

        self.thread = threading.Thread(
            target=self._reader_loop,
            daemon=True
        )

        self.thread.start()

    def _reader_loop(self):

        while self.running:

            try:

                raw = self.ser.readline()

                if not raw:
                    continue

                self.bytes_received += len(raw)

                try:
                    line = raw.decode(
                        "utf-8",
                        errors="ignore"
                    )
                except Exception:
                    continue

                packet = parse_csi_line(line)

                if packet is None:

                    self.bad_packets += 1

                    continue

                now = time.monotonic()

                packet["arrival_time"] = now

                # Keep raw ESP timestamp.
                # IMPORTANT:
                # This is NOT used directly for synchronization.
                # We save it for analysis.

                with self.lock:

                    self.buffer.append(packet)

                self.total_packets += 1

                self.last_packet_time = now

            except serial.SerialException as e:

                print()
                print("[CSI] Serial error:")
                print(e)

                break

            except Exception:

                traceback.print_exc()

        print("[CSI] reader stopped")

    def pop_all(self):

        with self.lock:

            items = list(self.buffer)

            self.buffer.clear()

        return items

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(timeout=2)

        if self.ser is not None:

            try:
                self.ser.close()
            except Exception:
                pass


# ============================================================
# SESSION WRITER
# ============================================================

class SessionWriter:

    def __init__(self, session_dir):

        self.session_dir = session_dir

        self.csi_dir = os.path.join(
            session_dir,
            "csi"
        )

        self.pose_dir = os.path.join(
            session_dir,
            "pose"
        )

        os.makedirs(
            self.csi_dir,
            exist_ok=True
        )

        os.makedirs(
            self.pose_dir,
            exist_ok=True
        )

        self.csi_buffer = []

        self.chunk_id = 0

        self.pose_times = []
        self.pose_data = []
        self.pose_confidence = []

        self.lock = threading.Lock()

    # --------------------------------------------------------

    def add_csi(self, packet):

        self.csi_buffer.append(packet)

        if len(self.csi_buffer) >= CSI_CHUNK_SIZE:

            self.flush_csi()

    # --------------------------------------------------------

    def flush_csi(self):

        if not self.csi_buffer:
            return

        items = self.csi_buffer

        self.csi_buffer = []

        n = len(items)

        n_sub = 128

        amplitude = np.zeros(
            (n, n_sub),
            dtype=np.float32
        )

        iq = np.zeros(
            (n, 256),
            dtype=np.int16
        )

        timestamp = np.zeros(
            n,
            dtype=np.int64
        )

        rssi = np.zeros(
            n,
            dtype=np.int16
        )

        channel = np.zeros(
            n,
            dtype=np.int16
        )

        arrival_time = np.zeros(
            n,
            dtype=np.float64
        )

        actual_n_subcarriers = np.zeros(
            n,
            dtype=np.int16
        )

        for idx, packet in enumerate(items):

            amp = packet["amplitude"]

            iq_packet = packet["iq"]

            length = min(
                len(amp),
                n_sub
            )

            amplitude[idx, :length] = amp[:length]

            iq_length = min(
                len(iq_packet),
                256
            )

            iq[idx, :iq_length] = iq_packet[:iq_length]

            timestamp[idx] = packet["timestamp"]

            rssi[idx] = packet["rssi"]

            channel[idx] = packet["channel"]

            arrival_time[idx] = packet["arrival_time"]

            actual_n_subcarriers[idx] = \
                packet["n_subcarriers"]

        filename = os.path.join(
            self.csi_dir,
            f"chunk_{self.chunk_id:05d}.npz"
        )

        np.savez_compressed(
            filename,

            csi_amplitude=amplitude,

            iq=iq,

            timestamp=timestamp,

            rssi=rssi,

            channel=channel,

            csi_arrival_time=arrival_time,

            n_subcarriers=actual_n_subcarriers,
        )

        print(
            f"[CSI SAVE] "
            f"chunk={self.chunk_id:05d} "
            f"samples={n}"
        )

        self.chunk_id += 1

    # --------------------------------------------------------

    def add_pose(
        self,
        camera_time,
        pose,
        confidence
    ):

        self.pose_times.append(
            camera_time
        )

        self.pose_data.append(
            pose.copy()
        )

        self.pose_confidence.append(
            confidence.copy()
        )

    # --------------------------------------------------------

    def flush_pose(self):

        if not self.pose_times:

            return

        pose_times = np.asarray(
            self.pose_times,
            dtype=np.float64
        )

        pose_data = np.asarray(
            self.pose_data,
            dtype=np.float32
        )

        pose_confidence = np.asarray(
            self.pose_confidence,
            dtype=np.float32
        )

        filename = os.path.join(
            self.pose_dir,
            "pose_timeline.npz"
        )

        np.savez_compressed(

            filename,

            camera_time=pose_times,

            pose=pose_data,

            pose_confidence=pose_confidence,
        )

        print()
        print(
            "[POSE SAVE]"
            f" samples={len(pose_times)}"
        )

    # --------------------------------------------------------

    def finalize(self):

        self.flush_csi()

        self.flush_pose()


# ============================================================
# YOLO POSE
# ============================================================

def extract_pose(model, frame):

    """
    Returns:

        pose:
            shape (17,2)

        confidence:
            shape (17,)

        detected:
            bool
    """

    pose = np.full(
        (17, 2),
        np.nan,
        dtype=np.float32
    )

    confidence = np.full(
        17,
        np.nan,
        dtype=np.float32
    )

    try:

        results = model(
            frame,
            verbose=False
        )

        if not results:
            return pose, confidence, False

        result = results[0]

        if result.keypoints is None:
            return pose, confidence, False

        if result.keypoints.xy is None:
            return pose, confidence, False

        xy = result.keypoints.xy.cpu().numpy()

        if len(xy) == 0:
            return pose, confidence, False

        # Confidence

        if result.keypoints.conf is not None:

            conf = result.keypoints.conf.cpu().numpy()

        else:

            conf = np.ones(
                (len(xy), 17),
                dtype=np.float32
            )

        # Select best person

        if len(xy) == 1:

            best = 0

        else:

            scores = []

            for person_idx in range(len(xy)):

                person_conf = conf[person_idx]

                valid = person_conf[
                    person_conf >= MIN_PERSON_CONF
                ]

                if len(valid) == 0:

                    score = 0

                else:

                    score = float(
                        np.mean(valid)
                    )

                scores.append(score)

            best = int(
                np.argmax(scores)
            )

        selected_xy = xy[best]

        selected_conf = conf[best]

        # Normalize coordinates to [0,1]

        selected_xy[:, 0] /= CAMERA_WIDTH

        selected_xy[:, 1] /= CAMERA_HEIGHT

        selected_xy = np.clip(
            selected_xy,
            0.0,
            1.0
        )

        pose[:, :] = selected_xy

        confidence[:] = selected_conf

        detected = bool(
            np.any(
                selected_conf >= MIN_PERSON_CONF
            )
        )

        return pose, confidence, detected

    except Exception:

        traceback.print_exc()

        return pose, confidence, False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("WiFi CSI + Camera Pose Dataset Collector V3")
    print("=" * 80)

    session_name = datetime.now().strftime(
        "session_%Y%m%d_%H%M%S"
    )

    session_dir = os.path.join(
        DATASET_ROOT,
        session_name
    )

    os.makedirs(
        session_dir,
        exist_ok=True
    )

    print()
    print(f"Session directory:")
    print(session_dir)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "version": "V3",

        "created_at":
            datetime.now().isoformat(),

        "serial": {
            "port": COM_PORT,
            "baud": BAUD_RATE,
        },

        "camera": {
            "index": CAMERA_INDEX,
            "backend": "DSHOW",
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fps_requested": CAMERA_FPS,
        },

        "yolo": {
            "model": YOLO_MODEL,
            "keypoints": 17,
        },

        "csi": {
            "expected_subcarriers": 128,
            "expected_iq_values": 256,
        },

        "synchronization": {

            "method":
                "offline alignment",

            "clock":
                "time.monotonic for CSI/camera",

            "raw_esp_timestamp_saved":
                True,
        },

        "notes": (
            "CSI and camera pose are recorded "
            "independently. Alignment is performed "
            "offline after acquisition."
        ),
    }

    metadata_file = os.path.join(
        session_dir,
        "metadata.json"
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
    # CSI
    # --------------------------------------------------------

    csi_reader = CSIReader()

    csi_reader.start()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Opening camera...")
    print("=" * 80)

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND
    )

    if not cap.isOpened():

        csi_reader.stop()

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
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Camera: "
        f"{actual_width}x{actual_height} "
        f"FPS={actual_fps}"
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print()
    print(
        f"Loading YOLO model: "
        f"{YOLO_MODEL}"
    )

    model = YOLO(
        YOLO_MODEL
    )

    print("YOLO loaded.")

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    writer = SessionWriter(
        session_dir
    )

    running = True

    start_time = time.monotonic()

    last_stats = start_time

    last_csi_count = 0

    last_pose_count = 0

    total_pose_frames = 0

    total_person_frames = 0

    print()
    print("=" * 80)
    print("START RECORDING")
    print("=" * 80)

    print()
    print("Controls:")
    print("  Q = quit")
    print("  ESC = quit")
    print()
    print("CSI is recorded continuously.")
    print("Pose is recorded independently.")
    print()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    try:

        while running:

            # ------------------------------------------------
            # CAMERA
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok:

                print(
                    "[CAMERA] Failed to read frame"
                )

                time.sleep(0.01)

                continue

            camera_time = time.monotonic()

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            pose, confidence, detected = \
                extract_pose(
                    model,
                    frame
                )

            writer.add_pose(
                camera_time,
                pose,
                confidence
            )

            total_pose_frames += 1

            if detected:

                total_person_frames += 1

            # ------------------------------------------------
            # CSI
            #
            # IMPORTANT:
            # We DO NOT require a pose match here.
            #
            # CSI is saved continuously.
            # ------------------------------------------------

            packets = csi_reader.pop_all()

            for packet in packets:

                writer.add_csi(
                    packet
                )

            # ------------------------------------------------
            # DRAW CAMERA
            # ------------------------------------------------

            display = frame.copy()

            if detected:

                for k in range(17):

                    x = pose[k, 0]

                    y = pose[k, 1]

                    c = confidence[k]

                    if not np.isfinite(x):
                        continue

                    if not np.isfinite(y):
                        continue

                    px = int(
                        x * CAMERA_WIDTH
                    )

                    py = int(
                        y * CAMERA_HEIGHT
                    )

                    cv2.circle(
                        display,
                        (px, py),
                        4,
                        (0, 255, 0),
                        -1
                    )

            elapsed = (
                time.monotonic()
                - start_time
            )

            cv2.putText(
                display,
                f"TIME: {elapsed:.1f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"CSI: {csi_reader.total_packets}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"POSE: {total_pose_frames}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            if detected:

                cv2.putText(
                    display,
                    "PERSON: YES",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "PERSON: NO",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "CSI + Pose Collector V3",
                display
            )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                running = False

            elif key == 27:

                running = False

            # ------------------------------------------------
            # STATS
            # ------------------------------------------------

            now = time.monotonic()

            if now - last_stats >= 5:

                current_csi = \
                    csi_reader.total_packets

                csi_delta = \
                    current_csi - last_csi_count

                elapsed_stats = \
                    now - last_stats

                csi_rate = \
                    csi_delta / elapsed_stats

                pose_delta = \
                    total_pose_frames - last_pose_count

                pose_rate = \
                    pose_delta / elapsed_stats

                print()
                print(
                    f"[STATS] "
                    f"CSI={current_csi} "
                    f"({csi_rate:.2f} Hz) | "
                    f"POSE={total_pose_frames} "
                    f"({pose_rate:.2f} Hz) | "
                    f"PERSON={total_person_frames}"
                )

                last_csi_count = current_csi

                last_pose_count = \
                    total_pose_frames

                last_stats = now

    except KeyboardInterrupt:

        print()
        print("CTRL+C received.")

    except Exception:

        traceback.print_exc()

    finally:

        print()
        print("=" * 80)
        print("STOPPING...")
        print("=" * 80)

        running = False

        # Stop camera

        try:
            cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()

        # Drain remaining CSI

        remaining = csi_reader.pop_all()

        print(
            f"Draining "
            f"{len(remaining)} CSI packets..."
        )

        for packet in remaining:

            writer.add_csi(
                packet
            )

        # Stop serial thread

        csi_reader.stop()

        # Flush everything

        writer.finalize()

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        total_time = (
            time.monotonic()
            - start_time
        )

        summary = {

            "session": session_name,

            "duration_seconds":
                total_time,

            "csi_packets":
                csi_reader.total_packets,

            "csi_bad_packets":
                csi_reader.bad_packets,

            "csi_rate_hz":
                (
                    csi_reader.total_packets
                    / total_time
                    if total_time > 0
                    else 0
                ),

            "pose_frames":
                total_pose_frames,

            "person_frames":
                total_person_frames,

            "person_detection_ratio":
                (
                    total_person_frames
                    / total_pose_frames
                    if total_pose_frames > 0
                    else 0
                ),

            "csi_chunks":
                writer.chunk_id,
        }

        summary_file = os.path.join(
            session_dir,
            "summary.json"
        )

        with open(
            summary_file,
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
        print("SESSION COMPLETE")
        print("=" * 80)

        print(
            json.dumps(
                summary,
                indent=2
            )
        )

        print()
        print(
            f"Dataset saved to:\n"
            f"{session_dir}"
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()