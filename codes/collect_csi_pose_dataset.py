# collect_csi_pose_dataset.py

import cv2
import serial
import time
import math
import threading
import queue
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

COM_PORT = "COM9"
BAUD_RATE = 921600

# IMPORTANT:
# 0 = EShare Virtual Camera
# 1 = HP HD Webcam [Fixed]
CAMERA_INDEX = 1

# We verified DSHOW + Camera 1 gives valid frames.
CAMERA_BACKEND = cv2.CAP_DSHOW

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_MODEL = "yolov8n-pose.pt"

EXPECTED_SUBCARRIERS = 128

# Maximum allowed difference between CSI arrival time
# and camera frame time for pairing.
MAX_PAIR_AGE_MS = 200.0

# Number of synchronized samples per NPZ chunk.
CHUNK_SIZE = 1000

OUTPUT_ROOT = Path("wifi_pose_dataset")

# Serial line timeout
SERIAL_TIMEOUT = 0.1

# Number of CSI frames kept in RAM for synchronization.
CSI_BUFFER_SIZE = 1000

# Display window name
WINDOW_NAME = "CSI + YOLO Pose Collector"


# ============================================================
# GLOBAL STATE
# ============================================================

stop_event = threading.Event()
pause_event = threading.Event()

stats_lock = threading.Lock()

stats = {
    "csi_received": 0,
    "csi_valid": 0,
    "camera_frames": 0,
    "poses_detected": 0,
    "samples_saved": 0,
    "bad_csi": 0,
}


# ============================================================
# CSI PARSER
# ============================================================

def parse_csi_line(line: str):
    """
    Parse CSI serial line.

    Expected format:

        timestamp,rssi,channel,data_len,I,Q,I,Q,...

    Returns:

        {
            "timestamp": int,
            "rssi": int,
            "channel": int,
            "node_id": 1,
            "n_antennas": 1,
            "n_subcarriers": 128,
            "noise_floor": -90,
            "iq_data": list[int]
        }

    or None on failure.
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

        data_start = 4
        data_end = data_start + data_len

        if len(parts) < data_end:
            return None

        raw_values = parts[data_start:data_end]

        iq_data = []

        for value in raw_values:
            v = int(value)

            # ESP32 CSI I/Q are signed int8.
            v = max(-128, min(127, v))

            iq_data.append(v)

        # Need complete I/Q pairs.
        if len(iq_data) < 2:
            return None

        if len(iq_data) % 2 != 0:
            iq_data = iq_data[:-1]

        n_subcarriers = len(iq_data) // 2

        if n_subcarriers != EXPECTED_SUBCARRIERS:
            return None

        return {
            "timestamp": timestamp,
            "rssi": rssi,
            "channel": channel,
            "node_id": 1,
            "n_antennas": 1,
            "n_subcarriers": n_subcarriers,
            "noise_floor": -90,
            "iq_data": iq_data,
            "arrival_time": time.perf_counter(),
        }

    except Exception:
        return None


# ============================================================
# I/Q -> AMPLITUDE
# ============================================================

def iq_to_amplitude(iq_data):
    """
    Convert:

        [I0,Q0,I1,Q1,...]

    to:

        [sqrt(I0²+Q0²), sqrt(I1²+Q1²), ...]
    """

    iq = np.asarray(iq_data, dtype=np.float32)

    if len(iq) != EXPECTED_SUBCARRIERS * 2:
        return None

    i = iq[0::2]
    q = iq[1::2]

    amplitude = np.sqrt(i * i + q * q)

    return amplitude.astype(np.float32)


# ============================================================
# CSI READER THREAD
# ============================================================

class CSIReader:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate

        self.ser = None

        self.buffer = queue.Queue(maxsize=CSI_BUFFER_SIZE)

        self.thread = None

        self.running = False

    def start(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=SERIAL_TIMEOUT,
            )

            print(f"[OK] Serial connected: {self.port}")

        except Exception as e:
            print()
            print("[ERROR] Could not open serial port.")
            print(f"        Port: {self.port}")
            print(f"        Error: {e}")
            raise

        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="CSIReader",
        )

        self.thread.start()

        print("[CSI] Serial reader started.")

    def _worker(self):

        while not stop_event.is_set():

            try:
                raw = self.ser.readline()

                if not raw:
                    continue

                with stats_lock:
                    stats["csi_received"] += 1

                try:
                    line = raw.decode(
                        "utf-8",
                        errors="ignore",
                    ).strip()

                except Exception:
                    continue

                packet = parse_csi_line(line)

                if packet is None:
                    with stats_lock:
                        stats["bad_csi"] += 1
                    continue

                amplitude = iq_to_amplitude(
                    packet["iq_data"]
                )

                if amplitude is None:
                    with stats_lock:
                        stats["bad_csi"] += 1
                    continue

                packet["amplitude"] = amplitude

                # Queue overflow policy:
                # Remove oldest packet and insert newest.
                try:
                    self.buffer.put_nowait(packet)

                except queue.Full:
                    try:
                        self.buffer.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        self.buffer.put_nowait(packet)
                    except queue.Full:
                        pass

                with stats_lock:
                    stats["csi_valid"] += 1

            except serial.SerialException as e:

                print()
                print("[ERROR] Serial communication error:")
                print(e)

                stop_event.set()
                break

            except Exception as e:

                print()
                print("[ERROR] CSI reader exception:")
                print(e)

                traceback.print_exc()

                stop_event.set()
                break

    def get_latest(self):
        """
        Return newest CSI packet currently in queue.

        Older packets are discarded.
        """

        latest = None

        while True:

            try:
                packet = self.buffer.get_nowait()
                latest = packet

            except queue.Empty:
                break

        return latest

    def get_closest_to(self, target_time):
        """
        Find the CSI packet whose arrival time is closest
        to target_time.

        Since the queue is small, a simple search is enough.
        """

        packets = []

        while True:

            try:
                packets.append(
                    self.buffer.get_nowait()
                )

            except queue.Empty:
                break

        if not packets:
            return None

        best = min(
            packets,
            key=lambda p: abs(
                p["arrival_time"] - target_time
            )
        )

        return best

    def close(self):

        self.running = False

        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass


# ============================================================
# DATASET WRITER
# ============================================================

class DatasetWriter:

    def __init__(
        self,
        output_root,
        chunk_size,
    ):

        self.output_root = Path(output_root)

        self.chunk_size = chunk_size

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.session_dir = (
            self.output_root
            / f"session_{timestamp}"
        )

        self.chunk_dir = (
            self.session_dir
            / "chunks"
        )

        self.chunk_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chunk_index = 0

        self.csi_amplitude = []
        self.iq = []
        self.pose = []
        self.pose_confidence = []

        self.timestamp = []
        self.rssi = []
        self.channel = []

        self.pose_age_ms = []

        self.csi_arrival_time = []
        self.camera_time = []

        self.total_saved = 0

        self._write_metadata()

    def _write_metadata(self):

        metadata_path = (
            self.session_dir
            / "metadata.txt"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "CSI + CAMERA POSE DATASET\n"
            )

            f.write("=" * 60 + "\n")

            f.write(
                f"created: "
                f"{datetime.now().isoformat()}\n"
            )

            f.write(
                f"serial_port: {COM_PORT}\n"
            )

            f.write(
                f"baud_rate: {BAUD_RATE}\n"
            )

            f.write(
                f"camera_index: {CAMERA_INDEX}\n"
            )

            f.write(
                f"camera_backend: CAP_DSHOW\n"
            )

            f.write(
                f"camera_resolution: "
                f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}\n"
            )

            f.write(
                f"yolo_model: {YOLO_MODEL}\n"
            )

            f.write(
                f"subcarriers: "
                f"{EXPECTED_SUBCARRIERS}\n"
            )

            f.write(
                f"pose_keypoints: 17\n"
            )

            f.write(
                f"pose_dimensions: 2\n"
            )

            f.write(
                f"max_pair_age_ms: "
                f"{MAX_PAIR_AGE_MS}\n"
            )

            f.write(
                "\n"
            )

            f.write(
                "NPZ fields:\n"
            )

            f.write(
                "  csi_amplitude : [N,128]\n"
            )

            f.write(
                "  iq             : [N,256]\n"
            )

            f.write(
                "  pose           : [N,17,2]\n"
            )

            f.write(
                "  pose_confidence: [N,17]\n"
            )

            f.write(
                "  timestamp      : [N]\n"
            )

            f.write(
                "  rssi           : [N]\n"
            )

            f.write(
                "  channel        : [N]\n"
            )

            f.write(
                "  pose_age_ms    : [N]\n"
            )

            f.write(
                "  csi_arrival_time: [N]\n"
            )

            f.write(
                "  camera_time    : [N]\n"
            )

    def add(
        self,
        csi_packet,
        pose,
        pose_confidence,
        camera_time,
    ):

        csi_amplitude = csi_packet["amplitude"]

        iq_data = np.asarray(
            csi_packet["iq_data"],
            dtype=np.int16,
        )

        pose = np.asarray(
            pose,
            dtype=np.float32,
        )

        pose_confidence = np.asarray(
            pose_confidence,
            dtype=np.float32,
        )

        age_ms = abs(
            camera_time
            - csi_packet["arrival_time"]
        ) * 1000.0

        self.csi_amplitude.append(
            csi_amplitude
        )

        self.iq.append(
            iq_data
        )

        self.pose.append(
            pose
        )

        self.pose_confidence.append(
            pose_confidence
        )

        self.timestamp.append(
            csi_packet["timestamp"]
        )

        self.rssi.append(
            csi_packet["rssi"]
        )

        self.channel.append(
            csi_packet["channel"]
        )

        self.pose_age_ms.append(
            age_ms
        )

        self.csi_arrival_time.append(
            csi_packet["arrival_time"]
        )

        self.camera_time.append(
            camera_time
        )

        self.total_saved += 1

        with stats_lock:
            stats["samples_saved"] += 1

        if len(self.csi_amplitude) >= self.chunk_size:
            self.flush()

    def flush(self):

        n = len(self.csi_amplitude)

        if n == 0:
            return

        output_path = (
            self.chunk_dir
            / f"chunk_{self.chunk_index:05d}.npz"
        )

        np.savez_compressed(
            output_path,

            csi_amplitude=np.asarray(
                self.csi_amplitude,
                dtype=np.float32,
            ),

            iq=np.asarray(
                self.iq,
                dtype=np.int16,
            ),

            pose=np.asarray(
                self.pose,
                dtype=np.float32,
            ),

            pose_confidence=np.asarray(
                self.pose_confidence,
                dtype=np.float32,
            ),

            timestamp=np.asarray(
                self.timestamp,
                dtype=np.int64,
            ),

            rssi=np.asarray(
                self.rssi,
                dtype=np.int16,
            ),

            channel=np.asarray(
                self.channel,
                dtype=np.int16,
            ),

            pose_age_ms=np.asarray(
                self.pose_age_ms,
                dtype=np.float64,
            ),

            csi_arrival_time=np.asarray(
                self.csi_arrival_time,
                dtype=np.float64,
            ),

            camera_time=np.asarray(
                self.camera_time,
                dtype=np.float64,
            ),
        )

        print()
        print(
            f"[SAVE] "
            f"{output_path.name} "
            f"({n} samples)"
        )

        self.chunk_index += 1

        self.csi_amplitude.clear()
        self.iq.clear()
        self.pose.clear()
        self.pose_confidence.clear()

        self.timestamp.clear()
        self.rssi.clear()
        self.channel.clear()

        self.pose_age_ms.clear()

        self.csi_arrival_time.clear()
        self.camera_time.clear()

    def close(self):

        self.flush()

        print()
        print(
            f"[DATASET] Total saved samples: "
            f"{self.total_saved}"
        )

        print(
            f"[DATASET] Location: "
            f"{self.session_dir}"
        )


# ============================================================
# YOLO POSE
# ============================================================

def load_yolo():

    print(
        "[INFO] Loading YOLO Pose..."
    )

    try:

        model = YOLO(YOLO_MODEL)

    except Exception as e:

        print()
        print(
            "[ERROR] Failed to load YOLO model."
        )

        print(e)

        raise

    print(
        "[OK] YOLO Pose loaded."
    )

    return model


# ============================================================
# EXTRACT BEST PERSON POSE
# ============================================================

def extract_pose(model, frame):
    """
    Run YOLO Pose and return the most confident person.

    Returns:

        pose:
            [17,2] normalized x,y

        confidence:
            [17]

        person_confidence:
            float

    or None if no person detected.
    """

    try:

        results = model.predict(
            source=frame,
            verbose=False,
            conf=0.25,
        )

    except Exception as e:

        print()
        print(
            "[ERROR] YOLO inference failed:"
        )

        print(e)

        return None, None, None

    if not results:
        return None, None, None

    result = results[0]

    if result.keypoints is None:
        return None, None, None

    if result.keypoints.xy is None:
        return None, None, None

    if len(result.keypoints.xy) == 0:
        return None, None, None

    frame_h, frame_w = frame.shape[:2]

    # keypoints:
    # [persons,17,2]

    xy = (
        result.keypoints.xy
        .detach()
        .cpu()
        .numpy()
    )

    if result.keypoints.conf is not None:

        kp_conf = (
            result.keypoints.conf
            .detach()
            .cpu()
            .numpy()
        )

    else:

        kp_conf = np.ones(
            (xy.shape[0], xy.shape[1]),
            dtype=np.float32,
        )

    # Person confidence
    if result.boxes is not None:
        if result.boxes.conf is not None:

            person_conf = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

        else:

            person_conf = np.ones(
                xy.shape[0],
                dtype=np.float32,
            )

    else:

        person_conf = np.ones(
            xy.shape[0],
            dtype=np.float32,
        )

    # Select best person.
    best_idx = int(
        np.argmax(person_conf)
    )

    best_xy = xy[best_idx]

    best_conf = kp_conf[best_idx]

    # Normalize coordinates to [0,1].
    best_xy[:, 0] /= float(frame_w)
    best_xy[:, 1] /= float(frame_h)

    best_xy = np.clip(
        best_xy,
        0.0,
        1.0,
    )

    return (
        best_xy.astype(np.float32),
        best_conf.astype(np.float32),
        float(person_conf[best_idx]),
    )


# ============================================================
# DRAW POSE
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


def draw_pose(
    frame,
    pose,
    confidence,
):

    if pose is None:
        return frame

    h, w = frame.shape[:2]

    # Draw skeleton edges.
    for a, b in POSE_EDGES:

        if (
            confidence[a] < 0.2
            or confidence[b] < 0.2
        ):
            continue

        xa = int(pose[a, 0] * w)
        ya = int(pose[a, 1] * h)

        xb = int(pose[b, 0] * w)
        yb = int(pose[b, 1] * h)

        cv2.line(
            frame,
            (xa, ya),
            (xb, yb),
            (0, 255, 0),
            2,
        )

    # Draw keypoints.
    for i in range(17):

        if confidence[i] < 0.2:
            continue

        x = int(pose[i, 0] * w)
        y = int(pose[i, 1] * h)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            frame,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# CAMERA OPEN
# ============================================================

def open_camera():

    print()
    print(
        "=" * 70
    )

    print(
        "OPENING HP WEBCAM"
    )

    print(
        "=" * 70
    )

    print(
        f"[CAMERA] Index  : {CAMERA_INDEX}"
    )

    print(
        "[CAMERA] Backend: CAP_DSHOW"
    )

    print(
        f"[CAMERA] Target : "
        f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
    )

    print()

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND,
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"index {CAMERA_INDEX}"
        )

    # Set requested resolution.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    # Give DirectShow some time.
    time.sleep(0.5)

    # Read a few frames.
    valid_frame = None

    for _ in range(10):

        ret, frame = cap.read()

        if ret and frame is not None:

            if (
                frame.size > 0
                and frame.shape[0] > 0
                and frame.shape[1] > 0
            ):

                valid_frame = frame
                break

        time.sleep(0.05)

    if valid_frame is None:

        cap.release()

        raise RuntimeError(
            "Camera opened but no valid frame "
            "was received."
        )

    h, w = valid_frame.shape[:2]

    print(
        "[OK] Camera opened successfully."
    )

    print(
        f"[CAMERA] Actual resolution: "
        f"{w}x{h}"
    )

    print(
        f"[CAMERA] Frame dtype: "
        f"{valid_frame.dtype}"
    )

    print(
        f"[CAMERA] Frame min/max: "
        f"{valid_frame.min()}/"
        f"{valid_frame.max()}"
    )

    print(
        f"[CAMERA] Frame mean: "
        f"{valid_frame.mean():.2f}"
    )

    print(
        f"[CAMERA] Frame std: "
        f"{valid_frame.std():.2f}"
    )

    if (
        valid_frame.max() == 0
        or valid_frame.mean() == 0
    ):

        cap.release()

        raise RuntimeError(
            "Camera returned an all-black frame."
        )

    return cap, valid_frame


# ============================================================
# STATUS
# ============================================================

def print_status(
    writer,
    csi_queue_size,
):

    with stats_lock:

        s = dict(stats)

    print(
        "\r"
        f"[STATUS] "
        f"camera={s['camera_frames']} | "
        f"poses={s['poses_detected']} | "
        f"CSI={s['csi_valid']} | "
        f"saved={s['samples_saved']} | "
        f"queue={csi_queue_size} | "
        f"badCSI={s['bad_csi']}",
        end="",
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "CSI + CAMERA POSE DATASET COLLECTOR"
    )

    print(
        "=" * 70
    )

    print(
        f"COM      : {COM_PORT}"
    )

    print(
        f"BAUD     : {BAUD_RATE}"
    )

    print(
        f"Camera   : {CAMERA_INDEX}"
    )

    print(
        f"Backend  : CAP_DSHOW"
    )

    print(
        f"Resolution: "
        f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
    )

    print(
        f"YOLO     : {YOLO_MODEL}"
    )

    print(
        f"Expected : "
        f"{EXPECTED_SUBCARRIERS} subcarriers"
    )

    print(
        "=" * 70
    )

    csi_reader = None
    camera = None
    writer = None

    try:

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        writer = DatasetWriter(
            OUTPUT_ROOT,
            CHUNK_SIZE,
        )

        print()
        print(
            f"Dataset:"
        )

        print(
            writer.session_dir
        )

        # ----------------------------------------------------
        # CSI
        # ----------------------------------------------------

        csi_reader = CSIReader(
            COM_PORT,
            BAUD_RATE,
        )

        csi_reader.start()

        # Give CSI reader a moment.
        time.sleep(0.5)

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        camera, first_frame = open_camera()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        model = load_yolo()

        print()
        print(
            "=" * 70
        )

        print(
            "READY"
        )

        print(
            "=" * 70
        )

        print(
            "P   = Pause / Resume"
        )

        print(
            "Q   = Quit"
        )

        print(
            "ESC = Quit"
        )

        print(
            "=" * 70
        )

        print()
        print(
            "[INFO] Camera is running."
        )

        print(
            "[INFO] Move in front of the camera."
        )

        print()

        # ----------------------------------------------------
        # MAIN CAMERA LOOP
        # ----------------------------------------------------

        last_status_time = time.perf_counter()

        latest_pose = None
        latest_confidence = None

        while not stop_event.is_set():

            # ------------------------------------------------
            # CAMERA FRAME
            # ------------------------------------------------

            camera_time = time.perf_counter()

            ret, frame = camera.read()

            if not ret:

                print()
                print(
                    "[ERROR] Camera frame read failed."
                )

                stop_event.set()
                break

            if frame is None:

                print()
                print(
                    "[ERROR] Camera returned None frame."
                )

                stop_event.set()
                break

            if frame.size == 0:

                print()
                print(
                    "[ERROR] Camera returned empty frame."
                )

                stop_event.set()
                break

            # Detect all-black frames.
            if frame.max() == 0:

                print()
                print(
                    "[ERROR] Camera returned "
                    "an all-black frame."
                )

                print(
                    "[ERROR] Stopping collector."
                )

                stop_event.set()
                break

            with stats_lock:
                stats["camera_frames"] += 1

            # ------------------------------------------------
            # PAUSE
            # ------------------------------------------------

            if pause_event.is_set():

                display = frame.copy()

                cv2.putText(
                    display,
                    "PAUSED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    stop_event.set()
                    break

                if key in (
                    ord("p"),
                    ord("P"),
                ):

                    pause_event.clear()

                    print()
                    print(
                        "[INFO] RESUMED"
                    )

                continue

            # ------------------------------------------------
            # YOLO POSE
            # ------------------------------------------------

            pose, confidence, person_conf = (
                extract_pose(
                    model,
                    frame,
                )
            )

            if pose is not None:

                latest_pose = pose
                latest_confidence = confidence

                with stats_lock:
                    stats["poses_detected"] += 1

            # ------------------------------------------------
            # CSI MATCHING
            # ------------------------------------------------

            if pose is not None:

                csi_packet = (
                    csi_reader.get_closest_to(
                        camera_time
                    )
                )

                if csi_packet is not None:

                    age_ms = abs(
                        camera_time
                        - csi_packet[
                            "arrival_time"
                        ]
                    ) * 1000.0

                    if age_ms <= MAX_PAIR_AGE_MS:

                        writer.add(
                            csi_packet=csi_packet,
                            pose=pose,
                            pose_confidence=confidence,
                            camera_time=camera_time,
                        )

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            display = frame.copy()

            if pose is not None:

                display = draw_pose(
                    display,
                    pose,
                    confidence,
                )

            # ------------------------------------------------
            # OVERLAY
            # ------------------------------------------------

            with stats_lock:
                s = dict(stats)

            cv2.putText(
                display,
                f"CSI: {s['csi_valid']}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Pose: {s['poses_detected']}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                f"Saved: {s['samples_saved']}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if pose is not None:

                cv2.putText(
                    display,
                    "PERSON DETECTED",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            else:

                cv2.putText(
                    display,
                    "NO PERSON",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            # ------------------------------------------------
            # KEYBOARD
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                print()
                print(
                    "[INFO] Quit requested."
                )

                stop_event.set()
                break

            elif key in (
                ord("p"),
                ord("P"),
            ):

                pause_event.set()

                print()
                print(
                    "[INFO] PAUSED"
                )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            now = time.perf_counter()

            if (
                now - last_status_time
                >= 2.0
            ):

                print_status(
                    writer,
                    csi_reader.buffer.qsize(),
                )

                last_status_time = now

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] Ctrl+C received."
        )

        stop_event.set()

    except Exception as e:

        print()
        print(
            "=" * 70
        )

        print(
            "[FATAL ERROR]"
        )

        print(e)

        print(
            "=" * 70
        )

        traceback.print_exc()

        stop_event.set()

    finally:

        print()
        print()

        print(
            "=" * 70
        )

        print(
            "SHUTTING DOWN"
        )

        print(
            "=" * 70
        )

        stop_event.set()

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        if camera is not None:

            try:
                camera.release()
            except Exception:
                pass

        # ----------------------------------------------------
        # CSI
        # ----------------------------------------------------

        if csi_reader is not None:

            csi_reader.close()

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        if writer is not None:

            try:
                writer.close()
            except Exception as e:

                print(
                    "[ERROR] Dataset flush failed:"
                )

                print(e)

        # ----------------------------------------------------
        # GUI
        # ----------------------------------------------------

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print()
        print(
            "[DONE]"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()