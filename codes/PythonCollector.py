import serial
import os
import time
import numpy as np
from datetime import datetime


# =============================================================================
# CONFIG
# =============================================================================

PORT = "COM9"
BAUD = 921600

DATA_DIR = "..\\Data"
DATASET_NAME = "one-Person-in-Room"

MAX_SIZE = 100 * 1024 * 1024

# ESP32 CSI packet:
# 256 signed bytes = 128 complex values
RAW_CSI_SIZE = 256
CSI_COMPLEX_SIZE = RAW_CSI_SIZE // 2

# Only packets with exactly 256 CSI bytes are stored.
REQUIRE_FULL_PACKET = True


# =============================================================================
# PREPARE
# =============================================================================

os.makedirs(DATA_DIR, exist_ok=True)

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)


# =============================================================================
# DATASET STATE
# =============================================================================

dataset_index = 1
chunk_index = 1

packet_count = 0
skipped_packet_count = 0

start_time = time.time()


# =============================================================================
# BUFFERS
# =============================================================================

csi_buffer = []
pc_time_buffer = []
esp_time_buffer = []
rssi_buffer = []
channel_buffer = []
csi_len_buffer = []


# =============================================================================
# FILE MANAGEMENT
# =============================================================================

def create_new_file():
    global dataset_index, chunk_index

    filename = os.path.join(
        DATA_DIR,
        f"{DATASET_NAME}_{dataset_index:04d}_{chunk_index:06d}.npz"
    )

    chunk_index += 1

    return filename


FILE = create_new_file()


# =============================================================================
# SAVE BUFFER
# =============================================================================

def save_buffer():
    global FILE

    if len(csi_buffer) == 0:
        return

    csi_array = np.asarray(
        csi_buffer,
        dtype=np.complex64
    )

    pc_time_array = np.asarray(
        pc_time_buffer
    )

    esp_time_array = np.asarray(
        esp_time_buffer
    )

    rssi_array = np.asarray(
        rssi_buffer,
        dtype=np.float32
    )

    channel_array = np.asarray(
        channel_buffer,
        dtype=np.int16
    )

    csi_len_array = np.asarray(
        csi_len_buffer,
        dtype=np.int16
    )

    np.savez_compressed(
        FILE,
        csi=csi_array,
        pc_time=pc_time_array,
        esp_time=esp_time_array,
        rssi=rssi_array,
        channel=channel_array,
        csi_len=csi_len_array
    )

    print()
    print("=" * 80)
    print("FILE SAVED")
    print("=" * 80)
    print(f"File          : {FILE}")
    print(f"Packets       : {len(csi_array)}")
    print(f"CSI shape     : {csi_array.shape}")
    print(f"CSI dtype     : {csi_array.dtype}")
    print(f"CSI len       : {np.unique(csi_len_array)}")
    print(f"File size     : {os.path.getsize(FILE) / (1024 * 1024):.2f} MB")
    print("=" * 80)
    print()

    # Clear buffers
    csi_buffer.clear()
    pc_time_buffer.clear()
    esp_time_buffer.clear()
    rssi_buffer.clear()
    channel_buffer.clear()
    csi_len_buffer.clear()

    # New file
    FILE = create_new_file()


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

try:

    print("=" * 80)
    print("ESP32 CSI DATA COLLECTOR")
    print("=" * 80)

    print(f"Port              : {PORT}")
    print(f"Baud              : {BAUD}")
    print(f"Output directory  : {DATA_DIR}")
    print(f"Dataset            : {DATASET_NAME}")
    print(f"Expected CSI bytes: {RAW_CSI_SIZE}")
    print(f"Expected complex   : {CSI_COMPLEX_SIZE}")
    print("=" * 80)
    print()

    while True:

        # ---------------------------------------------------------------------
        # READ SERIAL LINE
        # ---------------------------------------------------------------------

        line = ser.readline().decode(
            errors="ignore"
        ).strip()

        if not line:
            continue

        # ---------------------------------------------------------------------
        # SPLIT
        # ---------------------------------------------------------------------

        values = line.split(",")

        if len(values) < 4:
            continue

        # ---------------------------------------------------------------------
        # HEADER
        #
        # ESP32:
        #
        # timestamp,rssi,channel,csi_len,...
        # ---------------------------------------------------------------------

        esp_time = values[0]

        try:
            rssi = int(values[1])
            channel = int(values[2])
            csi_len = int(values[3])

        except ValueError:
            continue

        # ---------------------------------------------------------------------
        # RAW CSI
        #
        # IMPORTANT:
        #
        # We use csi_len here.
        #
        # Previously the code used:
        #
        #     values[4:]
        #
        # and then zero-padded everything to 256.
        #
        # That created artificial CSI values for short packets.
        # ---------------------------------------------------------------------

        csi_raw = values[
            4 : 4 + csi_len
        ]

        # ---------------------------------------------------------------------
        # VALIDATE RAW LENGTH
        # ---------------------------------------------------------------------

        if len(csi_raw) != csi_len:

            skipped_packet_count += 1

            print(
                f"[SKIP] Invalid packet length | "
                f"reported={csi_len}, "
                f"received={len(csi_raw)}"
            )

            continue

        # ---------------------------------------------------------------------
        # REQUIRE FULL CSI PACKET
        # ---------------------------------------------------------------------

        if REQUIRE_FULL_PACKET:

            if csi_len != RAW_CSI_SIZE:

                skipped_packet_count += 1

                print(
                    f"[SKIP] Incomplete CSI packet | "
                    f"csi_len={csi_len} | "
                    f"expected={RAW_CSI_SIZE}"
                )

                continue

        # ---------------------------------------------------------------------
        # RAW CSI DEBUG
        #
        # Print the first valid packet exactly as received from ESP32.
        #
        # This helps us determine where the permanent zero columns originate.
        # ---------------------------------------------------------------------

        if packet_count == 0:

            print()
            print("=" * 80)
            print("FIRST VALID RAW CSI PACKET")
            print("=" * 80)

            print(f"ESP time      : {esp_time}")
            print(f"RSSI          : {rssi}")
            print(f"Channel       : {channel}")
            print(f"CSI length    : {csi_len}")
            print(f"Raw values    : {len(csi_raw)}")

            print()
            print("RAW CSI VALUES:")
            print(csi_raw)

            print("=" * 80)
            print()

        # ---------------------------------------------------------------------
        # CONVERT RAW CSI TO NUMPY
        #
        # ESP32 sends signed int8 values.
        #
        # Example:
        #
        # I = 2
        # Q = -6
        #
        # complex = 2 - 6j
        # ---------------------------------------------------------------------

        try:

            raw_values = np.asarray(
                csi_raw,
                dtype=np.int8
            )

        except Exception as e:

            skipped_packet_count += 1

            print(
                f"[SKIP] CSI conversion error: {e}"
            )

            continue

        # ---------------------------------------------------------------------
        # CHECK EVEN LENGTH
        #
        # Every complex sample requires:
        #
        # I + Q
        #
        # ---------------------------------------------------------------------

        if len(raw_values) % 2 != 0:

            skipped_packet_count += 1

            print(
                f"[SKIP] Odd CSI byte count: "
                f"{len(raw_values)}"
            )

            continue

        # ---------------------------------------------------------------------
        # I / Q
        # ---------------------------------------------------------------------

        I = raw_values[0::2].astype(
            np.float32
        )

        Q = raw_values[1::2].astype(
            np.float32
        )

        # ---------------------------------------------------------------------
        # COMPLEX CSI
        # ---------------------------------------------------------------------

        csi_complex = (
            I + 1j * Q
        ).astype(
            np.complex64
        )

        # ---------------------------------------------------------------------
        # FINAL SIZE CHECK
        # ---------------------------------------------------------------------

        if len(csi_complex) != CSI_COMPLEX_SIZE:

            skipped_packet_count += 1

            print(
                f"[SKIP] Invalid complex CSI size: "
                f"{len(csi_complex)}"
            )

            continue

        # ---------------------------------------------------------------------
        # PC TIMESTAMP
        # ---------------------------------------------------------------------

        pc_time = datetime.now().isoformat(
            timespec="milliseconds"
        )

        # ---------------------------------------------------------------------
        # STORE
        # ---------------------------------------------------------------------

        csi_buffer.append(
            csi_complex
        )

        pc_time_buffer.append(
            pc_time
        )

        esp_time_buffer.append(
            esp_time
        )

        rssi_buffer.append(
            rssi
        )

        channel_buffer.append(
            channel
        )

        csi_len_buffer.append(
            csi_len
        )

        packet_count += 1

        # ---------------------------------------------------------------------
        # LIVE STATUS
        # ---------------------------------------------------------------------

        if packet_count % 100 == 0:

            elapsed = time.time() - start_time

            rate = (
                packet_count / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"[DATA] "
                f"packets={packet_count} | "
                f"skipped={skipped_packet_count} | "
                f"buffer={len(csi_buffer)} | "
                f"rate={rate:.2f} pkt/s | "
                f"CSI={csi_complex.shape} | "
                f"RSSI={rssi}"
            )

        # ---------------------------------------------------------------------
        # SAVE EVERY 1000 PACKETS
        # ---------------------------------------------------------------------

        if len(csi_buffer) >= 1000:

            save_buffer()


# =============================================================================
# STOP
# =============================================================================

except KeyboardInterrupt:

    print()
    print("=" * 80)
    print("COLLECTOR STOPPED BY USER")
    print("=" * 80)


# =============================================================================
# ERROR
# =============================================================================

except Exception as e:

    print()
    print("=" * 80)
    print("COLLECTOR ERROR")
    print("=" * 80)
    print(type(e).__name__)
    print(e)
    print("=" * 80)


# =============================================================================
# FINAL SAVE
# =============================================================================

finally:

    if len(csi_buffer) > 0:

        print()
        print("Saving remaining packets...")

        save_buffer()

    ser.close()

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)

    print(f"Valid packets   : {packet_count}")
    print(f"Skipped packets : {skipped_packet_count}")
    print(f"Elapsed time    : {elapsed:.2f} sec")

    if elapsed > 0:

        print(
            f"Average rate    : "
            f"{packet_count / elapsed:.2f} pkt/s"
        )

    print("=" * 80)


    if len(csi_buffer) > 0:

        save_buffer()



    ser.close()



    print()
    print("=" * 60)
    print("Recording stopped.")
    print(
        f"Total packets: {packet_count}"
    )
    print("=" * 60)