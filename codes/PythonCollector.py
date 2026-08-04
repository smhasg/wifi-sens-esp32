import serial
import os
import time
import numpy as np
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM9"
BAUD = 921600

DATA_DIR = "..\\Data"

DATASET_NAME = "one-Person-in-Room"

# Maximum size of each file (optional)
MAX_SIZE = 100 * 1024 * 1024

RAW_CSI_SIZE = 256
CSI_COMPLEX_SIZE = RAW_CSI_SIZE // 2


# ============================================================
# CREATE DATA DIRECTORY
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# SERIAL CONNECTION
# ============================================================

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)


print("=" * 60)
print("CSI DATA RECORDER")
print("=" * 60)
print(f"Port       : {PORT}")
print(f"Baud       : {BAUD}")
print(f"CSI Raw    : {RAW_CSI_SIZE} values")
print(f"CSI Complex: {CSI_COMPLEX_SIZE} samples")
print(f"Output     : {DATA_DIR}")
print("=" * 60)


# ============================================================
# FILE MANAGEMENT
# ============================================================

dataset_index = 1
chunk_index = 1

packet_count = 0

start_time = time.time()


def create_new_file():

    global dataset_index
    global chunk_index


    filename = os.path.join(
        DATA_DIR,
        f"{DATASET_NAME}_{dataset_index:04d}_{chunk_index:06d}.npz"
    )


    print()
    print("=" * 60)
    print("Creating new file:")
    print(filename)
    print("=" * 60)


    chunk_index += 1


    return filename



# ============================================================
# TEMP BUFFERS
# ============================================================

csi_buffer = []

pc_time_buffer = []

esp_time_buffer = []

rssi_buffer = []

channel_buffer = []

csi_len_buffer = []


FILE = create_new_file()



# ============================================================
# SAVE FUNCTION
# ============================================================

def save_buffer():

    global csi_buffer
    global pc_time_buffer
    global esp_time_buffer
    global rssi_buffer
    global channel_buffer
    global csi_len_buffer
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


    print(
        f"Saved {len(csi_array)} packets | "
        f"CSI shape: {csi_array.shape} | "
        f"File size: "
        f"{os.path.getsize(FILE)/(1024*1024):.2f} MB"
    )


    # clear RAM

    csi_buffer.clear()

    pc_time_buffer.clear()

    esp_time_buffer.clear()

    rssi_buffer.clear()

    channel_buffer.clear()

    csi_len_buffer.clear()


    # create next file

    FILE = create_new_file()



# ============================================================
# RECORDING
# ============================================================

print()
print("Recording CSI...")
print("Press CTRL+C to stop.")
print()


try:

    while True:
        line = ser.readline().decode(
            errors="ignore"
        ).strip()

        if not line:
            continue

        values = line.split(",")

        if len(values) < 4:
            continue
        esp_time = values[0]
        rssi = values[1]
        channel = values[2]

        try:

            csi_len = int(values[3])

        except:

            continue



        csi_raw = values[4:]



        if len(csi_raw) > RAW_CSI_SIZE:

            csi_raw = csi_raw[:RAW_CSI_SIZE]


        elif len(csi_raw) < RAW_CSI_SIZE:

            csi_raw += (
                ["0"] *
                (RAW_CSI_SIZE-len(csi_raw))
            )



        try:


            raw_values = np.asarray(
                csi_raw,
                dtype=np.float32
            )


            I = raw_values[0::2]

            Q = raw_values[1::2]


            csi_complex = (

                I + 1j * Q

            ).astype(
                np.complex64
            )


        except Exception as e:

            print(
                "CSI conversion error:",
                e
            )

            continue



        pc_time = datetime.now().isoformat(
            timespec="milliseconds"
        )



        csi_buffer.append(
            csi_complex
        )


        pc_time_buffer.append(
            pc_time
        )


        esp_time_buffer.append(
            esp_time
        )


        try:

            rssi_buffer.append(
                float(rssi)
            )

        except:

            rssi_buffer.append(
                0
            )



        try:

            channel_buffer.append(
                int(channel)
            )

        except:

            channel_buffer.append(
                0
            )



        csi_len_buffer.append(
            csi_len
        )



        packet_count += 1



        if packet_count % 100 == 0:


            elapsed = time.time()-start_time


            rate = (
                packet_count / elapsed
                if elapsed > 0
                else 0
            )


            print(

                f"Packets: {packet_count} | "
                f"Rate: {rate:.1f} Hz | "
                f"CSI: {csi_complex.shape}"

            )



        # save every 1000 packets

        if len(csi_buffer) >= 1000:

            save_buffer()



except KeyboardInterrupt:


    print()
    print("Stopping recording...")



finally:


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