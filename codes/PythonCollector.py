
import serial
import os
import time
from datetime import datetime


PORT = "COM9"
BAUD = 921600

DATA_DIR = "..\\Data"
FILE = os.path.join(
    DATA_DIR,
    "OnePerson-sitting.csv"
)

MAX_SIZE = 100 * 1024 * 1024


os.makedirs(
    DATA_DIR,
    exist_ok=True
)


ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)


# ============================================================
# CSI FORMAT
#
# Raw CSI:
#
# [I0, Q0, I1, Q1, I2, Q2, ...]
#
# 256 raw values
# =
# 128 I/Q complex samples
#
# Output:
#
# cs0 = I0 + jQ0
# cs1 = I1 + jQ1
# ...
# cs127 = I127 + jQ127
# ============================================================


HEADER = [
    "pc_time",
    "esp_time",
    "rssi",
    "channel",
    "csi_len"
]


# 256 raw values -> 128 complex I/Q samples
HEADER += [
    f"cs{i}"
    for i in range(128)
]


# ============================================================
# Create file if it does not exist
# ============================================================

if not os.path.exists(FILE):

    with open(
        FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            ",".join(HEADER) + "\n"
        )


print("Recording CSI...")
print(f"Port: {PORT}")
print(f"Baud: {BAUD}")
print(f"File: {FILE}")


with open(
    FILE,
    "a",
    buffering=1,
    encoding="utf-8"
) as f:

    packet_count = 0

    start = time.time()


    while True:

        try:

            # ====================================================
            # Read serial line
            # ====================================================

            line = ser.readline().decode(
                errors="ignore"
            ).strip()


            if not line:
                continue


            values = line.split(",")


            # Expected:
            #
            # esp_time,rssi,channel,len,I0,Q0,I1,Q1,...
            #

            if len(values) < 4:
                continue


            # ====================================================
            # Read packet metadata
            # ====================================================

            esp_time = values[0]
            rssi = values[1]
            channel = values[2]


            try:

                csi_len = int(
                    values[3]
                )

            except ValueError:

                print(
                    "Invalid CSI length:",
                    values[3]
                )

                continue


            # ====================================================
            # Extract raw CSI
            #
            # Raw:
            #
            # [I0,Q0,I1,Q1,...]
            # ====================================================

            csi_raw = values[4:]


            # ====================================================
            # Normalize raw CSI to 256 values
            # ====================================================

            if len(csi_raw) > 256:

                csi_raw = csi_raw[:256]


            elif len(csi_raw) < 256:

                csi_raw += [
                    "0"
                ] * (
                    256 - len(csi_raw)
                )


            # ====================================================
            # Convert raw I/Q values to complex values
            #
            # [I0,Q0,I1,Q1,...]
            #
            # =>
            #
            # [I0+jQ0, I1+jQ1, ...]
            # ====================================================

            csi_complex = []


            for i in range(
                0,
                256,
                2
            ):

                try:

                    I = float(
                        csi_raw[i]
                    )

                    Q = float(
                        csi_raw[i + 1]
                    )


                    # Store as:
                    #
                    # I+jQ
                    #

                    complex_value = (
                        f"{I}+j{Q}"
                        if Q >= 0
                        else f"{I}-j{abs(Q)}"
                    )


                    csi_complex.append(
                        complex_value
                    )


                except ValueError:

                    # Invalid CSI sample
                    # Replace with zero

                    csi_complex.append(
                        "0+j0"
                    )


            # ====================================================
            # PC timestamp
            # ====================================================

            pc_time = datetime.now().isoformat(
                timespec="milliseconds"
            )


            # ====================================================
            # Create CSV row
            # ====================================================

            row = [
                pc_time,
                esp_time,
                rssi,
                channel,
                str(csi_len)
            ]


            row += csi_complex


            # ====================================================
            # Write row
            # ====================================================

            f.write(
                ",".join(row) + "\n"
            )


            packet_count += 1


            # ====================================================
            # Print statistics every 100 packets
            # ====================================================

            if packet_count % 100 == 0:

                elapsed = (
                    time.time()
                    - start
                )


                rate = (
                    packet_count
                    / elapsed
                    if elapsed > 0
                    else 0
                )


                print(
                    f"Packets: {packet_count} | "
                    f"Rate: {rate:.1f} Hz | "
                    f"Last CSI: {csi_len} | "
                    f"I/Q Samples: {len(csi_complex)}"
                )


            # ====================================================
            # Check file size
            # ====================================================

            size = os.path.getsize(
                FILE
            )


            if size >= MAX_SIZE:

                print(
                    "Buffer limit reached"
                )

                break


        except KeyboardInterrupt:

            print(
                "\nRecording stopped by user."
            )

            break


        except Exception as e:

            print(
                "ERROR:",
                e
            )


# ============================================================
# Close serial connection
# ============================================================

ser.close()

print(
    f"Recording finished. "
    f"Total packets: {packet_count}"
)
