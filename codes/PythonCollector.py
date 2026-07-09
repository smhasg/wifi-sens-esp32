import serial
import os
import time
from datetime import datetime

PORT = "COM9"
BAUD = 921600

DATA_DIR = "..\\Data"
FILE = os.path.join(DATA_DIR, "OnePerson-sitting.csv")

MAX_SIZE = 100 * 1024 * 1024

os.makedirs(DATA_DIR, exist_ok=True)

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)


HEADER = [
    "pc_time",
    "esp_time",
    "rssi",
    "channel",
    "csi_len"
]

HEADER += [
    f"cs{i}"
    for i in range(256)
]


# create file
if not os.path.exists(FILE):

    with open(FILE, "w") as f:
        f.write(",".join(HEADER) + "\n")


print("Recording CSI...")

with open(FILE, "a", buffering=1) as f:

    packet_count = 0
    start = time.time()

    while True:

        try:

            line = ser.readline().decode(
                errors="ignore"
            ).strip()


            if not line:
                continue

            values = line.split(",")

            # expected:
            # esp_time,rssi,channel,len,csi...

            if len(values) < 4:
                continue

            esp_time = values[0]
            rssi = values[1]
            channel = values[2]
            csi_len = int(values[3])

            csi = values[4:]

            # normalize CSI length
            if len(csi) > 256:
                csi = csi[:256]

            elif len(csi) < 256:
                csi += [
                    "0"
                ] * (256 - len(csi))

            pc_time = datetime.now().isoformat(
                timespec="milliseconds"
            )

            row = [
                pc_time,
                esp_time,
                rssi,
                channel,
                str(csi_len)
            ]

            row += csi

            f.write(
                ",".join(row) + "\n"
            )

            packet_count += 1


            if packet_count % 100 == 0:

                elapsed = time.time() - start

                rate = packet_count / elapsed

                print(
                    f"Packets: {packet_count} | "
                    f"Rate: {rate:.1f} Hz | "
                    f"Last CSI: {csi_len}"
                )

            size = os.path.getsize(FILE)

            if size >= MAX_SIZE:

                print(
                    "Buffer limit reached"
                )

                break

        except Exception as e:

            print(
                "ERROR:",
                e
            )