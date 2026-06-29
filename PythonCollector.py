import serial
import os

PORT = "COM9"           # Linux: /dev/ttyACM0
BAUD = 921600

FILE = "csi_buffer.csv"

MAX_SIZE = 24 * 1024 * 1024

ser = serial.Serial(PORT, BAUD)

if not os.path.exists(FILE):
    with open(FILE, "w") as f:
        f.write("raw\n")

while True:

    line = ser.readline().decode(
        errors="ignore").strip()

    with open(FILE, "a") as f:
        f.write(line + "\n")

    size = os.path.getsize(FILE)

    if size >= MAX_SIZE:

        print("Recreating buffer...")

        os.remove(FILE)

        with open(FILE, "w") as f:
            f.write("raw\n")