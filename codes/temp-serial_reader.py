import serial

PORT = "COM9"
BAUDRATE = 921600

ser = serial.Serial(
    PORT,
    BAUDRATE,
    timeout=1
)

print(f"Listening on {PORT} @ {BAUDRATE}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        line = ser.readline()

        if line:
            print(repr(line))

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    ser.close()