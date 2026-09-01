import serial
import socket
import struct
import time
import sys
import math
import hashlib

# تنظیمات
COM_PORT = 'COM9'
BAUD_RATE = 921600
UDP_IP = '127.0.0.1'
UDP_PORT = 3333

# هدر ADR-018
CSI_MAGIC_V1 = 0xC5110001
CSI_HEADER_SIZE = 20


def parse_csi_line(line):
    try:
        # حذف کاراکترهای اضافی
        line = line.strip()
        if not line:
            return None

        # جداسازی با کاما
        parts = line.split(',')
        if len(parts) < 4:
            return None

        # استخراج فیلدها
        timestamp = int(parts[0])
        rssi = int(parts[1])
        channel = int(parts[2])
        data_len = int(parts[3])

        # داده‌های I/Q (بعد از ۴ فیلد اول)
        iq_bytes = []

        for i in range(4, min(4 + data_len, len(parts))):
            try:
                val = int(parts[i])

                # محدود کردن به محدوده i8
                if val > 127:
                    val = 127
                elif val < -128:
                    val = -128

                iq_bytes.append(val)

            except ValueError:
                continue

        # باید زوج باشه (I و Q)
        if len(iq_bytes) % 2 != 0:
            iq_bytes = iq_bytes[:-1]

        if len(iq_bytes) < 2:
            return None

        return {
            'timestamp': timestamp,
            'rssi': rssi,
            'channel': channel,
            'node_id': 1,  # ID ثابت
            'n_antennas': 1,
            'n_subcarriers': len(iq_bytes) // 2,
            'noise_floor': -90,  # مقدار پیش‌فرض
            'iq_data': iq_bytes
        }

    except Exception as e:
        return None


def build_adr018_frame(csi_data):
    magic = CSI_MAGIC_V1
    node_id = csi_data['node_id']
    n_antennas = csi_data['n_antennas']
    n_subcarriers = csi_data['n_subcarriers']
    channel = csi_data['channel']
    rssi = csi_data['rssi']
    noise_floor = csi_data['noise_floor']
    timestamp_us = csi_data['timestamp']
    iq_data = csi_data['iq_data']

    iq_len = len(iq_data)
    buf = bytearray(CSI_HEADER_SIZE + iq_len)

    # Magic (0-3)
    struct.pack_into('<I', buf, 0, magic)

    # Node ID (4)
    buf[4] = node_id

    # Number of antennas (5)
    buf[5] = n_antennas

    # Number of subcarriers (6-7)
    struct.pack_into('<H', buf, 6, n_subcarriers)

    # Channel (8)
    buf[8] = channel

    # RSSI (9) - تبدیل به unsigned
    buf[9] = rssi & 0xFF

    # Noise floor (10)
    buf[10] = noise_floor & 0xFF

    # Reserved (11-15) - صفر

    # Timestamp (16-19)
    struct.pack_into('<I', buf, 16, timestamp_us)

    for i, val in enumerate(iq_data):
        buf[CSI_HEADER_SIZE + i] = val & 0xFF

    return bytes(buf)


def main():
    print(f"🔌 connecting to {COM_PORT} @ {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"❌ Error in opening Port: {e}")
        return

    print(f"✅ UDP Sending on  {UDP_IP}:{UDP_PORT}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    frame_count = 0
    error_count = 0

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode(
                    'utf-8',
                    errors='ignore'
                ).strip()

                if line:
                    # پردازش خط
                    csi_data = parse_csi_line(line)

                    # اگر parser نتونست packet را parse کند
                    if csi_data is None:
                        error_count += 1

                        if error_count % 10 == 0:
                            print("⚠️ Error in processing raw data")
                            print("RAW LINE:")
                            print(line)

                        continue

                    print("RAW CSI:")
                    print(csi_data)

                    # -----------------------------------------
                    # بررسی CSI واقعی
                    # -----------------------------------------

                    iq_data = csi_data["iq_data"]

                    print("n_subcarriers:", csi_data["n_subcarriers"])
                    print("len(iq_data):", len(iq_data))
                    print("iq_data[:30]:", iq_data[:30])

                    # بررسی نوع داده‌ها
                    iq_types = set(type(x).__name__ for x in iq_data)
                    print("iq_data types:", iq_types)

                    # -----------------------------------------
                    # MD5 درست:
                    # فقط خود IQ data را hash می‌کنیم
                    # -----------------------------------------

                    try:
                        iq_bytes_for_hash = bytes(
                            (x & 0xFF) for x in iq_data
                        )

                        iq_hash = hashlib.md5(
                            iq_bytes_for_hash
                        ).hexdigest()

                        print("IQ MD5:", iq_hash)

                    except Exception as e:
                        print(f"⚠️ Error calculating IQ MD5: {e}")

                    # -----------------------------------------
                    # محاسبه AMP برای بررسی
                    # -----------------------------------------

                    amps = []

                    for i in range(0, len(iq_data) - 1, 2):
                        I = iq_data[i]
                        Q = iq_data[i + 1]

                        amp = math.sqrt(
                            I * I +
                            Q * Q
                        )

                        amps.append(amp)

                    print("AMP first 30:")
                    print([
                        round(x, 2)
                        for x in amps[:30]
                    ])

                    # -----------------------------------------
                    # ساخت و ارسال فریم ADR-018
                    # -----------------------------------------

                    if csi_data['n_subcarriers'] > 0:

                        # ساخت فریم باینری
                        frame = build_adr018_frame(csi_data)

                        # ارسال به UDP
                        try:
                            sock.sendto(
                                frame,
                                (UDP_IP, UDP_PORT)
                            )

                            frame_count += 1

                            if frame_count % 50 == 0:
                                print(
                                    f"📤 send {frame_count} frame - "
                                    f"RSSI: {csi_data['rssi']}, "
                                    f"Subcarriers: "
                                    f"{csi_data['n_subcarriers']}"
                                )

                        except Exception as e:
                            error_count += 1

                            if error_count % 10 == 0:
                                print(
                                    f"⚠️ Error in sending UDP: {e}"
                                )

                    else:
                        error_count += 1

                        if error_count % 10 == 0:
                            print(
                                "⚠️ Error in processing raw data"
                            )

            time.sleep(0.001)

    except KeyboardInterrupt:
        print(
            f"\n⏹️ paused. "
            f"{frame_count} frames sended."
        )

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        ser.close()
        sock.close()


if __name__ == "__main__":
    main()