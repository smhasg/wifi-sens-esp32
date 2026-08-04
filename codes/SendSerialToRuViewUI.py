import serial
import socket
import struct
import time
import random



# ============================================================
# CONFIGURATION
# ============================================================

SERIAL_PORT = "COM9"
BAUD_RATE = 921600

RUVIEW_IP = "127.0.0.1"
RUVIEW_PORT = 3333

CSI_MAGIC_V6 = 0xC5110006
# MAGIC = 0xC5110001

NODE_ID = 1

NUM_ANTENNAS = 1


# ============================================================
# CHANNEL -> FREQUENCY
# ============================================================

def channel_to_frequency(channel):

    channel = int(channel)

    if 1 <= channel <= 13:

        return 2407 + (
            channel * 5
        )

    if channel == 14:

        return 2484

    if 36 <= channel <= 165:

        return 5000 + (
            channel * 5
        )

    return 2437


# ============================================================
# CREATE ADR-018 FRAME
# ============================================================


def build_adr018(
        node_id=1,
        n_antennas=1,
        n_subcarriers=64,
        channel=6,
        rssi=-40,
        noise_floor=-90,
):

    timestamp_us = int(time.time() * 1_000_000)


    # Generate fake IQ
    iq = []

    for _ in range(n_subcarriers):
        iq.append(random.randint(-50, 50))  # I
        iq.append(random.randint(-50, 50))  # Q


    packet = bytearray()


    # 0-3 MAGIC
    packet += struct.pack(
        "<I",
        CSI_MAGIC_V6
    )


    # 4 node_id
    packet += struct.pack(
        "B",
        node_id
    )


    # 5 antennas
    packet += struct.pack(
        "B",
        n_antennas
    )


    # 6-7 subcarriers
    packet += struct.pack(
        "<H",
        n_subcarriers
    )


    # 8 channel
    packet += struct.pack(
        "B",
        channel
    )


    # 9 rssi
    packet += struct.pack(
        "b",
        rssi
    )


    # 10 noise floor
    packet += struct.pack(
        "b",
        noise_floor
    )


    # 11-15 reserved
    packet += bytes(5)


    # 16-19 timestamp
    packet += struct.pack(
        "<I",
        timestamp_us & 0xffffffff
    )


    # IQ payload
    packet += struct.pack(
        f"{len(iq)}b",
        *iq
    )


    return packet



# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"[INFO] Opening serial port "
        f"{SERIAL_PORT}"
    )


    ser = serial.Serial(

        port=SERIAL_PORT,

        baudrate=BAUD_RATE,

        timeout=1

    )


    print(

        f"[INFO] Serial connected: "

        f"{SERIAL_PORT} @ {BAUD_RATE}"

    )


    print(

        f"[INFO] Sending CSI to "

        f"{RUVIEW_IP}:{RUVIEW_PORT}"

    )


    # --------------------------------------------------------
    # UDP Socket
    # --------------------------------------------------------

    sock = socket.socket(

        socket.AF_INET,

        socket.SOCK_DGRAM

    )


    frame_counter = 0


    try:

        while True:

            data = build_adr018()

            sock.sendto(
                data,
                (
                    RUVIEW_IP,
                    RUVIEW_PORT
                )
            )


            print(
                f"sent {len(data)} bytes"
            )
            
            frame_counter += 1
            time.sleep(0.05)


    except KeyboardInterrupt:

        print(

            "\n[INFO] Stopping..."

        )


    finally:

        ser.close()

        sock.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()


