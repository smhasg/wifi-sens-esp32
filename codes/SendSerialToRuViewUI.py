import serial
import socket
import struct


# ============================================================
# CONFIGURATION
# ============================================================

SERIAL_PORT = "COM9"
BAUD_RATE = 921600

RUVIEW_IP = "127.0.0.1"
RUVIEW_PORT = 3333

MAGIC = 0xC5110001

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

def build_adr018_frame(
    timestamp,
    rssi,
    channel,
    csi
):

    # --------------------------------------------------------
    # Validate CSI
    # --------------------------------------------------------

    if not csi:

        return None


    if len(csi) % 2 != 0:

        print(
            f"[WARN] "
            f"CSI length must be even: "
            f"{len(csi)}"
        )

        return None


    # --------------------------------------------------------
    # CSI format
    #
    # [I0, Q0, I1, Q1, I2, Q2, ...]
    #
    # 256 CSI values
    # = 128 I/Q samples
    # --------------------------------------------------------

    num_subcarriers = len(csi) // 2


    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    frequency_mhz = channel_to_frequency(
        channel
    )


    # --------------------------------------------------------
    # Sequence
    #
    # Use timestamp as sequence
    # --------------------------------------------------------

    sequence = int(timestamp)


    # --------------------------------------------------------
    # ADR-018 Header
    # --------------------------------------------------------

    header = struct.pack(

        "<IBBHIIbbH",

        MAGIC,

        NODE_ID,

        NUM_ANTENNAS,

        num_subcarriers,

        frequency_mhz,

        sequence,

        max(
            -128,
            min(
                127,
                rssi
            )
        ),

        -90,

        0

    )


    # --------------------------------------------------------
    # I/Q PAYLOAD
    # --------------------------------------------------------

    iq_data = bytearray()


    for index in range(
        0,
        len(csi),
        2
    ):

        i_value = max(
            -128,
            min(
                127,
                int(csi[index])
            )
        )


        q_value = max(
            -128,
            min(
                127,
                int(csi[index + 1])
            )
        )


        # I

        iq_data.append(
            i_value & 0xFF
        )


        # Q

        iq_data.append(
            q_value & 0xFF
        )


    # --------------------------------------------------------
    # Final frame
    # --------------------------------------------------------

    frame = (

        header

        +

        bytes(iq_data)

    )


    return frame


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

            # ------------------------------------------------
            # Read serial line
            # ------------------------------------------------

            raw_line = ser.readline()


            if not raw_line:

                continue


            # ------------------------------------------------
            # Decode
            # ------------------------------------------------

            line = raw_line.decode(

                "utf-8",

                errors="ignore"

            ).strip()


            if not line:

                continue


            # ------------------------------------------------
            # Parse CSV
            #
            # timestamp
            # rssi
            # channel
            # csi_length
            # csi_0
            # csi_1
            # ...
            # ------------------------------------------------

            try:

                values = [

                    int(x.strip())

                    for x in line.split(",")

                    if x.strip() != ""

                ]


            except ValueError:

                print(

                    "[WARN] Invalid CSV:",

                    line[:200]

                )

                continue


            # ------------------------------------------------
            # Validate minimum fields
            # ------------------------------------------------

            if len(values) < 5:

                print(

                    "[WARN] Not enough fields:",

                    len(values)

                )

                continue


            # ------------------------------------------------
            # Extract metadata
            # ------------------------------------------------

            timestamp = values[0]

            rssi = values[1]

            channel = values[2]

            csi_length = values[3]


            # ------------------------------------------------
            # Extract CSI
            # ------------------------------------------------

            csi = values[4:]


            # ------------------------------------------------
            # Validate CSI length
            # ------------------------------------------------

            if len(csi) != csi_length:

                print(

                    f"[WARN] CSI length mismatch: "

                    f"header={csi_length} "

                    f"actual={len(csi)}"

                )

                continue


            # ------------------------------------------------
            # Build ADR-018
            # ------------------------------------------------

            frame = build_adr018_frame(

                timestamp,

                rssi,

                channel,

                csi

            )


            if frame is None:

                continue


            # ------------------------------------------------
            # Send UDP
            # ------------------------------------------------

            try:

                sent_bytes = sock.sendto(

                    frame,

                    (

                        RUVIEW_IP,

                        RUVIEW_PORT

                    )

                )


            except OSError as e:

                print(

                    f"[UDP ERROR] {e}"

                )

                continue


            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            print(

                f"[OK] "

                f"frame={frame_counter} "

                f"timestamp={timestamp} "

                f"rssi={rssi} "

                f"channel={channel} "

                f"csi={len(csi)} "

                f"subcarriers={len(csi) // 2} "

                f"bytes={sent_bytes}"

            )


            frame_counter += 1


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