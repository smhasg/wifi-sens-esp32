import serial
import threading
import time
from collections import deque


class SerialReader:

    def __init__(
        self,
        port="COM9",
        baud=921600,
        max_buffer=200
    ):

        self.port = port
        self.baud = baud

        self.buffer = deque(
            maxlen=max_buffer
        )

        self.running = False
        self.thread = None


    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self.read_loop,
            daemon=True
        )

        self.thread.start()



    def read_loop(self):

        try:

            ser = serial.Serial(
                self.port,
                self.baud,
                timeout=1
            )


            while self.running:


                line = ser.readline().decode(
                    errors="ignore"
                ).strip()


                if not line:
                    continue


                values = line.split(",")


                if len(values) < 5:
                    continue


                esp_time = values[0]
                rssi = float(values[1])
                channel = int(values[2])
                csi_len = int(values[3])


                csi = values[4:]


                csi = [
                    float(x)
                    for x in csi[:256]
                ]


                if len(csi)<256:

                    csi += [
                        0
                    ]*(256-len(csi))


                packet = {

                    "esp_time":esp_time,
                    "rssi":rssi,
                    "channel":channel,
                    "csi_len":csi_len,
                    "csi":csi

                }


                self.buffer.append(packet)


        except Exception as e:

            print(
                "Serial error:",
                e
            )


    def get_data(self):

        return list(self.buffer)