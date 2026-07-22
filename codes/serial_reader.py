import serial
import threading
from collections import deque


class SerialReader:


    def __init__(
        self,
        port="COM9",
        baud=921600,
        max_buffer=693
    ):

        self.port = port
        self.baud = baud

        self.buffer = deque(
            maxlen=max_buffer
        )

        self.running = False
        self.thread = None



    def start(self):

        if self.running:
            return

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



                values=line.split(",")


                if len(values)<5:
                    continue



                try:

                    packet={

                        "esp_time":
                            values[0],

                        "rssi":
                            float(values[1]),

                        "channel":
                            int(values[2]),

                        "csi_len":
                            int(values[3]),

                        "csi":
                            [
                                float(x)
                                for x in values[4:260]
                            ]

                    }


                    if len(packet["csi"]) < 256:

                        packet["csi"] += [
                            0
                        ] * (
                            256-len(packet["csi"])
                        )


                    self.buffer.append(packet)



                except:

                    continue



        except Exception as e:

            print(
                "Serial Error:",
                e
            )



    def get_data(self):

        return list(self.buffer)