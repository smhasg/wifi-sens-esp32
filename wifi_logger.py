import subprocess
import re
import csv
import time
from datetime import datetime

with open("wifi_log.csv", "a", newline="") as f:

    writer = csv.writer(f)

    if f.tell() == 0:
        writer.writerow([
            "timestamp",
            "signal_percent"
        ])

    while True:

        output = subprocess.check_output(
            "netsh wlan show interfaces",
            shell=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        m = re.search(r"Signal\s*:\s*(\d+)%", output)

        if m:

            signal = int(m.group(1))

            writer.writerow([
                datetime.now().isoformat(),
                signal
            ])

            print(signal)

            f.flush()

        time.sleep(1)