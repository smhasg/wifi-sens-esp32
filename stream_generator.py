import pandas as pd
import time

df = pd.read_csv("wifi_csi_dataset.csv")

while True:

    for _, row in df.iterrows():

        sample = row[:-1].values

        yield sample

        time.sleep(0.1)