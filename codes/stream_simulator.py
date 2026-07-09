import pandas as pd
import time

df = pd.read_csv("data/wifi_csi_dataset.csv")

def stream(window_size=20):
    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size].drop("label", axis=1).values
        label = df.iloc[i]["label"]
        yield window, label
        time.sleep(0.1)