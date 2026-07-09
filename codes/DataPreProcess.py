import pandas as pd 
import os 


from pathlib import Path

current_dir = str(Path.cwd())
os.chdir("C:\Users\smhas\Desktop\working\wifi-sens")
clean = []

with open("csi_buffer.csv") as f:
    for line in f:
        parts = line.strip().split(",")

        if len(parts) == 261:
            clean.append(parts)

df = pd.DataFrame(clean)


df.head()

time_col = df.columns[0]
feature_cols = df.columns[1:]

X = df[feature_cols].astype(float)
