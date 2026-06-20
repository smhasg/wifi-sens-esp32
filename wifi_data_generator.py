import numpy as np
import pandas as pd

N = 10000

empty = np.random.normal(
    loc=-55,
    scale=2,
    size=(N//2, 64)
)

human = np.random.normal(
    loc=-48,
    scale=6,
    size=(N//2, 64)
)

X = np.vstack([empty, human])

y = np.concatenate([
    np.zeros(N//2),
    np.ones(N//2)
])

df = pd.DataFrame(
    X,
    columns=[f"sc{i}" for i in range(64)]
)

df["label"] = y

df.to_csv("wifi_csi_dataset.csv", index=False)