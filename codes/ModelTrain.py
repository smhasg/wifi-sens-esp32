import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import pandas as pd
import csv
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


from pathlib import Path
import numpy as np

DATA_DIR = Path("../Data")

X_list = []
y_list = []

for file in sorted(DATA_DIR.rglob("*.npz")):

    print(f"Loading {file.name}")

    data = np.load(file)

    # CSI complex64 -> magnitude float32
    csi = np.abs(data["csi"]).astype(np.float32)

    # metadata
    meta = np.column_stack([
        data["rssi"].astype(np.float32),
        data["channel"].astype(np.float32),
        data["csi_len"].astype(np.float32),
    ])

    # X = [rssi, channel, csi_len, cs0...cs127]
    X_file = np.hstack([
        meta,
        csi
    ])

    label = file.stem.split("_")[0]

    y_file = np.full(
        X_file.shape[0],
        label
    )

    X_list.append(X_file)
    y_list.append(y_file)


# merge all files
X = np.vstack(X_list)
y = np.concatenate(y_list)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("Memory:", X.nbytes / 1024**2, "MB")

zero_cols = np.where(np.all(np.abs(X) == 0, axis=0))[0]
print("Removed:", len(zero_cols))
X = np.delete(X, zero_cols, axis=1)

print(X.shape)

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

y=encoder.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


rf=RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train,y_train)

pred=rf.predict(X_test)

print(classification_report(y_test,pred))



model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train)

print("accuracy:", model.score(X_test, y_test))

joblib.dump(model, "model/model.pkl")