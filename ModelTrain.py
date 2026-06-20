import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("wifi_csi_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6
)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

joblib.dump(model, "wifi_presence_model.pkl")