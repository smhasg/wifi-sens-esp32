import os
import joblib

from config import RF_MODEL
from config import XGB_MODEL


class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = "None"
        self.zero_cols = None
        self.n_features = None
        self.features = []

    def load(self, name):
        self.zero_cols = None
        self.model = None
        self.model_name = name
        self.n_features = None
        self.features = []

        if name == "None":
            return

        if name == "Random Forest":
            print("Exists:", os.path.exists(XGB_MODEL))
            if os.path.exists(RF_MODEL):
                package = joblib.load(RF_MODEL)
                model = package["model"]
                zero_cols = package["zero_cols"]
                n_features = package["n_features"]
                self.model = model 
                self.zero_cols = zero_cols    
                self.n_features = n_features
                self.features = package["features"]

        elif name == "XGBoost":
            print("Exists:", os.path.exists(XGB_MODEL))
            if os.path.exists(XGB_MODEL):
                package = joblib.load(XGB_MODEL)
                self.model = package["model"]
                self.zero_cols = package["zero_cols"]
                self.n_features = package["n_features"]
                self.features = package["features"]


    def predict(self, x):

        if self.model is None:
            return None
            
        try:
            return self.model.predict(x)[0]

        except Exception as e:
            print(e)
            raise