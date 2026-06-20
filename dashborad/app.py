import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

model = joblib.load("wifi_presence_model.pkl")

st.title("WiFi CSI Human Detection")

placeholder = st.empty()

while True:

    sample = np.random.normal(-50, 5, 64)

    pred = model.predict(sample.reshape(1, -1))[0]

    fig, ax = plt.subplots()

    heat = np.tile(sample, (20,1))

    ax.imshow(
        heat.T,
        aspect="auto"
    )

    placeholder.pyplot(fig)

    if pred == 0:
        st.success("Room Empty")
    else:
        st.error("Human Detected")