import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from stream_simulator import stream

model = joblib.load("model/model.pkl")

st.title("📡 WiFi CSI Sensing Dashboard")

# UI placeholders
heatmap_slot = st.empty()
plot_slot = st.empty()

people_metric = st.metric("Detected People", 0)
status = st.empty()

for window, true_label in stream():

    # flatten feature
    x = window[-1].reshape(1, -1)

    pred = model.predict(x)[0]

    # -------------------------
    # Heatmap
    # -------------------------
    fig, ax = plt.subplots()
    ax.imshow(window.T, aspect='auto', cmap='jet')
    ax.set_title("CSI Heatmap (Live)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Subcarrier")

    heatmap_slot.pyplot(fig)

    # -------------------------
    # waveform
    # -------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(window[-1])
    ax2.set_title("Latest CSI Packet")

    plot_slot.pyplot(fig2)

    # -------------------------
    # prediction logic
    # -------------------------
    people_metric.metric(
        "Detected People",
        int(pred)
    )

    if pred == 0:
        status.success("Empty Room")
    else:
        status.error("Human Detected")