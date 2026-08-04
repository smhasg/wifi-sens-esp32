import streamlit as st
import pandas as pd
import numpy as np
import time

# =========================
# Local Modules
# =========================

from config import (
    SERIAL_PORT,
    BAUD_RATE,
    BUFFER_SIZE
)

from serial_reader import SerialReader
from feature_extractor import FeatureExtractor
from model_manager import ModelManager
from utils import (
    compute_energy,
    compute_variance
)

from plots import (
    live_csi,
    waterfall,
    rssi
)

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="ESP32 CSI Radar",
    layout="wide"
)

st.title(
    "📡 ESP32-S3 CSI Human Sensing Dashboard"
)

# =========================
# SESSION STATE INIT
# =========================


if "reader" not in st.session_state:

    st.info(
        "Starting ESP32 Serial Reader..."
    )

    reader = SerialReader(
        SERIAL_PORT,
        BAUD_RATE,
        BUFFER_SIZE
    )

    reader.start()


    st.session_state.reader = reader



if "model_manager" not in st.session_state:

    st.session_state.model_manager = ModelManager()


reader = st.session_state.reader
model_manager = st.session_state.model_manager

# =========================
# SIDEBAR
# =========================


st.sidebar.header(
    "⚙ Settings"
)


model_choice = st.sidebar.radio(

    "AI Model",

    [

        "None",

        "Random Forest",

        "XGBoost"

    ]

)



visualization = st.sidebar.selectbox(

    "Visualization",

    [

        "Live CSI",

        "Waterfall",

        "RSSI",

        "Energy",

        "Variance"

    ]

)



# =========================
# MODEL LOAD
# =========================


if (model_manager.model_name!=model_choice):
    model_manager.load(model_choice)

    # print(type(model_manager))
    # print (model_manager['model'])
    # print (model_manager['zero_cols'])
    # print (model_manager['n_features'])


# =========================
# READ ESP32 DATA
# =========================


packets = reader.get_data()



if packets is None or len(packets) == 0:


    st.warning(
        "Waiting for ESP32 CSI packets..."
    )


    time.sleep(1)


    st.rerun()



# =========================
# CREATE DATAFRAME
# =========================


df = FeatureExtractor.packet_to_dataframe(
    packets
)



# =========================
# BASIC CHECK
# =========================


if len(df) == 0:
    st.warning(
        "No valid CSI data"
    )
    st.stop()


# =========================
# METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Packets",
        len(df)
    )

with col2:
    st.metric(
        "RSSI",
        round(
            float(df.rssi.iloc[-1]),
            2
        )
    )

with col3:
    st.metric(
        "Channel",
        int(
            df.channel.iloc[-1]
        )
    )

with col4:
    st.metric(
        "CSI Length",
        int(df.shape[1])
    )
    # st.write(df.shape)
# =========================
# FEATURE CALCULATION
# =========================

energy = compute_energy(df)
variance = compute_variance(df)
st.divider()

# =========================
# PLACE HOLDERS
# =========================

# left, right = st.columns(
#     [2,1]
# )

# with left:


# with right:
    # st.subheader(
    #     "🤖 Prediction"
    # )


# =========================
# MODEL PREDICTION
# =========================

st.divider()

# with right:
st.subheader(
        "🤖 Prediction"
    )
if model_choice == "None":
    st.info(
        "AI Model Disabled"
    )

else:
    try:
        sample = df.tail(100).copy()
        if model_manager.zero_cols:
            sample = sample.drop(   
                columns=list(model_manager.zero_cols),
                errors="ignore"
            )
        st.write(f"Input Shape : {sample.shape}")
        sample = sample[model_manager.features]
        st.write(f"expected Shape :{sample.shape}")
        if sample.shape[1] != model_manager.n_features:
            raise ValueError(
                f"Feature mismatch: {sample.shape[0]} != {model_manager.n_features}"
            )
        st.write(f"sample :{sample.iloc[0]}")
        prediction = model_manager.predict(sample)
        st.success(
            f"Prediction: {prediction}"
        )

    except Exception as e:
        st.error(
            f"Prediction Error: {e}"
        )

# =========================
# VISUALIZATION
# =========================

st.divider()
st.subheader(
        "📈 Live Signal"
    )

if visualization == "Live CSI":
    fig = live_csi(
        df
    )
    st.plotly_chart(
        fig,
        width="stretch"
    )


elif visualization == "Waterfall":
    fig = waterfall(
        df
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

elif visualization == "RSSI":

    fig = rssi(
        df
    )

    st.plotly_chart(

        fig,
        width="stretch"

    )


elif visualization == "Energy":
    fig = st.line_chart(
        energy
    )

elif visualization == "Variance":
    fig = st.line_chart(
        variance
    )

# =========================
# EXTRA LIVE VIEW
# =========================

st.divider()
st.subheader(
    "📡 Latest CSI Packet"
)

csi_cols = [

    f"cs{i}"

    for i in range(128)

]

latest_csi = df.iloc[-1][csi_cols]

st.line_chart(
    latest_csi
)

# =========================
# AUTO REFRESH
# =========================

time.sleep(
    3
)


st.rerun()

