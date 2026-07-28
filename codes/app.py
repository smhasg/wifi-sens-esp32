import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

import joblib
import os
import time


from serial_reader import SerialReader

import struct
import socket


class RuViewSender:

    MAGIC = 0xC5110001

    def __init__(
        self,
        ruview_ip,
        ruview_port=5005,
        node_id=1
    ):

        self.ruview_ip = ruview_ip
        self.ruview_port = ruview_port
        self.node_id = node_id

        self.sequence = 0

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

    def channel_to_frequency(
        self,
        channel
    ):

        return {
            1: 2412,
            2: 2417,
            3: 2422,
            4: 2427,
            5: 2432,
            6: 2437,
            7: 2442,
            8: 2447,
            9: 2452,
            10: 2457,
            11: 2462,
            12: 2467,
            13: 2472,
            14: 2484,
        }.get(
            int(channel),
            2437
        )

    def send(
        self,
        packet
    ):

        csi = np.asarray(
            packet["csi"],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is only a temporary mapping.
        #
        # If your CSI is amplitude-only,
        # this does NOT reconstruct real I/Q.
        # ----------------------------------------------------

        csi = np.clip(
            csi,
            -127,
            127
        ).astype(
            np.int8
        )

        num_subcarriers = len(
            csi
        )

        # Create fake imaginary component
        imag = np.zeros_like(
            csi
        )

        iq = np.empty(
            num_subcarriers * 2,
            dtype=np.int8
        )

        iq[0::2] = csi
        iq[1::2] = imag

        rssi = int(
            np.clip(
                packet.get(
                    "rssi",
                    -50
                ),
                -128,
                127
            )
        )

        channel = int(
            packet.get(
                "channel",
                6
            )
        )

        frequency = self.channel_to_frequency(
            channel
        )

        # ADR-018
        header = struct.pack(
            "<IBBHIIbbH",

            self.MAGIC,

            self.node_id,

            1,

            num_subcarriers,

            frequency,

            self.sequence,

            rssi,

            -90,

            0
        )

        data = (
            header
            + iq.tobytes()
        )

        self.sock.sendto(
            data,
            (
                self.ruview_ip,
                self.ruview_port
            )
        )

        self.sequence += 1



# =====================
# CONFIG
# =====================


st.set_page_config(
    page_title="ESP32 CSI Radar",
    layout="wide"
)
st.title(
    "📡 ESP32-S3 CSI Human Sensing"
)

# =====================
# SERIAL START
# =====================

if "reader" not in st.session_state:

    reader=SerialReader(
        "COM9",
        921600,
        2000
    )

    reader.start()

    st.session_state.reader=reader

reader=st.session_state.reader

# =====================
# LOAD MODEL
# =====================
MODEL_PATH="xgb_csi_model.pkl"
model=None

if os.path.exists(MODEL_PATH):

    model=joblib.load(
        MODEL_PATH
    )
    st.sidebar.success(
        "🟢 XGBoost model loaded"
    )

else:

    st.sidebar.warning(
        "🟡 Running without AI model"
    )




########

# ---------------------------------------------------------
# Initialize RuView sender
# ---------------------------------------------------------

if "ruview_sender" not in st.session_state:

    st.session_state.ruview_sender = RuViewSender(

        ruview_ip="192.168.1.100",

        ruview_port=5005,

        node_id=1
    )


# ---------------------------------------------------------
# Read your ESP32 CSI
# ---------------------------------------------------------

packets = reader.get_data()


if len(packets) < 30:

    st.warning(
        "Waiting for CSI packets..."
    )

    time.sleep(
        2
    )

    st.rerun()


# ---------------------------------------------------------
# Send packets to RuView
# ---------------------------------------------------------

for packet in packets:

    try:

        st.session_state.ruview_sender.send(
            packet
        )

    except Exception as e:

        st.error(
            f"RuView send error: {e}"
        )


# ---------------------------------------------------------
# Your existing Streamlit processing
# ---------------------------------------------------------

rows = []

for p in packets:

    rows.append(

        [
            p["rssi"],
            p["channel"],
            p["csi_len"],
            *p["csi"]
        ]

    )


columns = [

    "rssi",
    "channel",
    "csi_len"

]


columns += [

    f"cs{i}"

    for i in range(256)

]


df = pd.DataFrame(

    rows,

    columns=columns

)

########


# =====================
# GET CSI DATA
# =====================
packets=reader.get_data()
if len(packets)<30:
    st.warning(
        "Waiting for CSI packets..."
    )
    time.sleep(2)
    st.rerun()
rows=[]

for p in packets:

    rows.append(

        [
            p["rssi"],
            p["channel"],
            p["csi_len"],
            *p["csi"]
        ]

    )

columns=[

    "rssi",
    "channel",
    "csi_len"

]


columns += [

    f"cs{i}"
    for i in range(256)

]



df=pd.DataFrame(
    rows,
    columns=columns
)



# =====================
# HEADER METRICS
# =====================


c1,c2,c3,c4=st.columns(4)



c1.metric(
    "Packets",
    len(df)
)


c2.metric(
    "RSSI",
    round(
        df.rssi.iloc[-1],
        2
    )
)


c3.metric(
    "CSI Length",
    df.csi_len.iloc[-1]
)



# =====================
# MODEL PREDICTION
# =====================


if model:


    sample=df.tail(1)


    pred=model.predict(
        sample
    )[0]


    st.subheader(
        "👥 People Detection"
    )


    st.metric(
        "Detected People",
        int(pred)
    )


else:

    st.info(
        "AI prediction disabled"
    )





# =====================
# FEATURES
# =====================


csi_cols=[

    f"cs{i}"
    for i in range(256)

]



csi=df[csi_cols]



# energy

energy=np.sum(
    np.square(
        csi.values
    ),
    axis=1
)


# variance

variance=np.var(
    csi.values,
    axis=1
)



# =====================
# VISUALIZATION MENU
# =====================



mode=st.sidebar.selectbox(

    "Visualization",

    [

        "CSI Waterfall",
        "CSI Energy",
        "CSI Variance",
        "RSSI",
        "Single Subcarrier",
        "Normalized CSI"

    ]

)



# =====================
# CSI WATERFALL
# =====================


if mode=="CSI Waterfall":


    matrix=csi.values.T



    fig=go.Figure(

        data=go.Heatmap(

            z=matrix

        )

    )


    fig.update_layout(

        height=700,

        title="CSI Waterfall"

    )


    st.plotly_chart(
    fig,
    width="stretch"
)




# =====================
# ENERGY
# =====================


elif mode=="CSI Energy":


    fig=go.Figure()


    fig.add_trace(

        go.Scatter(
            y=energy,
            mode="lines"
        )

    )


    fig.update_layout(
        title="CSI Energy / Motion",
        height=400
    )


    st.plotly_chart(
    fig,
    width="stretch"
)




# =====================
# VARIANCE
# =====================


elif mode=="CSI Variance":


    fig=go.Figure()


    fig.add_trace(

        go.Scatter(
            y=variance,
            mode="lines"
        )

    )


    fig.update_layout(
        title="CSI Variance",
        height=400
    )


    st.plotly_chart(
    fig,
    width="stretch"
)




# =====================
# RSSI
# =====================


elif mode=="RSSI":


    fig=go.Figure()


    fig.add_trace(

        go.Scatter(

            y=df.rssi,

            mode="lines"

        )

    )


    fig.update_layout(
        title="RSSI",
        height=400
    )


    st.plotly_chart(
    fig,
    width="stretch"
)




# =====================
# SINGLE CSI
# =====================


elif mode=="Single Subcarrier":


    cs=st.sidebar.slider(
        "Subcarrier",
        0,
        255,
        0
    )


    fig=go.Figure()


    fig.add_trace(

        go.Scatter(

            y=df[f"cs{cs}"],

            mode="lines"

        )

    )


    fig.update_layout(
        title=f"CSI {cs}",
        height=400
    )


    st.plotly_chart(
        fig
    )





# =====================
# NORMALIZED
# =====================


elif mode=="Normalized CSI":


    x=MinMaxScaler().fit_transform(
        csi
    )


    fig=go.Figure(

        go.Heatmap(
            z=x.T
        )

    )


    fig.update_layout(
        height=700
    )


    st.plotly_chart(
    fig,
    width="stretch"
)



# refresh

time.sleep(2)

st.rerun()