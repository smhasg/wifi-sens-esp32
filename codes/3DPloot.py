import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="WiFi CSI Dashboard",
    layout="wide"
)

###########################################################
# Settings
###########################################################

CSV_FILE = "csi_buffer.csv"

REFRESH_RATE = 0.2          # seconds
MAX_PACKETS = 200           # آخرین Packetها

###########################################################
# Sidebar
###########################################################

st.sidebar.title("Settings")

MAX_PACKETS = st.sidebar.slider(
    "Packets",
    50,
    1000,
    200
)

REFRESH_RATE = st.sidebar.slider(
    "Refresh",
    0.05,
    2.0,
    0.2
)

###########################################################
# Placeholders
###########################################################

metric_placeholder = st.empty()

surface_placeholder = st.empty()

heatmap_placeholder = st.empty()

waterfall_placeholder = st.empty()

###########################################################
# Main Loop
###########################################################

while True:

    try:
        clean = []
        with open("csi_buffer.csv") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) == 261:
                    clean.append(parts)

        df = pd.DataFrame(clean)

        ###################################################
        # پیدا کردن ستون‌های CSI
        ###################################################

        ignore = [
            "timestamp",
            "mac",
            "rssi",
            "channel",
            "rate",
            "sig_mode",
            "mcs",
            "bandwidth"
        ]

        csi_columns = [
            c for c in df.columns
            if c not in ignore
        ]

        csi = df[csi_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        csi = csi.tail(MAX_PACKETS)

        data = csi.to_numpy()

        ###################################################
        # Metrics
        ###################################################

        with metric_placeholder.container():

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Packets",
                len(df)
            )

            c2.metric(
                "Displayed",
                len(csi)
            )

            c3.metric(
                "Subcarriers",
                data.shape[1]
            )

            c4.metric(
                "Mean",
                f"{np.nanmean(data):.2f}"
            )

        ###################################################
        # 3D Surface
        ###################################################

        fig_surface = go.Figure(
            data=[
                go.Surface(
                    z=data,
                    colorscale="Viridis"
                )
            ]
        )

        fig_surface.update_layout(

            title="3D CSI Surface",

            scene=dict(

                xaxis_title="Subcarrier",

                yaxis_title="Packet",

                zaxis_title="Amplitude"

            ),

            height=650
        )

        surface_placeholder.plotly_chart(
            fig_surface,
            use_container_width=True
        )

        ###################################################
        # Heatmap
        ###################################################

        fig_heat = go.Figure(

            data=go.Heatmap(

                z=data,

                colorscale="Turbo"

            )

        )

        fig_heat.update_layout(

            title="Heatmap",

            height=500

        )

        heatmap_placeholder.plotly_chart(
            fig_heat,
            use_container_width=True
        )

        ###################################################
        # Waterfall
        ###################################################

        fig_water = go.Figure()

        step = max(1, len(data)//30)

        for i in range(0, len(data), step):

            fig_water.add_trace(

                go.Scatter3d(

                    x=np.arange(data.shape[1]),

                    y=np.ones(data.shape[1])*i,

                    z=data[i],

                    mode="lines"

                )

            )

        fig_water.update_layout(

            title="3D Waterfall",

            height=650,

            scene=dict(

                xaxis_title="Subcarrier",

                yaxis_title="Packet",

                zaxis_title="Amplitude"

            )

        )

        waterfall_placeholder.plotly_chart(
            fig_water,
            use_container_width=True
        )

    except Exception as e:

        st.error(e)

    time.sleep(REFRESH_RATE)