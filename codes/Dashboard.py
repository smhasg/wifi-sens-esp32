import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import time
import serial
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ESP32 CSI Dashboard", layout="wide")

st.title("📡 ESP32 CSI Real-Time Dashboard")

# =========================
# LOAD DATA
# =========================

PORT = "COM9"
BAUD = 921600
ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)
HEADER = [
    "pc_time",
    "esp_time",
    "rssi",
    "channel",
    "csi_len"
]

HEADER += [
    f"cs{i}"
    for i in range(256)
]


packet_count = 0
start = time.time()
clean = []

while True :
    try:
            line = ser.readline().decode(
                errors="ignore"
            ).strip()

            if not line:
                continue

            values = line.split(",")

            if len(values) < 4:
                continue

            esp_time = values[0]
            rssi = values[1]
            channel = values[2]
            csi_len = int(values[3])

            csi = values[4:]

            # normalize CSI length
            if len(csi) > 256:
                csi = csi[:256]

            elif len(csi) < 256:
                csi += [
                    "0"
                ] * (256 - len(csi))

            pc_time = datetime.now().isoformat(
                timespec="milliseconds"
            )

            row = [
                pc_time,
                esp_time,
                rssi,
                channel,
                str(csi_len)
            ]

            row += csi

            clean.append(row)

            packet_count += 1

            if packet_count % 100 == 0:

                elapsed = time.time() - start

                rate = packet_count / elapsed

                print(
                    f"Packets: {packet_count} | "
                    f"Rate: {rate:.1f} Hz | "
                    f"Last CSI: {csi_len}"
                )
                break

    except Exception as e:

            print(
                "ERROR:",
                e
            )

# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

df = pd.DataFrame(clean)
st.write("### Raw Data Preview")
st.dataframe(df.head())

# if uploaded_file:
    # df = pd.read_csv(uploaded_file)

st.write("### Raw Data Preview")
st.dataframe(df.head())

# =========================
# SPLIT COLUMNS
# =========================
time_col = df.columns[0]
feature_cols = df.columns[2:]

X = df[feature_cols].astype(float)

# =========================
# NORMALIZATION
# =========================
scaler_minmax = MinMaxScaler()
scaler_z = StandardScaler()

X_minmax = scaler_minmax.fit_transform(X)
X_z = scaler_z.fit_transform(X)

df_minmax = pd.DataFrame(X_minmax, columns=feature_cols)
df_z = pd.DataFrame(X_z, columns=feature_cols)

df_minmax[time_col] = df[time_col]
df_z[time_col] = df[time_col]

# =========================
# SIDEBAR CONTROLS
# =========================
mode = st.sidebar.selectbox(
    "Visualization Mode",
    [
        "Raw Signal",
        "MinMax Normalized",
        "Z-Score Normalized",
        "Moving Average",
        "Heatmap (CSI Matrix)"
    ]
)

window = st.sidebar.slider("Moving Window", 5, 100, 20)

selected_feature = st.sidebar.selectbox("Feature Column", feature_cols)

# =========================
# REUSABLE FUNCTION
# =========================
def realtime_plot(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode="lines",
        name=title
    ))
    fig.update_layout(height=400)
    return fig

# =========================
# DISPLAY MODES
# =========================

st.subheader(f"📊 Mode: {mode}")

if mode == "Raw Signal":
    st.plotly_chart(realtime_plot(df[selected_feature], "Raw"))

elif mode == "MinMax Normalized":
    st.plotly_chart(realtime_plot(df_minmax[selected_feature], "MinMax"))

elif mode == "Z-Score Normalized":
    st.plotly_chart(realtime_plot(df_z[selected_feature], "Z-Score"))

elif mode == "Moving Average":
    ma = df[selected_feature].rolling(window).mean()
    st.plotly_chart(realtime_plot(ma, "Moving Average"))

elif mode == "Heatmap (CSI Matrix)":

    matrix = df[feature_cols].values.T

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale="Viridis"
    ))

    fig.update_layout(height=600)
    st.plotly_chart(fig)

# =========================
# LIVE SIMULATION (fake realtime)
# =========================
st.subheader("⚡ Realtime Simulation")

placeholder = st.empty()

for i in range(50, len(df)):

    live_data = df[selected_feature].iloc[i-50:i]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=live_data, mode="lines"))

    placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(0.05)