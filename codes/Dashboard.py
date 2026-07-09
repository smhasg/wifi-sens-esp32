import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="ESP32 CSI Dashboard", layout="wide")

st.title("📡 ESP32 CSI Real-Time Dashboard")

# =========================
# LOAD DATA
# =========================
import pandas as pd
import plotly.graph_objects as go
import numpy as np

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
clean = []
with open("csi_buffer.csv") as f:
    for line in f:
        parts = line.strip().split(",")

        if len(parts) == 261:
            clean.append(parts)

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