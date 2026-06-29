import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os

FILE = "csi_buffer.csv"

st.set_page_config(layout="wide")

st.title("ESP32 CSI Monitor")

placeholder = st.empty()

while True:

    if os.path.exists(FILE):

        try:

            df = pd.read_csv(FILE)

            rows = []

            for line in df["raw"].tail(300):

                parts = str(line).split(",")

                if len(parts) < 4:
                    continue

                rows.append({
                    "timestamp": int(parts[0]),
                    "rssi": int(parts[1]),
                    "channel": int(parts[2]),
                    "len": int(parts[3])
                })

            if len(rows):

                rdf = pd.DataFrame(rows)

                with placeholder.container():

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Current RSSI",
                        rdf["rssi"].iloc[-1]
                    )

                    c2.metric(
                        "Mean RSSI",
                        round(rdf["rssi"].mean(), 2)
                    )

                    c3.metric(
                        "Samples",
                        len(rdf)
                    )

                    fig = px.line(
                        rdf,
                        x="timestamp",
                        y="rssi",
                        title="Realtime RSSI"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    fig2 = px.histogram(
                        rdf,
                        x="rssi",
                        nbins=40,
                        title="RSSI Distribution"
                    )

                    st.plotly_chart(
                        fig2,
                        use_container_width=True
                    )

        except Exception as e:
            st.error(e)

    time.sleep(1)