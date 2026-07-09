import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

import joblib
import time


from serial_reader import SerialReader



st.set_page_config(
    page_title="ESP32 CSI Dashboard",
    layout="wide"
)


st.title(
    "📡 ESP32-S3 CSI Human Sensing Dashboard"
)



# =========================
# SERIAL
# =========================


if "reader" not in st.session_state:


    reader = SerialReader(
        "COM9",
        921600
    )

    reader.start()

    st.session_state.reader = reader



reader = st.session_state.reader



# =========================
# MODEL
# =========================


model_file = st.sidebar.file_uploader(
    "Upload XGBoost model",
    type=["pkl"]
)


model=None


if model_file:

    model = joblib.load(
        model_file
    )

    st.sidebar.success(
        "Model loaded"
    )



# =========================
# LABEL
# =========================


label = st.sidebar.radio(
    "People inside room",
    [
        0,
        1,
        2
    ]
)



# =========================
# READ DATA
# =========================


packets = reader.get_data()



if len(packets)==0:

    st.warning(
        "Waiting ESP32 data..."
    )

    st.stop()



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



st.subheader(
    "Live Data"
)


st.dataframe(
    df.tail()
)



# =========================
# PREDICTION
# =========================


if model:


    X=df.tail(1)


    prediction=model.predict(
        X
    )[0]


    st.metric(
        "Model Prediction",
        f"{prediction} people"
    )


else:


    st.info(
        "No model uploaded - visualization mode"
    )




# =========================
# VISUALIZATION
# =========================


mode=st.sidebar.selectbox(

    "Visualization",

    [

    "Raw Signal",
    "MinMax",
    "Z-score",
    "Moving Average",
    "Heatmap"

    ]

)



feature=st.sidebar.selectbox(

    "CSI antenna",

    [

        f"cs{i}"
        for i in range(256)

    ]

)



def line_plot(data,title):

    fig=go.Figure()


    fig.add_trace(
        go.Scatter(
            y=data,
            mode="lines"
        )
    )


    fig.update_layout(
        height=400,
        title=title
    )


    return fig




if mode=="Raw Signal":

    st.plotly_chart(
        line_plot(
            df[feature],
            "Raw CSI"
        ),
        use_container_width=True
    )



elif mode=="MinMax":


    x=MinMaxScaler().fit_transform(
        df[[feature]]
    )


    st.plotly_chart(
        line_plot(
            x,
            "MinMax"
        )
    )



elif mode=="Z-score":


    x=StandardScaler().fit_transform(
        df[[feature]]
    )


    st.plotly_chart(
        line_plot(
            x,
            "Z-score"
        )
    )



elif mode=="Moving Average":


    ma=df[feature].rolling(
        20
    ).mean()


    st.plotly_chart(
        line_plot(
            ma,
            "Moving Average"
        )
    )



elif mode=="Heatmap":


    matrix=df[
        [
        f"cs{i}"
        for i in range(256)
        ]
    ].values.T


    fig=go.Figure(

        go.Heatmap(
            z=matrix
        )

    )


    fig.update_layout(
        height=600
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



time.sleep(0.1)

st.rerun()