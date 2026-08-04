import numpy as np
import plotly.graph_objects as go


def live_csi(df):

    cols = [f"cs{i}" for i in range(128)]

    y = df.iloc[-1][cols]

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=np.arange(128),

            y=y,

            mode="lines"

        )

    )

    fig.update_layout(

        title="Live CSI",

        height=350

    )

    return fig


def waterfall(df):

    cols = [f"cs{i}" for i in range(128)]

    matrix = df[cols].values.T

    fig = go.Figure(

        go.Heatmap(

            z=matrix

        )

    )

    fig.update_layout(

        title="CSI Waterfall",

        height=500

    )

    return fig


def rssi(df):

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            y=df["rssi"],

            mode="lines"

        )

    )

    fig.update_layout(

        title="RSSI",

        height=300

    )

    return fig