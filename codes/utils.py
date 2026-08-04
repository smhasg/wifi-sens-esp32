import numpy as np


CSI_COLUMNS = [
    f"cs{i}"
    for i in range(128)
]


def compute_energy(df):
    """
    Calculate CSI energy for each packet

    Input:
        DataFrame containing cs0..cs128

    Output:
        numpy array
    """

    csi = df[CSI_COLUMNS].values

    energy = np.sum(
        np.square(csi),
        axis=1
    )

    return energy



def compute_variance(df):
    """
    Calculate CSI variance for each packet

    Input:
        DataFrame containing cs0..cs128

    Output:
        numpy array
    """

    csi = df[CSI_COLUMNS].values

    variance = np.var(
        csi,
        axis=1
    )

    return variance



def get_csi_columns():

    return CSI_COLUMNS