import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

NPZ_FILE = "one-Person-in-Room_0001_000001.npz"


# ============================================================
# LOAD
# ============================================================

data = np.load(NPZ_FILE)

print("=" * 80)
print("NPZ KEYS")
print("=" * 80)

print(data.files)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("BASIC INFORMATION")
print("=" * 80)

for key in data.files:
    x = data[key]

    print(f"\n--- {key} ---")
    print("shape :", x.shape)
    print("dtype :", x.dtype)

    if np.issubdtype(x.dtype, np.number):
        print("min   :", np.nanmin(x))
        print("max   :", np.nanmax(x))
        print("mean  :", np.nanmean(x))
        print("std   :", np.nanstd(x))

        print("NaN   :", np.isnan(x).sum())
        print("Inf   :", np.isinf(x).sum())


# ============================================================
# CSI CHECK
# ============================================================

csi = data["csi"]
import numpy as np

print("\n" + "=" * 80)
print("CSI ANALYSIS")

print("Shape:", csi.shape)
print("Dtype:", csi.dtype)
print("=" * 80)

# Real / imaginary
real = np.real(csi)
imag = np.imag(csi)

print("\nREAL PART")
print("min :", real.min())
print("max :", real.max())
print("mean:", real.mean())
print("std :", real.std())

print("\nIMAGINARY PART")
print("min :", imag.min())
print("max :", imag.max())
print("mean:", imag.mean())
print("std :", imag.std())


# ============================================================
# AMPLITUDE
# ============================================================

amp = np.abs(csi)

print("\n" + "=" * 80)
print("AMPLITUDE")
print("=" * 80)

print("Shape:", amp.shape)

print("min :", amp.min())
print("max :", amp.max())
print("mean:", amp.mean())
print("std :", amp.std())


# ============================================================
# PHASE
# ============================================================

phase = np.angle(csi)

print("\n" + "=" * 80)
print("PHASE")
print("=" * 80)

print("Shape:", phase.shape)

print("min :", phase.min())
print("max :", phase.max())
print("mean:", phase.mean())
print("std :", phase.std())


# ============================================================
# ZERO CHECK PER CSI COLUMN
# ============================================================

print("\n" + "=" * 80)
print("ZERO CHECK - CSI COLUMNS")
print("=" * 80)

zero_count = np.sum(np.abs(csi) == 0, axis=0)

zero_ratio = zero_count / csi.shape[0]

zero_report = pd.DataFrame({
    "column": np.arange(csi.shape[1]),
    "zero_count": zero_count,
    "zero_ratio": zero_ratio
})

print(zero_report.to_string(index=False))

print("\nColumns that are 100% ZERO:")

all_zero = np.where(zero_ratio == 1.0)[0]

print(all_zero)

print("\nNumber of all-zero columns:", len(all_zero))


# ============================================================
# CONSTANT COLUMN CHECK
# ============================================================

print("\n" + "=" * 80)
print("CONSTANT CSI COLUMNS")
print("=" * 80)

column_std = np.std(csi, axis=0)

constant_columns = np.where(column_std == 0)[0]

print("Constant columns:")
print(constant_columns)

print("Number:", len(constant_columns))


# ============================================================
# UNIQUE VALUES PER COLUMN
# ============================================================

print("\n" + "=" * 80)
print("UNIQUE VALUE CHECK")
print("=" * 80)

for i in range(csi.shape[1]):

    unique_count = len(np.unique(csi[:, i]))

    print(
        f"CSI[{i:3d}] -> "
        f"unique values = {unique_count}"
    )


# ============================================================
# FIRST PACKETS
# ============================================================

print("\n" + "=" * 80)
print("FIRST 10 CSI PACKETS")
print("=" * 80)

print(csi[:10])


# ============================================================
# FIRST PACKET DETAILS
# ============================================================

print("\n" + "=" * 80)
print("FIRST CSI PACKET")
print("=" * 80)

print("Complex:")
print(csi[0])

print("\nAmplitude:")
print(amp[0])

print("\nPhase:")
print(phase[0])


# ============================================================
# PACKET-TO-PACKET CHANGE
# ============================================================

print("\n" + "=" * 80)
print("TEMPORAL CSI STABILITY")
print("=" * 80)

# Difference between consecutive CSI packets
diff = np.abs(csi[1:] - csi[:-1])

print("Mean packet difference :", diff.mean())
print("Std packet difference  :", diff.std())
print("Max packet difference  :", diff.max())


# ============================================================
# IDENTICAL PACKETS
# ============================================================

identical_packets = np.all(
    csi[1:] == csi[:-1],
    axis=1
)

print("\nIdentical consecutive packets:")
print(identical_packets.sum())

print(
    "Ratio:",
    identical_packets.mean()
)


# ============================================================
# RSSI
# ============================================================

if "rssi" in data.files:

    rssi = data["rssi"]

    print("\n" + "=" * 80)
    print("RSSI")
    print("=" * 80)

    print("min :", rssi.min())
    print("max :", rssi.max())
    print("mean:", rssi.mean())
    print("std :", rssi.std())

    print("\nUnique RSSI values:")
    print(np.unique(rssi))


# ============================================================
# CHANNEL
# ============================================================

if "channel" in data.files:

    channel = data["channel"]

    print("\n" + "=" * 80)
    print("CHANNEL")
    print("=" * 80)

    print("Unique channels:")
    print(np.unique(channel, return_counts=True))


# ============================================================
# CSI LENGTH
# ============================================================

if "csi_len" in data.files:

    csi_len = data["csi_len"]

    print("\n" + "=" * 80)
    print("CSI LENGTH")
    print("=" * 80)

    print("Unique CSI lengths:")
    print(np.unique(csi_len, return_counts=True))


# ============================================================
# TIMESTAMP
# ============================================================

if "pc_time" in data.files:

    pc_time = data["pc_time"]

    print("\n" + "=" * 80)
    print("PC TIMESTAMP")
    print("=" * 80)

    print("First :", pc_time[0])
    print("Last  :", pc_time[-1])

    dt = pd.to_datetime(pc_time)

    delta = dt[1:] - dt[:-1]

    print("\nTime delta statistics:")
    print("mean:", delta.mean())
    print("min :", delta.min())
    print("max :", delta.max())


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("CSI shape:", csi.shape)

print("All-zero columns:", len(all_zero))

print("Constant columns:", len(constant_columns))

print("Amplitude mean/std:", amp.mean(), amp.std())

print("Phase mean/std:", phase.mean(), phase.std())

print(
    "Identical consecutive packet ratio:",
    identical_packets.mean()
)

print("\nDONE.")

import matplotlib.pyplot as plt
# ============================================================
# CSI VISUALIZATION
# ============================================================

plt.figure(figsize=(16, 6))

plt.imshow(
    np.abs(csi),
    aspect="auto",
    interpolation="nearest"
)

plt.xlabel("CSI Subcarrier")
plt.ylabel("Packet")
plt.title("CSI Amplitude")

plt.colorbar(label="Amplitude")

plt.tight_layout()
plt.show()


# ============================================================
# PHASE
# ============================================================

plt.figure(figsize=(16, 6))

plt.imshow(
    np.angle(csi),
    aspect="auto",
    interpolation="nearest",
    cmap="twilight"
)

plt.xlabel("CSI Subcarrier")
plt.ylabel("Packet")
plt.title("CSI Phase")

plt.colorbar(label="Phase")

plt.tight_layout()
plt.show()


# ============================================================
# SELECTED CSI COLUMNS OVER TIME
# ============================================================

plt.figure(figsize=(16, 7))

for i in range(min(10, csi.shape[1])):
    plt.plot(
        np.abs(csi[:, i]),
        label=f"SC {i}"
    )

plt.xlabel("Packet")
plt.ylabel("Amplitude")
plt.title("CSI Amplitude Over Time")

plt.legend()
plt.tight_layout()
plt.show()