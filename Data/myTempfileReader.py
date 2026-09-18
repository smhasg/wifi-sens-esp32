
import numpy as np

file_path = "one-Person-in-Room_0001_000001.npz"

data = np.load(file_path, allow_pickle=True)

print("Keys / Header:")
print(data.files)

for key in data.files:
    arr = data[key]
    print(f"\n--- {key} ---")
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("First values:", arr.flatten()[:10])