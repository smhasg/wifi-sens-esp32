import cv2
import numpy as np

print("=" * 70)
print("CAMERA INDEX TEST")
print("=" * 70)

for index in range(10):
    print(f"\n[INDEX {index}]")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("  NOT OPENED")
        cap.release()
        continue

    print("  OPENED")

    ret, frame = cap.read()

    if not ret or frame is None:
        print("  READ FAILED")
        cap.release()
        continue

    print(f"  shape   = {frame.shape}")
    print(f"  dtype   = {frame.dtype}")
    print(f"  min     = {frame.min()}")
    print(f"  max     = {frame.max()}")
    print(f"  mean    = {frame.mean():.2f}")
    print(f"  std     = {frame.std():.2f}")
    print(f"  nonzero = {(np.count_nonzero(frame) / frame.size):.4f}")

    # ذخیره فریم برای بررسی
    cv2.imwrite(f"camera_index_{index}.jpg", frame)

    cap.release()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)