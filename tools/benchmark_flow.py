"""Time sparse background flow on a supplied local recording."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motion_predictor import BackgroundFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    flow = BackgroundFlow()
    times, valid = [], []
    for _ in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break
        side = min(frame.shape[:2])
        left = (frame.shape[1] - side) // 2
        frame = frame[:side, left:left + side]
        start = time.perf_counter()
        shift = flow.update(frame, (300, 300, 20, 50))
        times.append((time.perf_counter() - start) * 1000)
        if shift is not None:
            valid.append(shift)
    capture.release()
    warm = times[10:] if len(times) > 10 else times
    print(f"frames={len(times)} valid={len(valid)} avg_ms={sum(times)/len(times):.3f} "
          f"max_ms={max(times):.3f} warm_p95_ms={np.percentile(warm,95):.3f} "
          f"warm_p99_ms={np.percentile(warm,99):.3f} warm_max_ms={max(warm):.3f}")
    print("last_valid_shifts=", valid[-5:])


if __name__ == "__main__":
    main()
