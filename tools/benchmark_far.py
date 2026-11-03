"""Inspect full-frame and one-crop detections on supplied still images."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from target_policy import class_boxes, StandingBodyFilter
from yellow_proposals import find_yellow_proposals, recheck_yellow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    ort.preload_dlls(directory="")
    session = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    key = session.get_inputs()[0].name

    def infer(blob):
        return session.run(None, {key: blob})[0]

    for filename in args.images:
        frame = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(filename)
        height, width = frame.shape[:2]
        size = min(960, height, width)
        left, top = (width - size) // 2, (height - size) // 2
        frame = frame[top:top + size, left:left + size]
        small = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(small[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = infer(blob)
        body, bs, bk = class_boxes(output, 0, 0.20, 0.45)
        head, hs, hk = class_boxes(output, 1, 0.20, 0.45)
        strict = StandingBodyFilter().filter(body, bk, 0.0, head[hk])
        accepted = StandingBodyFilter().filter(body, bk, 0.0, head[hk], allow_tiny_confirmed=True)
        proposals = find_yellow_proposals(frame, max_candidates=3)
        crop_body, crop_scores, crop_head, _, used = recheck_yellow(
            frame, infer, class_boxes, 640, 0.39, 0.45, max_candidates=1, proposals=proposals[:1])
        print(Path(filename).name, "ROI", (left, top, size), "full_body", len(bk),
              "strict", len(strict), "tiny_head_confirmed", len(accepted),
              "full_head", len(hk), "yellow", proposals,
              "crop_body", len(crop_body), "crop_head", len(crop_head), "used", used)
        for i in bk[:8]:
            print(" BODY", round(float(bs[i]), 3), np.rint(body[i]).astype(int).tolist(),
                  "accepted", int(i) in accepted)
        for i in hk[:8]:
            print(" HEAD", round(float(hs[i]), 3), np.rint(head[i]).astype(int).tolist())
        for box, score in zip(crop_body, crop_scores):
            print(" CROP", round(float(score), 3), np.rint(box).astype(int).tolist())


if __name__ == "__main__":
    main()
