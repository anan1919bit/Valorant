"""Yellow-highlight region proposals; never treated as confirmed targets."""

import cv2
import numpy as np


def find_yellow_proposals(frame, max_candidates=3):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (18, 105, 145), (42, 255, 255))
    mask[:int(h * 0.18)] = 0       # top HUD
    mask[int(h * 0.78):] = 0       # weapon / bottom HUD
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)))
    found = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (area >= 40 and 10 <= bw <= w * 0.11 and 16 <= bh <= h * 0.16
                and 0.45 <= bh / bw <= 5):
            found.append((x, y, bw, bh, area))
    found.sort(key=lambda item: item[4], reverse=True)
    return [item[:4] for item in found[:max_candidates]]


def recheck_yellow(frame, infer, class_boxes, input_size, confidence, nms_threshold,
                   max_candidates=2, crop_size=256, proposals=None):
    """Return mapped body/head detections only after cropped YOLO confirms them."""
    height, width = frame.shape[:2]
    bodies, body_scores, heads, head_scores = [], [], [], []
    proposals = (find_yellow_proposals(frame, max_candidates=max_candidates)
                 if proposals is None else proposals[:max_candidates])
    for px, py, pw, ph in proposals:
        cx, cy = px + pw / 2, py + ph / 2
        side = min(crop_size, width, height)
        left = max(0, min(width - side, int(round(cx - side / 2))))
        top = max(0, min(height - side, int(round(cy - side / 2))))
        crop = frame[top:top + side, left:left + side]
        small = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(small[:, :, ::-1].transpose(2, 0, 1)[None],
                                    dtype=np.float32) / 255.0
        output = infer(blob)
        raw_bodies, bs, bk = class_boxes(output, 0, max(confidence, 0.48), nms_threshold)
        raw_heads, hs, hk = class_boxes(output, 1, max(confidence, 0.48), nms_threshold)

        def map_box(box):
            x, y, w, h = map(float, box)
            return ((left + x * side / input_size) * input_size / width,
                    (top + y * side / input_size) * input_size / height,
                    w * side / width, h * side / height)

        for i in bk:
            x, y, w, h = map(float, raw_bodies[i])
            # A confirmed box must contain the yellow region that proposed it.
            if (w <= 0 or h <= 0 or not 0.75 <= h / w <= 2.8 or
                    not x - w * 0.2 <= (cx - left) * input_size / side <= x + w * 1.2 or
                    not y - h * 0.3 <= (cy - top) * input_size / side <= y + h * 1.2):
                continue
            bodies.append(map_box(raw_bodies[i]))
            body_scores.append(float(bs[i]))
        for i in hk:
            heads.append(map_box(raw_heads[i]))
            head_scores.append(float(hs[i]))
    return (np.asarray(bodies, dtype=np.float32).reshape(-1, 4),
            np.asarray(body_scores, dtype=np.float32),
            np.asarray(heads, dtype=np.float32).reshape(-1, 4),
            np.asarray(head_scores, dtype=np.float32), len(proposals))


def zoom_locked_target(frame, lock_box, infer, class_boxes, input_size,
                       confidence, nms_threshold):
    """Recheck a prior real lock at higher scale; return only nearby YOLO boxes."""
    x, y, w, h = map(float, lock_box)
    fh, fw = frame.shape[:2]
    cx = (x + w / 2) * fw / input_size
    cy = (y + h / 2) * fh / input_size
    bodies, scores, heads, head_scores, _ = recheck_yellow(
        frame, infer, class_boxes, input_size, confidence, nms_threshold,
        max_candidates=1, proposals=[(cx, cy, 1, 1)])
    if not len(bodies):
        return bodies, scores, heads[:0], head_scores[:0]
    centers = bodies[:, :2] + bodies[:, 2:] / 2
    distances = np.hypot(centers[:, 0] - (x + w / 2),
                         centers[:, 1] - (y + h / 2))
    nearest = int(np.argmin(distances))
    if distances[nearest] > max(35.0, 1.5 * max(w, h)):
        return bodies[:0], scores[:0], heads[:0], head_scores[:0]
    body = bodies[nearest:nearest + 1]
    # Head association is performed again by the caller's match_head policy.
    return body, scores[nearest:nearest + 1], heads, head_scores
