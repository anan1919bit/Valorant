"""Head-first selection with stable body fallback for the local bot model."""

import numpy as np
import cv2
from person_aim import match_head


def class_boxes(output, class_id, confidence, nms_threshold):
    """Extract one YOLOv8 class from [1, 9, N] without mixing class scores."""
    rows = np.asarray(output[0].T, dtype=np.float32)
    scores = rows[:, 4 + class_id]
    valid = (np.isfinite(rows[:, :4]).all(axis=1) & np.isfinite(scores)
             & (rows[:, 2] > 0) & (rows[:, 3] > 0)
             & (scores >= confidence))
    rows = rows[valid]
    scores = scores[valid]
    if len(rows) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=int)
    boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2,
                             rows[:, 1] - rows[:, 3] / 2,
                             rows[:, 2], rows[:, 3])).astype(np.float32)
    keep = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                        confidence, nms_threshold), dtype=int).reshape(-1)
    return boxes, scores, keep


class StandingBodyFilter:
    """Reject flat floor remnants and implausible edge slivers before tracking."""

    def __init__(self, memory_s=0.40):
        self.memory_s = float(memory_s)
        self.previous = []

    def filter(self, boxes, keep, now, head_boxes=None, allow_tiny_confirmed=False):
        old = [(b, t) for b, t in self.previous if now - t <= self.memory_s]
        accepted = []
        for i in keep:
            x, y, w, h = map(float, boxes[i])
            ratio = h / max(w, 1.0)
            has_real_head = (head_boxes is not None and
                             match_head(boxes[i], head_boxes) is not None)
            tiny_confirmed = (allow_tiny_confirmed and has_real_head
                              and w >= 5 and h >= 16 and 1.35 <= ratio <= 3.5)
            crouch_continuation = any(
                1.0 <= ratio < 1.35
                and abs(x + w / 2 - (ox + ow / 2)) <= max(w, ow) * 0.35
                and abs(y + h - (oy + oh)) <= max(10.0, oh * 0.25)
                and 0.48 <= h / max(oh, 1.0) <= 1.18
                for (ox, oy, ow, oh), _ in old
            )
            # A crouched bot is shorter/wider than a standing one. Admit that
            # shape with a current real head or a geometrically continuous body.
            if not (tiny_confirmed or
                    (w >= 8 and h >= 24 and
                     (1.35 <= ratio <= 2.6 or
                      has_real_head and 0.75 <= ratio <= 2.6 or
                      crouch_continuation))):
                continue
            # A disappearing bot can leave a low, small box at its feet.
            remnant = any(
                abs(x + w / 2 - (ox + ow / 2)) <= max(ow, w) * 0.65
                and y >= oy + oh * 0.42 and h <= oh * 0.70
                for (ox, oy, ow, oh), _ in old
            )
            if not remnant or has_real_head or crouch_continuation:
                accepted.append(int(i))
        self.previous = old + [(tuple(map(float, boxes[i])), now) for i in accepted]
        return np.asarray(accepted, dtype=int)


class HeadBodyPolicy:
    def __init__(self, miss_grace=2, reacquire_hits=2):
        self.miss_grace = int(miss_grace)
        self.reacquire_hits = int(reacquire_hits)
        self.reset()

    def reset(self):
        self.active = 1  # 1=head, 0=body
        self.had_head = False
        self.head_misses = 0
        self.head_hits = 0

    def choose(self, head_count, body_count, head_preferred=True):
        """Return class id; head gaps briefly use prediction before fallback."""
        if not head_preferred:
            self.active = 0
            return 0
        if self.active == 1:
            if head_count:
                self.had_head = True
                self.head_misses = 0
            else:
                self.head_misses += 1
                if body_count and (not self.had_head or self.head_misses > self.miss_grace):
                    self.active = 0
                    self.head_hits = 0
            return self.active
        if head_count:
            self.head_hits += 1
            if self.head_hits >= self.reacquire_hits:
                self.active = 1
                self.had_head = True
                self.head_misses = 0
        else:
            self.head_hits = 0
        return self.active


def target_point(box, body_fallback=False):
    x, y, w, h = map(float, box)
    return x + w / 2, y + (0.13 * h if body_fallback else 0.5 * h)


def trigger_box(box, body_fallback=False):
    """A small virtual head box for click tests when only a body is observed."""
    if not body_fallback:
        return tuple(map(float, box))
    x, y, w, h = map(float, box)
    cx, cy = target_point(box, True)
    side = min(w * 0.5, h * 0.18)
    return (cx - side / 2, cy - side / 2, side, side)
