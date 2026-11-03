"""Camera-motion compensation and a tightly bounded detection-gap bridge."""

import math

import cv2
import numpy as np


class BackgroundFlow:
    """Estimate camera motion from sparse background corners, not the target."""

    def __init__(self, side=192):
        self.side = int(side)
        self.reset()

    def reset(self):
        self.previous = None
        self.points = None
        self.frames = 0
        self.skip_next = False
        self.last_shift = None
        self.interval_count = 1

    def _features(self, gray, target_box):
        mask = np.zeros_like(gray, dtype=np.uint8)
        lo, hi = int(self.side * 0.17), int(self.side * 0.72)
        mask[lo:hi, :] = 255
        if target_box is not None:
            x, y, w, h = map(float, target_box)
            # Boxes use 640 input coordinates; gray is 320 square.
            scale = self.side / 640.0
            x0 = max(0, int((x - w) * scale))
            y0 = max(0, int((y - h) * scale))
            x1 = min(self.side, int((x + 2 * w) * scale))
            y1 = min(self.side, int((y + 2 * h) * scale))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 0, -1)
        return cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=0.015,
                                       minDistance=5, mask=mask)

    def update(self, frame, target_box):
        if target_box is None:
            self.reset()
            return None
        if self.skip_next:
            self.skip_next = False
            self.interval_count += 1
            return self.last_shift
        gray = cv2.cvtColor(cv2.resize(frame, (self.side, self.side),
                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self.previous is None:
            self.previous = gray
            self.points = self._features(gray, target_box)
            return None
        if self.points is None or len(self.points) < 12:
            self.points = self._features(self.previous, target_box)
        if self.points is None or len(self.points) < 12:
            self.previous = gray
            self.points = self._features(gray, target_box)
            return None
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, self.points, None, winSize=(13, 13), maxLevel=1)
        if next_points is None:
            self.previous = gray
            self.points = self._features(gray, target_box)
            return None
        valid = status.reshape(-1).astype(bool) & (errors.reshape(-1) < 15)
        shifts = (next_points - self.points).reshape(-1, 2)[valid]
        intervals = self.interval_count
        self.interval_count = 1
        self.previous = gray
        self.frames += 1
        self.points = next_points.reshape(-1, 2)[valid].reshape(-1, 1, 2)
        if len(self.points) < 25 or self.frames % 8 == 0:
            self.points = self._features(gray, target_box)
        if len(shifts) < 12:
            return None
        median = np.median(shifts, axis=0)
        scatter = np.median(np.linalg.norm(shifts - median, axis=1))
        if scatter > 2.5 or np.linalg.norm(median) > 9:
            return None
        self.last_shift = tuple(float(v * 640.0 / self.side / intervals)
                                for v in median)
        self.skip_next = True
        return self.last_shift


class PredictedBridge:
    """Carry a confirmed aim point over at most two missed real detections."""

    def __init__(self, max_misses=2, max_age_s=0.075):
        self.max_misses = int(max_misses)
        self.max_age_s = float(max_age_s)
        self.reset()

    def reset(self):
        self.track_id = None
        self.point = None
        self.body_box = None
        self.velocity = (0.0, 0.0)
        self.real_time = None
        self.camera_sum = [0.0, 0.0]

    def observe(self, track_id, point, body_box, velocity, now):
        self.track_id = track_id
        self.point = tuple(map(float, point[:2]))
        self.body_box = tuple(map(float, body_box))
        self.velocity = tuple(map(float, velocity))
        self.real_time = float(now)
        self.camera_sum = [0.0, 0.0]

    def predict(self, track_id, body_box, misses, camera_shift, now):
        if (self.point is None or track_id != self.track_id
                or not 1 <= misses <= self.max_misses):
            return None
        age = float(now) - self.real_time
        if not 0 <= age <= self.max_age_s:
            return None
        old_x, old_y, old_w, old_h = self.body_box
        x, y, w, h = map(float, body_box)
        if math.hypot(x + w / 2 - old_x - old_w / 2,
                      y + h / 2 - old_y - old_h / 2) > max(12.0, old_w * 1.5):
            return None
        if camera_shift is not None:
            self.camera_sum[0] += camera_shift[0]
            self.camera_sum[1] += camera_shift[1]
        dx = self.velocity[0] * age + self.camera_sum[0]
        dy = self.velocity[1] * age + self.camera_sum[1]
        limit = min(8.0, max(2.0, old_w * 0.35)) * misses
        length = math.hypot(dx, dy)
        if length > limit:
            dx, dy = dx * limit / length, dy * limit / length
        return self.point[0] + dx, self.point[1] + dy, "PREDICTED BRIDGE", dx, dy

    def camera_since_real(self, track_id, current_shift):
        """Include camera movement during bridged misses on reacquisition."""
        if self.track_id != track_id or self.real_time is None:
            return current_shift
        x, y = self.camera_sum
        if current_shift is not None:
            x += current_shift[0]
            y += current_shift[1]
        return (x, y) if current_shift is not None or x or y else None
