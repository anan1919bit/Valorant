"""Small multi-object box tracker: Kalman prediction + Hungarian assignment.

Coordinates are x, y, width, height in the 320x320 detector image. Observed
frames expose the *unmodified detector box*; only missed frames expose the
Kalman-predicted box. This distinction is important for the trigger gate.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    box: tuple[float, float, float, float]
    score: float
    observed: bool
    misses: int


def hungarian(cost):
    """Minimum-cost one-to-one assignment; return (row, column) pairs.

    Rectangular variant of the O(n^3) potential algorithm. Large finite costs
    may be used as gates; the caller rejects those pairs after assignment.
    """
    a = np.asarray(cost, dtype=np.float64)
    if a.size == 0:
        return []
    if a.ndim != 2:
        raise ValueError("cost must be a matrix")
    rows, cols = a.shape
    if rows == 0 or cols == 0:
        return []
    if not np.isfinite(a).all():
        raise ValueError("cost must be finite")
    transposed = rows > cols
    if transposed:
        a = a.T
    n, m = a.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, math.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(j, i) for i, j in pairs]
    return sorted(pairs)


class _Track:
    def __init__(self, track_id, box, score, now):
        self.id = track_id
        self.state = np.zeros(8, dtype=np.float64)
        self.state[:4] = _xywh_to_center(box)
        self.cov = np.diag([9, 9, 9, 9, 100000, 100000, 40000, 40000]).astype(float)
        self.box = tuple(map(float, box))
        self.score = float(score)
        self.observed = True
        self.misses = 0
        self.time = float(now)

    def predict(self, now):
        dt = min(0.05, max(1 / 500, float(now) - self.time))
        self.time = float(now)
        f = np.eye(8)
        f[:4, 4:] = np.eye(4) * dt
        self.state = f @ self.state
        # A little process noise lets the filter follow rapid screen motion.
        q = np.diag([2, 2, 1, 1, 500, 500, 200, 200]) * dt
        self.cov = f @ self.cov @ f.T + q
        self.observed = False
        self.misses += 1
        self.box = _center_to_xywh(self.state[:4])

    def correct(self, box, score):
        z = _xywh_to_center(box)
        h = np.eye(4, 8)
        r = np.diag([6, 6, 9, 9])
        innovation = z - h @ self.state
        s = h @ self.cov @ h.T + r
        gain = np.linalg.solve(s, h @ self.cov).T
        self.state += gain @ innovation
        self.cov = (np.eye(8) - gain @ h) @ self.cov
        self.state[4:] = np.clip(self.state[4:], -1800, 1800)
        # Never substitute the smoothed estimate for an actual detection.
        self.box = tuple(map(float, box))
        self.score = float(score)
        self.observed = True
        self.misses = 0

    def output(self):
        return TrackedBox(self.id, self.box, self.score,
                          self.observed, self.misses)


def _xywh_to_center(box):
    x, y, w, h = map(float, box)
    return np.array([x + w / 2, y + h / 2, w, h], dtype=float)


def _center_to_xywh(center):
    cx, cy, w, h = map(float, center)
    w = max(2.0, min(320.0, w))
    h = max(2.0, min(320.0, h))
    return (cx - w / 2, cy - h / 2, w, h)


class BoxTracker:
    def __init__(self, match_radius=55.0, max_prediction_frames=10):
        self.match_radius = float(match_radius)
        self.max_prediction_frames = int(max_prediction_frames)
        if self.match_radius <= 0 or not 0 <= self.max_prediction_frames <= 30:
            raise ValueError("invalid box-tracker settings")
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def drop(self, track_id):
        self.tracks = [track for track in self.tracks if track.id != track_id]

    def update(self, detections, now):
        """Return stable-ID tracks, with raw boxes on hits and predictions on misses.

        detections: iterable of (xywh box, confidence). Each call is one frame.
        """
        measured = []
        for box, score in detections:
            z = _xywh_to_center(box)
            if np.isfinite(z).all() and z[2] > 0 and z[3] > 0:
                measured.append((tuple(map(float, box)), float(score)))
        for track in self.tracks:
            track.predict(now)
        costs = np.empty((len(self.tracks), len(measured)), dtype=float)
        for i, track in enumerate(self.tracks):
            tc = track.state[:4]
            for j, (box, _score) in enumerate(measured):
                dc = _xywh_to_center(box)
                distance = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
                gate = self.match_radius + min(20.0, 3.0 * track.misses)
                size_penalty = 0.15 * math.hypot(tc[2] - dc[2], tc[3] - dc[3])
                costs[i, j] = distance + size_penalty if distance <= gate else 1e6
        matched_detections = set()
        for i, j in hungarian(costs):
            if costs[i, j] < 1e6:
                box, score = measured[j]
                self.tracks[i].correct(box, score)
                matched_detections.add(j)
        self.tracks = [track for track in self.tracks
                       if track.misses <= self.max_prediction_frames]
        for j, (box, score) in enumerate(measured):
            if j not in matched_detections:
                self.tracks.append(_Track(self.next_id, box, score, now))
                self.next_id += 1
        return [track.output() for track in self.tracks]
