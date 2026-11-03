"""Aim Lab tracking and bounded mouse-step control (no capture or hardware imports)."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    index: int
    x: float  # 320x320 detector coordinates
    y: float
    score: float
    track_id: int | None = None
    observed: bool = True


class TargetTracker:
    """Keep one target through small detection gaps; never jump on a missed frame."""

    def __init__(self, radius=42.0, max_misses=3, switch_misses=2):
        self.radius = float(radius)
        self.max_misses = int(max_misses)
        self.switch_misses = int(switch_misses)
        self.reset()

    def reset(self):
        self.center = None
        self.velocity = (0.0, 0.0)
        self.seen_at = None
        self.misses = 0
        self.track_id = None
        self.last_observed_center = None
        self.last_observed_at = None
        self.row_axis = None
        self.row_rank = None
        self.row_count = 0

    @staticmethod
    def _row(candidates):
        """Return a stable order for three or more nearly collinear targets."""
        if len(candidates) < 3:
            return None, None
        span_x = max(c.x for c in candidates) - min(c.x for c in candidates)
        span_y = max(c.y for c in candidates) - min(c.y for c in candidates)
        if span_x >= max(12.0, 2.0 * span_y):
            return "x", sorted(candidates, key=lambda c: c.x)
        if span_y >= max(12.0, 2.0 * span_x):
            return "y", sorted(candidates, key=lambda c: c.y)
        return None, None

    def _remember_row(self, candidates, choice):
        axis, ordered = self._row(candidates)
        self.row_axis = axis
        self.row_rank = next((rank for rank, c in enumerate(ordered)
                              if c is choice), None) if ordered else None
        self.row_count = len(candidates) if ordered else 0

    def update(self, candidates, aim_x, aim_y, now):
        """Return (selected detector index or None, acquired_new_target)."""
        # When the box tracker supplies persistent identities, prefer identity
        # over nearest-center matching. Its predicted box remains a valid aim
        # point for a brief all-target detection gap, but never a fire target.
        if self.track_id is not None:
            same = next((c for c in candidates if c.track_id == self.track_id), None)
            observed_others = [c for c in candidates
                               if c.observed and c.track_id != self.track_id]
            if same is not None and (same.observed or not observed_others
                                     or self.misses + 1 < self.switch_misses):
                if same.observed:
                    if self.last_observed_center is not None:
                        dt = float(now - self.last_observed_at)
                        if dt > 1e-4:
                            vx = (same.x - self.last_observed_center[0]) / dt
                            vy = (same.y - self.last_observed_center[1]) / dt
                            self.velocity = (0.6 * self.velocity[0] + 0.4 * vx,
                                             0.6 * self.velocity[1] + 0.4 * vy)
                    self.last_observed_center = (same.x, same.y)
                    self.last_observed_at = now
                    self.misses = 0
                else:
                    self.misses += 1
                self.center = (same.x, same.y)
                self.seen_at = now
                return same.index, False
            if same is not None and observed_others:
                candidates = observed_others
            else:
                candidates = [c for c in candidates if c.observed]
            self.reset()
            if not candidates:
                return None, False

        if self.center is not None and candidates:
            dt = max(0.0, min(float(now - self.seen_at), 0.05))
            px = self.center[0] + self.velocity[0] * dt
            py = self.center[1] + self.velocity[1] * dt
            choice = min(candidates, key=lambda c: (c.x - px) ** 2 + (c.y - py) ** 2)
            axis, ordered = self._row(candidates)
            row_locked = False
            if (axis is not None and axis == self.row_axis
                    and len(candidates) == self.row_count
                    and self.row_rank is not None):
                # The whole row moves when the camera moves. Nearest-center
                # matching would jump to a neighbor after a one-ball-width move.
                row_choice = ordered[self.row_rank]
                if math.hypot(row_choice.x - self.center[0],
                              row_choice.y - self.center[1]) <= self.radius * 2:
                    choice = row_choice
                    row_locked = True
            distance = math.hypot(choice.x - px, choice.y - py)
            if distance <= self.radius or (row_locked and distance <= self.radius * 2):
                if dt > 1e-4:
                    vx = (choice.x - self.center[0]) / dt
                    vy = (choice.y - self.center[1]) / dt
                    self.velocity = (
                        0.6 * self.velocity[0] + 0.4 * vx,
                        0.6 * self.velocity[1] + 0.4 * vy,
                    )
                self.center = (choice.x, choice.y)
                self.seen_at = now
                self.misses = 0
                self.track_id = choice.track_id
                if choice.observed:
                    self.last_observed_center = (choice.x, choice.y)
                    self.last_observed_at = now
                self._remember_row(candidates, choice)
                return choice.index, False

        if self.center is not None:
            self.misses += 1
            # 另有目标可选时尽快切换；画面没有目标时仍容忍短暂漏检。
            miss_limit = self.switch_misses if candidates else self.max_misses
            if self.misses < miss_limit:
                return None, False
            self.reset()

        candidates = [c for c in candidates if c.observed]
        if not candidates:
            return None, False

        choice = min(candidates, key=lambda c: (
            (c.x - aim_x) ** 2 + (c.y - aim_y) ** 2,
            -c.score,
        ))
        self.center = (choice.x, choice.y)
        self.seen_at = now
        self.track_id = choice.track_id
        if choice.observed:
            self.last_observed_center = (choice.x, choice.y)
            self.last_observed_at = now
        self._remember_row(candidates, choice)
        return choice.index, True


class AimController:
    """Two independent PID axes; parameters convert ROI pixels to mouse counts."""

    def __init__(self, kp=0.06, ki=0.0, kd=0.0, deadzone=2.0,
                 max_step=12, far_boost=1.0, far_threshold=120.0,
                 near_boost=1.0, near_radius=90.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.deadzone = float(deadzone)
        self.max_step = int(max_step)
        self.far_boost = float(far_boost)
        self.far_threshold = float(far_threshold)
        self.near_boost = float(near_boost)
        self.near_radius = float(near_radius)
        self.reset()

    def reset(self):
        self.last_time = None
        self.last_error = [0.0, 0.0]
        self.integral = [0.0, 0.0]
        self.derivative = [0.0, 0.0]
        self.fraction = [0.0, 0.0]

    def update(self, error_x, error_y, now):
        errors = [float(error_x), float(error_y)]
        if not all(math.isfinite(e) for e in errors):
            self.reset()
            return 0, 0

        first = self.last_time is None
        dt = 0.01 if first else max(1 / 240, min(float(now - self.last_time), 0.05))
        self.last_time = float(now)
        steps = []
        for axis, error in enumerate(errors):
            if abs(error) <= self.deadzone:
                self.integral[axis] = 0.0
                self.derivative[axis] = 0.0
                self.fraction[axis] = 0.0
                self.last_error[axis] = error
                steps.append(0)
                continue

            # 远距离平滑提速，接近中心时自动回到原来的 Kp 与限幅。
            ramp = min(1.0, max(0.0,
                       (abs(error) - self.far_threshold) / self.far_threshold))
            boost = 1.0 + (self.far_boost - 1.0) * ramp
            step_limit = min(127, max(1, round(self.max_step * boost)))
            # 临近目标时增加中段拉力；球心周围渐退，避免把增益带进停止圈。
            near_inner = min(self.near_radius * 0.6, max(12.0, self.deadzone))
            if near_inner < abs(error) < self.near_radius:
                phase = (abs(error) - near_inner) / (self.near_radius - near_inner)
                boost *= 1.0 + (self.near_boost - 1.0) * math.sin(math.pi * phase)
            raw_d = 0.0 if first else (error - self.last_error[axis]) / dt
            self.derivative[axis] = 0.2 * raw_d + 0.8 * self.derivative[axis]
            self.integral[axis] = max(-self.max_step, min(
                self.max_step, self.integral[axis] + self.ki * error * dt
            ))
            value = (self.kp * error * boost + self.integral[axis]
                     + self.kd * self.derivative[axis])
            value = max(-step_limit, min(step_limit, value))
            value += self.fraction[axis]
            step = math.trunc(value)
            step = max(-step_limit, min(step_limit, step))
            self.fraction[axis] = 0.0 if abs(step) == step_limit else value - step
            self.last_error[axis] = error
            steps.append(step)
        return tuple(steps)


def mouse_command(dx, dy):
    """Firmware contract to confirm: ASCII M,dx,dy followed by newline."""
    return f"M,{int(dx)},{int(dy)}\n".encode("ascii")


def click_command():
    """Ask the ESP32-S3 HID bridge for one left-button pulse."""
    return b"C\n"


def split_mouse_steps(dx, dy, chunk=20):
    """Split a logical move into firmware-proven small M commands."""
    chunk = int(chunk)
    if chunk < 1 or chunk > 127:
        raise ValueError("chunk must be in 1..127")
    remain_x, remain_y = int(dx), int(dy)
    while remain_x or remain_y:
        step_x = max(-chunk, min(chunk, remain_x))
        step_y = max(-chunk, min(chunk, remain_y))
        yield step_x, step_y
        remain_x -= step_x
        remain_y -= step_y
