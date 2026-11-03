"""Per-person body lock and damped closed-loop motion for the bot experiment."""

import math


class StickyBodyTracker:
    """Hold one body track until it is gone; never choose a nearer neighbor mid-lock."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.track_id = None

    def update(self, candidates, aim_x, aim_y, now=None, allow_reacquire=True):
        if self.track_id is not None:
            for candidate in candidates:
                if candidate.track_id == self.track_id:
                    if candidate.observed:
                        return candidate.index, False
                    # A ghost must not block a newly visible bot for 3 frames.
                    visible = [c for c in candidates if c.observed]
                    if visible and allow_reacquire:
                        choice = min(visible, key=lambda c: (
                            (c.x - aim_x) ** 2 + (c.y - aim_y) ** 2, -c.score))
                        self.track_id = choice.track_id
                        return choice.index, True
                    return candidate.index, False
            self.track_id = None
            if not allow_reacquire:
                return None, False
        visible = [c for c in candidates if c.observed]
        if not visible:
            return None, False
        choice = min(visible, key=lambda c: ((c.x - aim_x) ** 2 + (c.y - aim_y) ** 2,
                                              -c.score))
        self.track_id = choice.track_id
        return choice.index, True


class ClickPulse:
    """One physical button-down edge starts one bounded aim attempt."""

    def __init__(self, timeout_s=1.2, arrival_px=6.0):
        self.timeout_s = float(timeout_s)
        self.arrival_px = float(arrival_px)
        self.until = 0.0

    def start(self, now):
        self.until = float(now) + self.timeout_s

    def active(self, now):
        return float(now) < self.until

    def stop(self):
        self.until = 0.0

    def arrived(self, error_x, error_y):
        return math.hypot(error_x, error_y) <= self.arrival_px


def match_head(body_box, head_boxes):
    """Find the real head belonging to this body, not a neighboring person."""
    bx, by, bw, bh = map(float, body_box)
    expected_x = bx + bw / 2
    expected_y = by + bh * 0.125
    choices = []
    for box in head_boxes:
        hx, hy, hw, hh = map(float, box)
        cx, cy = hx + hw / 2, hy + hh / 2
        if (bx - bw * 0.20 <= cx <= bx + bw * 1.20
                and by - bh * 0.45 <= cy <= by + bh * 0.50):
            distance = abs(cx - expected_x) / max(bw, 1) + abs(cy - expected_y) / max(bh, 1)
            choices.append((distance, box))
    return min(choices, key=lambda pair: pair[0])[1] if choices else None


def person_target(body_box, head_boxes, head_mode=True):
    """Return aim point: real matched head, otherwise upper body, or body center."""
    bx, by, bw, bh = map(float, body_box)
    if not head_mode:
        return bx + bw / 2, by + bh / 2, False
    head = match_head(body_box, head_boxes)
    if head is not None:
        hx, hy, hw, hh = map(float, head)
        return hx + hw / 2, hy + hh / 2, True
    return bx + bw / 2, by + 0.125 * bh, False


class DampedController:
    """Direct target-centered pursuit with extra near pull and bounded counts."""

    def __init__(self, kp=0.18, deadzone=2.0, max_step=48, ki=0.0,
                 far_boost=1.0, far_threshold=120.0,
                 near_boost=1.0, near_radius=90.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.deadzone = float(deadzone)
        self.max_step = int(max_step)
        self.far_boost = float(far_boost)
        self.far_threshold = float(far_threshold)
        self.near_boost = float(near_boost)
        self.near_radius = float(near_radius)
        self.reset()

    def reset(self):
        self.integral = [0.0, 0.0]
        self.last_error = [0.0, 0.0]
        self.last_time = None
        self.last_large_move_at = float("-inf")

    def update(self, error_x, error_y, now):
        errors = (float(error_x), float(error_y))
        if not all(math.isfinite(e) for e in errors):
            self.reset()
            return 0, 0
        distance = math.hypot(*errors)
        braking = now - self.last_large_move_at <= 0.18 and distance <= 160.0
        dt = 0.0 if self.last_time is None else max(0.0, min(0.05, now - self.last_time))
        self.last_time = now
        approach = min(1.0, max(0.0, (self.near_radius - distance)
                                / max(1.0, self.near_radius * 0.5)))
        near_gain = 1.0 + (self.near_boost - 1.0) * approach
        steps = []
        for axis, error in enumerate(errors):
            if abs(error) <= self.deadzone:
                self.integral[axis] = 0.0
                steps.append(0)
                continue
            # Suppress only the first tiny counter-move after crossing the aim point.
            # The next frame may correct it; larger reversals stay immediate.
            if abs(error) <= 8.0 and error * self.last_error[axis] < 0:
                self.integral[axis] = 0.0
                steps.append(0)
                continue
            # Separation integral: only persistent, small same-direction errors.
            if self.ki and self.deadzone < distance < 24 and error * self.last_error[axis] > 0:
                self.integral[axis] = max(-6.0, min(6.0,
                                                   self.integral[axis] + error * dt))
            else:
                self.integral[axis] = 0.0
            ramp = min(1.0, max(0.0, (abs(error) - self.far_threshold) / self.far_threshold))
            far_gain = 1.0 + (self.far_boost - 1.0) * ramp
            gain = self.kp * max(far_gain, near_gain)
            # The transport splits this frame budget into <=20-count HID reports.
            limit = min(240, max(1, self.max_step))
            integral_gain = max(0.0, min(1.0, (24.0 - distance) / 16.0))
            raw = abs(gain * error + self.ki * self.integral[axis] * integral_gain)
            step = math.copysign(min(limit, max(1, math.floor(raw + 0.5))), error)
            # Only the final few pixels get a small corrective pulse. Large
            # pursuit steps keep their existing speed and 240-count ceiling.
            if distance <= 12.0:
                micro_limit = max(1, int(round(distance * 0.35)))
                if self.ki * abs(self.integral[axis]) * integral_gain >= 0.5:
                    micro_limit += 1  # Persistent small error gets one extra corrective count.
                step = math.copysign(min(abs(step), micro_limit), error)
            # A large previous pull can still be finishing in the HID queue.
            # Bound the next near-target correction to the remaining error.
            if braking:
                brake_limit = max(1, int(round(abs(error) * 0.32)))
                step = math.copysign(min(abs(step), brake_limit), error)
            steps.append(step)
        self.last_error = list(errors)
        if max(map(abs, steps)) >= 70:
            self.last_large_move_at = now
        return tuple(map(int, steps))


class AimSanityGate:
    """Confirm a new real body before a large move; ignore top HUD detections."""

    def __init__(self, confirm_frames=2, min_y=120.0, max_y=610.0):
        self.confirm_frames = int(confirm_frames)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.reset()

    def reset(self):
        self.track_id = None
        self.hits = 0

    def confirmed(self, track_id):
        return self.track_id == track_id and self.hits >= self.confirm_frames

    def allow(self, track_id, point_y, observed=True):
        if not observed or not math.isfinite(point_y):
            return False, "NO REAL BODY"
        if not self.min_y <= point_y <= self.max_y:
            return False, "OUTSIDE AIM ZONE"
        if track_id != self.track_id:
            self.track_id = track_id
            self.hits = 1
        else:
            self.hits = min(self.confirm_frames, self.hits + 1)
        if self.hits < self.confirm_frames:
            return False, "CONFIRM NEW BOT"
        return True, ""


class TargetLead:
    """Predict only a locked person's independent screen motion, not camera pan."""

    def __init__(self, max_lead_px=20.0):
        self.max_lead_px = float(max_lead_px)
        self.reset()

    def reset(self):
        self.track_id = None
        self.last_body_center = None
        self.last_time = None
        self.last_centers = {}
        self.velocity = (0.0, 0.0)
        self.motion_hits = 0
        self.head_offset_ratio = None
        self.last_body_height = None
        self.filtered_point = None

    def update(self, track_id, body_box, head_point, body_centers, now,
               head_mode=True, observed=True, lead_s=0.035, last_step=0,
               camera_shift=None):
        """Return a point only while this body's detection is real."""
        if self.track_id != track_id:
            self.reset()
            self.track_id = track_id
        if not observed:
            return None  # Keep predicted boxes for association, never for HID motion.
        bx, by, bw, bh = map(float, body_box)
        cx, cy = bx + bw / 2, by + bh / 2
        centers = dict(body_centers)
        posture_changed = (self.last_body_height is not None and
                           abs(bh - self.last_body_height) / max(1.0, self.last_body_height) > 0.18)

        if not head_mode:
            point = (cx, cy)
            source = "BODY"
        elif head_point is not None:
            point = tuple(map(float, head_point))
            source = "REAL HEAD"
        elif self.head_offset_ratio is not None:
            point = (cx + self.head_offset_ratio[0] * bw,
                     cy + self.head_offset_ratio[1] * bh)
            source = "HELD HEAD"
        else:
            point = (cx, by + 0.125 * bh)
            source = "BODY ANCHOR"

        # Suppress small detector-box wobble without delaying a large strafe.
        if posture_changed:
            self.filtered_point = None  # Do not smooth a crouch head toward its old standing position.
        if self.filtered_point is not None:
            delta = math.hypot(point[0] - self.filtered_point[0],
                               point[1] - self.filtered_point[1])
            alpha = 0.62 + 0.38 * min(1.0, max(0.0, (delta - 6.0) / 12.0))
            point = (self.filtered_point[0] + alpha * (point[0] - self.filtered_point[0]),
                     self.filtered_point[1] + alpha * (point[1] - self.filtered_point[1]))
        self.filtered_point = point
        if head_point is not None and head_mode:
            self.head_offset_ratio = ((point[0] - cx) / max(1.0, bw),
                                      (point[1] - cy) / max(1.0, bh))
        self.last_body_height = bh

        lead_x = lead_y = 0.0
        if self.last_body_center is not None and self.last_time is not None:
            dt = now - self.last_time
            if 0.008 <= dt <= 0.15:
                other_shifts = []
                for other_id, center in centers.items():
                    if other_id != track_id and other_id in self.last_centers:
                        old = self.last_centers[other_id]
                        other_shifts.append((center[0] - old[0], center[1] - old[1]))
                if camera_shift is not None:
                    camera_x, camera_y = camera_shift
                    can_predict = (True, True)
                elif other_shifts:
                    shifts_x = sorted(shift[0] for shift in other_shifts)
                    shifts_y = sorted(shift[1] for shift in other_shifts)
                    middle = len(other_shifts) // 2
                    camera_x, camera_y = shifts_x[middle], shifts_y[middle]
                    can_predict = (True, True)
                else:
                    camera_x = camera_y = 0.0
                    # With one bot, camera pan moves its box opposite to HID motion.
                    # Motion *with* the commanded direction is thus a conservative
                    # lower bound on the bot's independent strafe, not camera pan.
                    if isinstance(last_step, (tuple, list)):
                        step_x, step_y = last_step
                        can_predict = (abs(step_x) <= 8 or
                                       (cx - self.last_body_center[0]) * step_x > 0,
                                       abs(step_y) <= 8 or
                                       (cy - self.last_body_center[1]) * step_y > 0)
                    else:
                        can_predict = (abs(last_step) <= 8,) * 2
                dx = cx - self.last_body_center[0] - camera_x
                dy = cy - self.last_body_center[1] - camera_y
                self.motion_hits = (min(3, self.motion_hits + 1)
                                    if dx * self.velocity[0] + dy * self.velocity[1] > 0
                                    else 0)
                velocity = []
                for previous, delta, allowed in zip(self.velocity, (dx, dy), can_predict):
                    if allowed:
                        noise_floor = max(0.5, min(1.2, bw * 0.025))
                        raw = max(-500.0, min(500.0,
                                  (delta if abs(delta) >= noise_floor else 0.0) / dt))
                        velocity.append(previous * 0.55 + raw * 0.45)
                    else:
                        velocity.append(previous * 0.5)
                self.velocity = tuple(velocity)
            else:
                self.velocity = (0.0, 0.0)
                self.motion_hits = 0
        else:
            self.velocity = (0.0, 0.0)
            self.motion_hits = 0
        if observed:
            self.last_body_center = (cx, cy)
            self.last_time = now
            self.last_centers = centers
            speed = math.hypot(*self.velocity)
            confidence = min(1.0, self.motion_hits / 2.0)
            dynamic_s = max(0.0, lead_s) * (0.2 + 0.8 * min(1.0, speed / 140.0) * confidence)
            # A distant head can be only a few pixels wide at 640 input.
            # Scale the prediction ceiling to the observed body width.
            lead_limit = min(self.max_lead_px, max(1.0, bw * 0.25))
            lead_x = max(-lead_limit, min(lead_limit,
                         self.velocity[0] * dynamic_s))
            lead_y = max(-lead_limit, min(lead_limit,
                         self.velocity[1] * dynamic_s))
        return point[0] + lead_x, point[1] + lead_y, source, lead_x, lead_y
