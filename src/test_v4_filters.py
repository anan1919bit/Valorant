import unittest

import numpy as np

from person_aim import AimSanityGate, DampedController, TargetLead, match_head
from target_policy import StandingBodyFilter


class V4FilterTests(unittest.TestCase):
    def test_camera_motion_is_removed_from_single_target_velocity(self):
        compensated = TargetLead()
        uncompensated = TargetLead()
        for lead in (compensated, uncompensated):
            lead.update(1, (100, 100, 20, 60), (110, 108), {1: (110, 130)}, 1.0)
        compensated.update(1, (104, 100, 20, 60), (114, 108),
                           {1: (114, 130)}, 1.02, camera_shift=(3, 0))
        uncompensated.update(1, (104, 100, 20, 60), (114, 108),
                             {1: (114, 130)}, 1.02)
        self.assertGreater(uncompensated.velocity[0], compensated.velocity[0])
        self.assertGreater(compensated.velocity[0], 0)

    def test_predicted_bridge_requires_confirmed_real_track(self):
        gate = AimSanityGate(confirm_frames=2)
        self.assertFalse(gate.confirmed(7))
        self.assertFalse(gate.allow(7, 300)[0])
        self.assertFalse(gate.confirmed(7))
        self.assertTrue(gate.allow(7, 300)[0])
        self.assertTrue(gate.confirmed(7))
        self.assertFalse(gate.confirmed(8))

    def test_floor_remnant_after_body_disappears(self):
        filt = StandingBodyFilter()
        self.assertEqual(filt.filter(np.array([[100, 100, 40, 80]]), [0], 1.0).tolist(), [0])
        self.assertEqual(filt.filter(np.array([[105, 151, 35, 31]]), [0], 1.04).tolist(), [])
        self.assertEqual(filt.filter(np.array([[102, 171, 44, 12]]), [0], 1.08).tolist(), [])
        self.assertEqual(filt.filter(np.array([[250, 100, 40, 80]]), [0], 1.12).tolist(), [0])

    def test_crouched_body_continues_without_head_and_accepts_real_head(self):
        filt = StandingBodyFilter()
        self.assertEqual(filt.filter(np.array([[100, 100, 40, 80]]), [0], 1.0).tolist(), [0])
        crouch = np.array([[100, 140, 40, 50]])
        self.assertEqual(filt.filter(crouch, [0], 1.03).tolist(), [0])
        head = np.array([[110, 116, 20, 18]])
        self.assertEqual(filt.filter(crouch, [0], 1.06, head).tolist(), [0])
        self.assertIsNotNone(match_head(crouch[0], head))

    def test_distant_color_crop_needs_real_head_to_pass_tiny_filter(self):
        body = np.array([[319.2, 280.1, 6.5, 21.0]], dtype=np.float32)
        head = np.array([[321.4, 280.1, 2.0, 5.0]], dtype=np.float32)
        self.assertEqual(StandingBodyFilter().filter(body, [0], 1.0, head).tolist(), [])
        self.assertEqual(StandingBodyFilter().filter(
            body, [0], 1.0, head, allow_tiny_confirmed=True).tolist(), [0])
        self.assertEqual(StandingBodyFilter().filter(
            body, [0], 1.0, None, allow_tiny_confirmed=True).tolist(), [])

    def test_distant_full_frame_body_with_real_head_is_accepted(self):
        body = np.array([[318, 279, 13, 22]], dtype=np.float32)
        head = np.array([[322, 279, 5, 6]], dtype=np.float32)
        self.assertEqual(StandingBodyFilter().filter(body, [0], 1.0, head).tolist(), [])
        self.assertEqual(StandingBodyFilter().filter(
            body, [0], 1.0, head, allow_tiny_confirmed=True).tolist(), [0])
        self.assertEqual(StandingBodyFilter().filter(
            body, [0], 1.0, None, allow_tiny_confirmed=True).tolist(), [])

    def test_distant_lead_stays_within_small_target_width(self):
        def sample(width):
            lead = TargetLead(max_lead_px=20)
            result = None
            for i, x in enumerate((100, 104, 108, 112)):
                result = lead.update(1, (x, 100, width, 50), (x + width / 2, 107),
                                     {1: (x + width / 2, 125)}, 1 + i * 0.02,
                                     lead_s=0.08, last_step=(0, 0))
            return result[3]
        self.assertLessEqual(sample(12), 3.0)
        self.assertGreater(sample(100), sample(12))

    def test_crouch_scales_held_head_offset_and_drops_old_smoothing(self):
        lead = TargetLead()
        lead.update(1, (100, 100, 40, 80), (120, 110), {1: (120, 140)}, 1.0,
                    lead_s=0)
        held = lead.update(1, (100, 140, 40, 50), None, {1: (120, 165)}, 1.04,
                           lead_s=0)
        self.assertEqual(held[2], "HELD HEAD")
        self.assertAlmostEqual(held[1], 146.25)
        real = lead.update(1, (100, 140, 40, 50), (120, 130), {1: (120, 165)}, 1.08,
                           lead_s=0)
        self.assertLess(real[1], 140)

    def test_single_bot_chase_keeps_conservative_lead(self):
        lead = TargetLead()
        lead.update(1, (90, 90, 20, 80), (100, 95), {1: (100, 130)}, 1.0)
        point = lead.update(1, (100, 90, 20, 80), (110, 95),
                            {1: (110, 130)}, 1.05, last_step=(100, 0))
        self.assertGreater(point[3], 0)
        lead.reset()
        lead.update(1, (90, 90, 20, 80), (100, 95), {1: (100, 130)}, 1.0)
        pan = lead.update(1, (80, 90, 20, 80), (90, 95),
                          {1: (90, 130)}, 1.05, last_step=(100, 0))
        self.assertEqual(pan[3], 0)

    def test_small_head_wobble_is_filtered_but_large_move_is_immediate(self):
        lead = TargetLead()
        lead.update(1, (90, 90, 20, 80), (100, 95), {1: (100, 130)}, 1.0)
        wobble = lead.update(1, (90, 90, 20, 80), (102, 95),
                             {1: (100, 130)}, 1.05)
        self.assertGreater(wobble[0], 100)
        self.assertLess(wobble[0], 102)
        dash = lead.update(1, (90, 90, 20, 80), (130, 95),
                           {1: (100, 130)}, 1.10)
        self.assertEqual(dash[0], 130)

    def test_close_correction_does_not_slow_far_pursuit(self):
        control = DampedController(kp=0.413, max_step=240, deadzone=4,
                                   near_boost=2.5, near_radius=130)
        self.assertLessEqual(abs(control.update(6, 0, 1.0)[0]), 2)
        self.assertEqual(control.update(300, 0, 1.1)[0], 124)

    def test_dynamic_lead_grows_only_with_consistent_motion(self):
        lead = TargetLead()
        values = []
        for n, x in enumerate((100, 103, 106, 109)):
            result = lead.update(1, (x, 100, 30, 70), (x + 15, 110),
                                 {1: (x + 15, 135)}, 1 + n * 0.02,
                                 lead_s=0.025, last_step=(0, 0))
            values.append(result[3])
        self.assertEqual(values[0], 0)
        self.assertGreater(values[-1], values[1])

    def test_separation_integral_only_on_persistent_near_error(self):
        control = DampedController(kp=0.419, ki=0.4, deadzone=1.5,
                                   near_boost=2.5, near_radius=150)
        steps = [control.update(4, 0, 1 + n * 0.02)[0] for n in range(40)]
        self.assertEqual(steps[0], 1)
        self.assertGreater(steps[-1], steps[0])
        self.assertEqual(control.update(-4, 0, 2.0)[0], 0)
        self.assertEqual(control.integral[0], 0)
        self.assertEqual(control.update(-4, 0, 2.02)[0], -1)

    def test_small_reversal_pauses_once_but_large_reversal_is_immediate(self):
        control = DampedController(kp=0.394, deadzone=1.5,
                                   near_boost=2.1, near_radius=80)
        self.assertGreater(control.update(5, 0, 1.0)[0], 0)
        self.assertEqual(control.update(-5, 0, 1.02)[0], 0)
        self.assertLess(control.update(-5, 0, 1.04)[0], 0)
        self.assertGreater(control.update(30, 0, 1.06)[0], 0)

    def test_large_pull_brakes_only_on_arrival(self):
        control = DampedController(kp=0.405, max_step=194, deadzone=1.5,
                                   far_boost=2.1, far_threshold=180,
                                   near_boost=2.1, near_radius=80)
        self.assertEqual(control.update(470, 0, 1.0)[0], 194)
        self.assertGreater(control.update(220, 0, 1.02)[0], 50)
        self.assertLessEqual(control.update(100, 0, 1.04)[0], 32)
        self.assertLessEqual(control.update(20, 0, 1.06)[0], 6)
        self.assertGreater(control.update(100, 0, 1.30)[0], 32)

if __name__ == "__main__":
    unittest.main()
