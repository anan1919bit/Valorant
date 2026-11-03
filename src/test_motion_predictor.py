import unittest

import cv2
import numpy as np

from motion_predictor import BackgroundFlow, PredictedBridge


class MotionPredictorTests(unittest.TestCase):
    def test_background_shift_is_measured_without_target(self):
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)
        shifted = cv2.warpAffine(base, np.float32([[1, 0, 4], [0, 1, -3]]),
                                 (480, 480), borderMode=cv2.BORDER_REFLECT)
        flow = BackgroundFlow()
        self.assertIsNone(flow.update(base, (200, 200, 30, 60)))
        shift = flow.update(shifted, (200, 200, 30, 60))
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(shift[0], 4 * 640 / 480, delta=1.5)
        self.assertAlmostEqual(shift[1], -3 * 640 / 480, delta=1.5)

    def test_bridge_is_short_stable_and_bound_to_confirmed_track(self):
        bridge = PredictedBridge()
        self.assertIsNone(bridge.predict(1, (100, 100, 12, 40), 1, None, 1.02))
        bridge.observe(1, (106, 108), (100, 100, 12, 40), (120, 0), 1.0)
        first = bridge.predict(1, (101, 100, 12, 40), 1, (-1, 0), 1.02)
        self.assertIsNotNone(first)
        self.assertEqual(first[2], "PREDICTED BRIDGE")
        self.assertLessEqual(abs(first[0] - 106), 4.2)
        self.assertEqual(bridge.camera_since_real(1, (-2, 0)), (-3, 0))
        self.assertIsNone(bridge.predict(2, (101, 100, 12, 40), 1, None, 1.02))
        self.assertIsNone(bridge.predict(1, (102, 100, 12, 40), 3, None, 1.04))
        self.assertIsNone(bridge.predict(1, (102, 100, 12, 40), 2, None, 1.10))


if __name__ == "__main__":
    unittest.main()
