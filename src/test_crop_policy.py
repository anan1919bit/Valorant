import unittest

from crop_policy import choose_color_candidate, should_zoom_lock


class CropPolicyTests(unittest.TestCase):
    def test_small_real_lock_zooms_except_full_refresh(self):
        box = (300, 280, 13, 22)
        self.assertTrue(should_zoom_lock(box, 1))
        self.assertFalse(should_zoom_lock(box, 5))
        self.assertFalse(should_zoom_lock((300, 280, 40, 90), 1))
        self.assertFalse(should_zoom_lock(None, 1))

    def test_probe_rotates_without_lock_but_stays_near_lock(self):
        proposals = [(100, 200, 20, 30), (300, 200, 20, 30), (500, 200, 20, 30)]
        self.assertEqual(choose_color_candidate(proposals, 0), proposals[:1])
        self.assertEqual(choose_color_candidate(proposals, 1), proposals[1:2])
        self.assertEqual(choose_color_candidate(proposals, 2), proposals[2:3])
        self.assertEqual(choose_color_candidate(proposals, 3), proposals[:1])
        self.assertEqual(choose_color_candidate(proposals, 0, (310, 215)), proposals[1:2])
        self.assertEqual(choose_color_candidate([], 0), [])


if __name__ == "__main__":
    unittest.main()
