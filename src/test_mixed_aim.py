import unittest

from mixed_aim import MixedAimPolicy


class MixedAimTests(unittest.TestCase):
    def test_one_third_head_by_stable_track_id(self):
        policy = MixedAimPolicy()
        self.assertEqual([policy.head_for(i) for i in range(1, 7)],
                         [False, False, True, False, False, True])
        self.assertFalse(policy.head_for(1))
        self.assertTrue(policy.head_for(3))


if __name__ == "__main__":
    unittest.main()
