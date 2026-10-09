import unittest

from tracer.core import FINGER_LANDMARKS, HandObservation, StableTracker, Trail, map_point


def hand(label="Right", index=(0.5, 0.5), thumb=(0.4, 0.5)):
    return HandObservation(label, {FINGER_LANDMARKS["Index"]: index,
                                   FINGER_LANDMARKS["Thumb"]: thumb})


class CoreTests(unittest.TestCase):
    def test_tracker_keeps_selected_hand_and_finger(self):
        tracker = StableTracker()
        tracker.select("Left", "Thumb")
        self.assertEqual(tracker.update([hand("Right"), hand("Left", thumb=(0.2, 0.3))]), (0.2, 0.3))
        self.assertIsNone(tracker.update([hand("Right", thumb=(0.8, 0.8))]))

    def test_tracker_rejects_implausible_jump(self):
        tracker = StableTracker()
        self.assertEqual(tracker.update([hand(index=(0.2, 0.2))]), (0.2, 0.2))
        self.assertIsNone(tracker.update([hand(index=(0.9, 0.9))]))

    def test_tracker_allows_fast_but_plausible_motion(self):
        tracker = StableTracker()
        self.assertEqual(tracker.update([hand(index=(0.2, 0.2))]), (0.2, 0.2))
        self.assertIsNotNone(tracker.update([hand(index=(0.5, 0.5))]))

    def test_trail_breaks_and_clears(self):
        trail = Trail()
        trail.start()
        trail.add((1, 1))
        trail.break_segment()
        trail.add((2, 2))
        self.assertEqual(trail.segments, [[(1, 1)], [], [(2, 2)]])
        trail.clear()
        self.assertEqual(trail.segments, [])

    def test_map_point_clamps_and_mirrors(self):
        self.assertEqual(map_point((0, 0), (0, 0, 1, 1), 100, 100), (100, 0))
        self.assertEqual(map_point((2, 2), (0, 0, 1, 1), 100, 100), (0, 100))

    def test_trail_history_is_bounded(self):
        trail = Trail(max_points=3)
        trail.start()
        for point in ((1, 1), (2, 2), (3, 3), (4, 4)):
            trail.add(point)
        self.assertEqual(sum(len(segment) for segment in trail.segments), 3)
        self.assertEqual(trail.segments[-1][-1], (4, 4))

    def test_tracker_accepts_detected_finger_observation(self):
        tracker = StableTracker()
        observation = HandObservation(
            "Right", {FINGER_LANDMARKS["Index"]: (0.4, 0.4)}, 0.95)
        self.assertEqual(tracker.update([observation]), (0.4, 0.4))

    def test_tracker_updates_multiple_selected_fingers(self):
        tracker = StableTracker()
        tracker.select_many("Right", ("Index", "Middle"))
        observation = HandObservation("Right", {
            FINGER_LANDMARKS["Index"]: (0.4, 0.4),
            FINGER_LANDMARKS["Middle"]: (0.6, 0.6),
        })
        self.assertEqual(tracker.update_many([observation]), {
            "Index": (0.4, 0.4), "Middle": (0.6, 0.6)})


if __name__ == "__main__":
    unittest.main()
