from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import monotonic
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
FINGER_LANDMARKS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}


@dataclass(frozen=True)
class HandObservation:
    label: str
    landmarks: Dict[int, Point]
    confidence: float = 1.0

    def fingertip(self, finger: str) -> Optional[Point]:
        index = FINGER_LANDMARKS.get(finger)
        return self.landmarks.get(index) if index is not None else None


class StableTracker:
    """Keeps hand/finger identity stable and emits None during interruptions."""

    def __init__(self, smoothing: float = 0.65, jump_limit: float = 0.28,
                 loss_limit: int = 8) -> None:
        self.smoothing = smoothing
        self.jump_limit = jump_limit
        self.loss_limit = loss_limit
        self.selected_hand = "Either"
        self.selected_finger = "Index"
        self.selected_fingers = ("Index",)
        self._smoothed: Dict[str, Point] = {}
        self._lost_frames: Dict[str, int] = {}
        self._last_time: Dict[str, float] = {}
        self._velocity: Dict[str, Point] = {}

    def select(self, hand: str, finger: str) -> None:
        self.select_many(hand, (finger,))

    def select_many(self, hand: str, fingers: Tuple[str, ...]) -> None:
        if hand not in ("Either", "Left", "Right"):
            raise ValueError("hand must be Either, Left, or Right")
        if not fingers or any(finger not in FINGER_NAMES for finger in fingers):
            raise ValueError("select at least one known finger")
        self.selected_hand = hand
        self.selected_finger = fingers[0]
        selected = set(fingers)
        self.selected_fingers = tuple(fingers)
        self._smoothed = {key: value for key, value in self._smoothed.items() if key in selected}
        self._lost_frames = {key: value for key, value in self._lost_frames.items() if key in selected}
        self._last_time = {key: value for key, value in self._last_time.items() if key in selected}
        self._velocity = {key: value for key, value in self._velocity.items() if key in selected}

    def update(self, observations: List[HandObservation]) -> Optional[Point]:
        return self.update_many(observations).get(self.selected_finger)

    def update_many(self, observations: List[HandObservation]) -> Dict[str, Point]:
        candidates = [item for item in observations
                      if self.selected_hand == "Either" or item.label == self.selected_hand]
        observation = candidates[0] if candidates else None
        result: Dict[str, Point] = {}
        for finger in FINGER_NAMES:
            if finger not in self._selected_fingers():
                continue
            point = observation.fingertip(finger) if observation else None
            if point is None:
                self._lost_frames[finger] = self._lost_frames.get(finger, 0) + 1
                if self._lost_frames[finger] >= self.loss_limit:
                    self._smoothed.pop(finger, None)
                continue
            self._lost_frames[finger] = 0
            previous = self._smoothed.get(finger)
            if previous is None:
                self._smoothed[finger] = point
                self._last_time[finger] = monotonic()
                result[finger] = point
                continue
            now = monotonic()
            elapsed = max(1 / 120, now - self._last_time.get(finger, now))
            distance = hypot(point[0] - previous[0], point[1] - previous[1])
            speed = distance / elapsed
            self._velocity[finger] = (point[0] - previous[0], point[1] - previous[1])
            allowed_jump = min(0.45, max(self.jump_limit, speed * 0.04))
            if distance > allowed_jump:
                continue
            alpha = self.smoothing
            smoothed = (previous[0] + alpha * (point[0] - previous[0]),
                        previous[1] + alpha * (point[1] - previous[1]))
            self._smoothed[finger] = smoothed
            self._last_time[finger] = now
            result[finger] = smoothed
        return result

    def _selected_fingers(self) -> Tuple[str, ...]:
        return self.selected_fingers


class Trail:
    def __init__(self, max_points: int = 10000) -> None:
        self.max_points = max_points
        self.segments: List[List[Point]] = []
        self.active = False
        self.paused = False

    def start(self) -> None:
        self.active, self.paused = True, False

    def pause(self) -> None:
        if self.active:
            self.paused = True

    def resume(self) -> None:
        if self.active:
            self.paused = False

    def stop(self) -> None:
        self.active, self.paused = False, False

    def clear(self) -> None:
        self.segments.clear()

    def add(self, point: Optional[Point]) -> None:
        if not self.active or self.paused or point is None:
            return
        if not self.segments or not self.segments[-1]:
            self.segments.append([point])
        else:
            self.segments[-1].append(point)
        while sum(len(segment) for segment in self.segments) > self.max_points:
            if len(self.segments[0]) <= 1:
                self.segments.pop(0)
            else:
                self.segments[0].pop(0)

    def break_segment(self) -> None:
        if self.active and self.segments and self.segments[-1]:
            self.segments.append([])


def map_point(point: Point, bounds: Tuple[float, float, float, float],
              width: int, height: int, mirror: bool = True) -> Point:
    left, top, right, bottom = bounds
    x = min(1.0, max(0.0, (point[0] - left) / max(0.001, right - left)))
    y = min(1.0, max(0.0, (point[1] - top) / max(0.001, bottom - top)))
    if mirror:
        x = 1.0 - x
    return x * width, y * height
