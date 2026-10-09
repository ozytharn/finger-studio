from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from time import perf_counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .core import HandObservation


class CameraProcessor:
    def __init__(self, camera_index: int = 0, width: int = 1920, height: int = 1080) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None
        model_path = Path(__file__).with_name("hand_landmarker.task")
        if not model_path.exists():
            try:
                urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/1/hand_landmarker.task", model_path)
            except Exception as error:
                raise RuntimeError(
                    "MediaPipe hand model could not be downloaded. "
                    "Check internet access and restart the app.") from error
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            # Lower gates preserve detections when the palm is cropped or partly
            # occluded; temporal VIDEO tracking and the one-frame queue prevent
            # the relaxed gates from creating a stale-frame latency penalty.
            min_hand_detection_confidence=0.20,
            min_hand_presence_confidence=0.15, min_tracking_confidence=0.15)
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.inference_ms = 0.0

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError("Unable to open the webcam. Check camera permissions and connection.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.timestamp_ms = 0

    def read(self) -> Tuple[object, List[HandObservation]]:
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("The webcam stopped returning frames.")
        frame = cv2.flip(frame, 1)
        # Use the complete camera frame for inference. Small isolated fingers
        # disappear when the image is aggressively reduced before landmarking.
        inference_frame = frame
        max_inference_width = 1280
        if frame.shape[1] > max_inference_width:
            scale = max_inference_width / frame.shape[1]
            inference_frame = cv2.resize(
                frame, (max_inference_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA)
        self.timestamp_ms += 1
        started = perf_counter()
        result = self._detect(inference_frame)
        # A second, enlarged pass is only used when the full-frame detector finds
        # nothing. This improves recovery for a lone finger near an edge without
        # adding latency to normal frames.
        if not result.hand_landmarks:
            enlarged = cv2.resize(frame, None, fx=1.35, fy=1.35,
                                  interpolation=cv2.INTER_CUBIC)
            self.timestamp_ms += 1
            result = self._detect(enlarged)
        self.inference_ms = (perf_counter() - started) * 1000
        observations: List[HandObservation] = []
        if result.hand_landmarks and result.handedness:
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                # MediaPipe Tasks returns one list of categories per hand.
                category = handedness[0]
                points = {index: (landmark.x, landmark.y)
                          for index, landmark in enumerate(landmarks)}
                observations.append(HandObservation(
                    category.category_name, points, category.score))
        return frame, observations

    def _detect(self, frame: object):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp_ms)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.hands.close()
