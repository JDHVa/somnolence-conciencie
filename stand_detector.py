"""
stand_detector.py — Detector para las vistas del stand.
Expone landmarks crudos + historial de señales para las visualizaciones.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import time
import numpy as np
import urllib.request
import os
import threading
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detector import (
    LEFT_EYE, RIGHT_EYE,
    MOUTH_OUTER_LEFT, MOUTH_OUTER_RIGHT,
    MOUTH_TOP_INNER, MOUTH_BOTTOM_INNER,
    MOUTH_TOP_L, MOUTH_BOTTOM_L, MOUTH_TOP_R, MOUTH_BOTTOM_R,
    HEAD_POSE_IDX, FACE_MODEL_3D,
    EAR_RATIO_CLOSED, EAR_FALLBACK, CALIB_SECONDS, SMOOTH_WINDOW,
    PERCLOS_WINDOW_SEC, MAR_YAWN_THRESHOLD,
    HEAD_PITCH_DOWN, HEAD_PITCH_WINDOW,
    FATIGUE_TIRED, FATIGUE_DROWSY, FATIGUE_ALERT, CLOSED_SECONDS,
    eye_aspect_ratio, mouth_aspect_ratio, estimate_head_pose, apply_clahe,
    _download_model, MODEL_PATH,
)

FRAME_SKIP = 2
HISTORY_SECONDS = 30


class StandDetector:

    def __init__(self, camera_index=0):
        _download_model()
        self.camera_index = camera_index
        self.cap = None
        self._lock = threading.Lock()
        self._raw_frame_bytes = None
        self._frame_count = 0
        self._t_start = time.time()

        self._ear_buffer = deque(maxlen=SMOOTH_WINDOW)
        self._ear_baseline_samples = []
        self._ear_baseline = None
        self._ear_threshold = EAR_FALLBACK
        self._eye_history = deque()
        self._eyes_closed_since = None
        self._was_closed = False
        self._blink_start = None
        self._blink_events = deque()
        self._yawn_start = None
        self._yawn_events = deque()
        self._nod_start = None

        self._hist_ear = deque()
        self._hist_mar = deque()
        self._hist_perclos = deque()
        self._hist_score = deque()

        self._state = {
            "face_detected": False, "ear": 0.0, "ear_threshold": EAR_FALLBACK,
            "mar": 0.0, "perclos": 0.0, "fatigue_score": 0.0, "level": "NORMAL",
            "eyes_closed": False, "yawning": False, "head_nodding": False,
            "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
            "calibrating": True, "calib_progress": 0.0, "alert": False,
            "closed_time": 0.0, "blinks_per_min": 0, "yawns_per_min": 0,
            "landmarks": [],
            "eye_left_pts": [], "eye_right_pts": [], "mouth_pts": [], "head_pose_pts": [],
            "ear_a_left": 0.0, "ear_b_left": 0.0, "ear_c_left": 0.0,
            "ear_a_right": 0.0, "ear_b_right": 0.0, "ear_c_right": 0.0,
            "frame_width": 640, "frame_height": 480,
            "history": {"ear": [], "mar": [], "perclos": [], "score": []},
        }

        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._detector = FaceLandmarker.create_from_options(options)

    def start(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            print(f"No se pudo abrir camara {self.camera_index}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Camara abierta: indice {self.camera_index}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            mean_lum = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            if mean_lum < 70:
                frame = apply_clahe(frame)

            self._frame_count += 1
            if self._frame_count % FRAME_SKIP == 0:
                self._process(frame)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._raw_frame_bytes = buf.tobytes()

    def _process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        now = time.time()
        self._state["frame_width"] = w
        self._state["frame_height"] = h

        if not result.face_landmarks:
            self._state.update({"face_detected": False, "landmarks": [],
                                "eye_left_pts": [], "eye_right_pts": [],
                                "mouth_pts": [], "head_pose_pts": []})
            self._eyes_closed_since = None
            return

        lms = result.face_landmarks[0]

        ear_l = eye_aspect_ratio(lms, LEFT_EYE, w, h)
        ear_r = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
        ear_raw = (ear_l + ear_r) / 2.0
        self._ear_buffer.append(ear_raw)
        ear = float(np.mean(self._ear_buffer))

        elapsed_total = now - self._t_start
        calibrating = elapsed_total < CALIB_SECONDS
        if calibrating:
            self._ear_baseline_samples.append(ear_raw)
            self._state["calib_progress"] = float(elapsed_total / CALIB_SECONDS)
        elif self._ear_baseline is None and self._ear_baseline_samples:
            self._ear_baseline = float(np.median(self._ear_baseline_samples))
            self._ear_threshold = max(0.15, self._ear_baseline * EAR_RATIO_CLOSED)
            self._state["calib_progress"] = 1.0

        eyes_closed = ear < self._ear_threshold

        if eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            closed_time = now - self._eyes_closed_since
        else:
            self._eyes_closed_since = None
            closed_time = 0.0

        if eyes_closed and not self._was_closed:
            self._blink_start = now
        elif not eyes_closed and self._was_closed and self._blink_start:
            dur = now - self._blink_start
            if 0.08 <= dur <= 0.50:
                self._blink_events.append(now)
            self._blink_start = None
        self._was_closed = eyes_closed
        cut60 = now - 60
        while self._blink_events and self._blink_events[0] < cut60:
            self._blink_events.popleft()
        blinks_per_min = len(self._blink_events)

        self._eye_history.append((now, eyes_closed))
        cutoff = now - PERCLOS_WINDOW_SEC
        while self._eye_history and self._eye_history[0][0] < cutoff:
            self._eye_history.popleft()
        perclos = (sum(1 for _, c in self._eye_history if c) / len(self._eye_history)
                   if self._eye_history else 0.0)

        mar = mouth_aspect_ratio(lms, w, h)
        yawning = mar > MAR_YAWN_THRESHOLD
        if yawning:
            if self._yawn_start is None:
                self._yawn_start = now
            elif (now - self._yawn_start >= 1.5
                  and (not self._yawn_events or now - self._yawn_events[-1] > 3)):
                self._yawn_events.append(now)
        else:
            self._yawn_start = None
        while self._yawn_events and self._yawn_events[0] < cut60:
            self._yawn_events.popleft()
        yawns_per_min = len(self._yawn_events)

        pitch, yaw, roll = estimate_head_pose(lms, w, h)
        if pitch > HEAD_PITCH_DOWN:
            if self._nod_start is None:
                self._nod_start = now
            head_nodding = (now - self._nod_start) >= HEAD_PITCH_WINDOW
        else:
            self._nod_start = None
            head_nodding = False

        s_perclos = float(np.clip(perclos / 0.40, 0, 1))
        s_closed = float(np.clip(closed_time / CLOSED_SECONDS, 0, 1))
        s_yawn = 1.0 if yawning else 0.0
        s_nod = 1.0 if head_nodding else 0.0
        s_blink = float(np.clip((blinks_per_min - 20) / 20.0, 0, 1))
        score = float(np.clip(
            0.35*s_perclos + 0.30*s_closed + 0.15*s_nod + 0.12*s_yawn + 0.08*s_blink, 0, 1))

        if closed_time >= CLOSED_SECONDS or score >= FATIGUE_ALERT:
            level = "ALERTA"
        elif score >= FATIGUE_DROWSY:
            level = "SOMNOLIENTO"
        elif score >= FATIGUE_TIRED:
            level = "CANSADO"
        else:
            level = "NORMAL"

        landmarks_norm = [[float(lm.x), float(lm.y)] for lm in lms]
        eye_left_pts = [[lms[i].x * w, lms[i].y * h] for i in LEFT_EYE]
        eye_right_pts = [[lms[i].x * w, lms[i].y * h] for i in RIGHT_EYE]

        def dist(p1, p2):
            return float(np.linalg.norm(np.array(p1) - np.array(p2)))

        mouth_pts = [
            [lms[MOUTH_OUTER_LEFT].x * w, lms[MOUTH_OUTER_LEFT].y * h],
            [lms[MOUTH_OUTER_RIGHT].x * w, lms[MOUTH_OUTER_RIGHT].y * h],
            [lms[MOUTH_TOP_INNER].x * w, lms[MOUTH_TOP_INNER].y * h],
            [lms[MOUTH_BOTTOM_INNER].x * w, lms[MOUTH_BOTTOM_INNER].y * h],
        ]
        head_pose_pts = [
            [lms[HEAD_POSE_IDX[k]].x * w, lms[HEAD_POSE_IDX[k]].y * h]
            for k in ["nose_tip", "chin", "left_eye_l", "right_eye_r", "mouth_l", "mouth_r"]
        ]

        self._hist_ear.append((now, ear))
        self._hist_mar.append((now, mar))
        self._hist_perclos.append((now, perclos))
        self._hist_score.append((now, score))
        for hist in [self._hist_ear, self._hist_mar, self._hist_perclos, self._hist_score]:
            cut = now - HISTORY_SECONDS
            while hist and hist[0][0] < cut:
                hist.popleft()

        with self._lock:
            self._state.update({
                "face_detected": True, "ear": round(ear, 3),
                "ear_threshold": round(float(self._ear_threshold), 3),
                "mar": round(mar, 3), "perclos": round(perclos, 3),
                "fatigue_score": round(score, 3), "level": level,
                "eyes_closed": bool(eyes_closed), "yawning": bool(yawning),
                "head_nodding": bool(head_nodding),
                "head_pitch": round(pitch, 1), "head_yaw": round(yaw, 1), "head_roll": round(roll, 1),
                "calibrating": bool(calibrating), "alert": bool(level == "ALERTA"),
                "closed_time": round(closed_time, 1),
                "blinks_per_min": blinks_per_min, "yawns_per_min": yawns_per_min,
                "landmarks": landmarks_norm,
                "eye_left_pts": eye_left_pts, "eye_right_pts": eye_right_pts,
                "mouth_pts": mouth_pts, "head_pose_pts": head_pose_pts,
                "ear_a_left": round(dist(eye_left_pts[1], eye_left_pts[5]), 1),
                "ear_b_left": round(dist(eye_left_pts[2], eye_left_pts[4]), 1),
                "ear_c_left": round(dist(eye_left_pts[0], eye_left_pts[3]), 1),
                "ear_a_right": round(dist(eye_right_pts[1], eye_right_pts[5]), 1),
                "ear_b_right": round(dist(eye_right_pts[2], eye_right_pts[4]), 1),
                "ear_c_right": round(dist(eye_right_pts[0], eye_right_pts[3]), 1),
                "history": {
                    "ear": [[t - self._t_start, v] for t, v in self._hist_ear],
                    "mar": [[t - self._t_start, v] for t, v in self._hist_mar],
                    "perclos": [[t - self._t_start, v] for t, v in self._hist_perclos],
                    "score": [[t - self._t_start, v] for t, v in self._hist_score],
                },
            })

    def get_raw_frame(self):
        with self._lock:
            return self._raw_frame_bytes

    def get_state(self):
        with self._lock:
            return dict(self._state)
