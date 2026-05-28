"""
detector.py — Lógica de visión computacional (versión avanzada)
Sistema multimodal de detección de somnolencia:
  • EAR (ojos)           — adaptativo con calibración por usuario
  • PERCLOS              — % de cierre ocular en ventana móvil (estándar industria)
  • MAR (boca)           — detección de bostezos
  • Head pose            — cabeceo / inclinación de la cabeza
  • Frecuencia parpadeo  — parpadeos por minuto
  • Score de fatiga      — fusión ponderada de todas las señales
  • Niveles graduales    — NORMAL / CANSADO / SOMNOLIENTO / ALERTA
  • Alarma sonora        — beep generado por NumPy + sounddevice
  • CLAHE                — mejora de contraste para poca luz
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
import time
import numpy as np
import urllib.request
import os
import threading
from collections import deque

# Audio — sounddevice es opcional; si no está, la alarma sigue siendo visual
try:
    import sounddevice as sd

    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False
    print("⚠  sounddevice no disponible — la alarma será solo visual.")
    print("   Instala con:  pip install sounddevice")


# ═══════════════════════════════════════════════════════════════
#  ÍNDICES DE LANDMARKS  (MediaPipe Face Mesh, 478 puntos)
# ═══════════════════════════════════════════════════════════════
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Boca — para MAR (Mouth Aspect Ratio)
#   Verticales: 13 (labio sup. interior) ↔ 14 (labio inf. interior)
#               81 ↔ 178   y   311 ↔ 402
#   Horizontal: 78 (comisura izq.) ↔ 308 (comisura der.)
MOUTH_OUTER_LEFT = 78
MOUTH_OUTER_RIGHT = 308
MOUTH_TOP_INNER = 13
MOUTH_BOTTOM_INNER = 14
MOUTH_TOP_L = 81
MOUTH_BOTTOM_L = 178
MOUTH_TOP_R = 311
MOUTH_BOTTOM_R = 402

# Head pose — 6 puntos faciales clave para solvePnP
HEAD_POSE_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_l": 263,  # comisura externa ojo izq.
    "right_eye_r": 33,  # comisura externa ojo der.
    "mouth_l": 287,
    "mouth_r": 57,
}

# Modelo 3D genérico de cara (en mm aproximados) — coordenadas estándar
FACE_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye outer corner
        (225.0, 170.0, -135.0),  # right eye outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)


# ═══════════════════════════════════════════════════════════════
#  PARÁMETROS DE DETECCIÓN
# ═══════════════════════════════════════════════════════════════

# EAR
EAR_RATIO_CLOSED = 0.75  # umbral = baseline * 0.75 (calibrado por usuario)
EAR_FALLBACK = 0.22  # umbral fijo si no se ha calibrado aún
CALIB_SECONDS = 3.0  # tiempo de calibración inicial
SMOOTH_WINDOW = 5  # frames para promedio móvil del EAR

# PERCLOS  (Percentage of eye closure)
PERCLOS_WINDOW_SEC = 30  # ventana de 30 s para porcentaje de cierre
PERCLOS_WARN = 0.20  # 20% cerrado = cansado
PERCLOS_ALERT = 0.40  # 40% cerrado = somnoliento

# MAR (bostezo)
MAR_YAWN_THRESHOLD = 0.55  # boca abierta vertical (bostezo)
YAWN_MIN_DURATION = 1.5  # un bostezo dura >= 1.5 s
YAWN_WINDOW_SEC = 60  # cuenta bostezos en último minuto

# Head pose
HEAD_PITCH_DOWN = 18.0  # grados hacia abajo = cabeceo
HEAD_PITCH_WINDOW = 1.5  # debe sostenerse >= 1.5 s

# Parpadeo
BLINK_MIN_DURATION = 0.08  # parpadeo normal: 80-400 ms
BLINK_MAX_DURATION = 0.50
BLINK_WINDOW_SEC = 60  # parpadeos por minuto

# Niveles de alerta
FATIGUE_TIRED = 0.35  # umbral para "cansado"
FATIGUE_DROWSY = 0.60  # umbral para "somnoliento"
FATIGUE_ALERT = 0.80  # umbral para "ALERTA" + alarma
CLOSED_SECONDS = 2.0  # ojos cerrados continuos -> ALERTA directa

# Performance
FRAME_SKIP = 2

# Modelo
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇  Descargando modelo de MediaPipe (≈ 6 MB)… solo la primera vez.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✔  Modelo descargado.")


def _lm_to_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """EAR ≈ 0.30 abierto, < 0.22 cerrado."""
    p = [_lm_to_xy(landmarks[i], w, h) for i in eye_indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def mouth_aspect_ratio(landmarks, w, h):
    """
    MAR similar al EAR pero para la boca.
    Boca cerrada: ~ 0.05 - 0.15
    Hablando:     ~ 0.20 - 0.40
    Bostezando:   > 0.55
    """
    top1 = _lm_to_xy(landmarks[MOUTH_TOP_INNER], w, h)
    bot1 = _lm_to_xy(landmarks[MOUTH_BOTTOM_INNER], w, h)
    top2 = _lm_to_xy(landmarks[MOUTH_TOP_L], w, h)
    bot2 = _lm_to_xy(landmarks[MOUTH_BOTTOM_L], w, h)
    top3 = _lm_to_xy(landmarks[MOUTH_TOP_R], w, h)
    bot3 = _lm_to_xy(landmarks[MOUTH_BOTTOM_R], w, h)

    left = _lm_to_xy(landmarks[MOUTH_OUTER_LEFT], w, h)
    right = _lm_to_xy(landmarks[MOUTH_OUTER_RIGHT], w, h)

    v = (
        np.linalg.norm(top1 - bot1)
        + np.linalg.norm(top2 - bot2)
        + np.linalg.norm(top3 - bot3)
    ) / 3.0
    horiz = np.linalg.norm(left - right)
    return v / horiz if horiz > 0 else 0.0


def estimate_head_pose(landmarks, w, h):
    """
    Retorna (pitch, yaw, roll) en grados.
    pitch positivo = cabeza hacia abajo (cabeceo).
    """
    img_pts = np.array(
        [
            _lm_to_xy(landmarks[HEAD_POSE_IDX["nose_tip"]], w, h),
            _lm_to_xy(landmarks[HEAD_POSE_IDX["chin"]], w, h),
            _lm_to_xy(landmarks[HEAD_POSE_IDX["left_eye_l"]], w, h),
            _lm_to_xy(landmarks[HEAD_POSE_IDX["right_eye_r"]], w, h),
            _lm_to_xy(landmarks[HEAD_POSE_IDX["mouth_l"]], w, h),
            _lm_to_xy(landmarks[HEAD_POSE_IDX["mouth_r"]], w, h),
        ],
        dtype=np.float64,
    )

    focal = w
    cam_matrix = np.array(
        [
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coef = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        FACE_MODEL_3D, img_pts, cam_matrix, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    # Convert rotation matrix to Euler angles (pitch, yaw, roll)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
        yaw = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
        roll = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
        yaw = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
        roll = 0.0

    # Normalizar el pitch: solvePnP devuelve valores cerca de ±180,
    # los pasamos al rango [-90, 90] que es más intuitivo.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return float(pitch), float(yaw), float(roll)


def apply_clahe(frame_bgr):
    """Mejora de contraste para condiciones de poca luz."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════
#  ALARMA SONORA (generada en memoria, sin archivos externos)
# ═══════════════════════════════════════════════════════════════


class AlarmPlayer:
    """Reproduce un beep continuo mientras esté activa."""

    def __init__(self, freq=880, sr=44100):
        self.sr = sr
        self.freq = freq
        self._playing = False
        self._thread = None
        self._stop_evt = threading.Event()

    def _make_beep(self, duration=0.4):
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * self.freq * t)
        # envelope para evitar clicks
        env = np.ones_like(wave)
        ramp = int(0.01 * self.sr)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        return (wave * env).astype(np.float32)

    def _loop(self):
        beep = self._make_beep()
        silence = np.zeros(int(self.sr * 0.15), dtype=np.float32)
        pattern = np.concatenate([beep, silence])
        while not self._stop_evt.is_set():
            try:
                sd.play(pattern, self.sr, blocking=True)
            except Exception:
                time.sleep(0.5)

    def start(self):
        if not _AUDIO_OK or self._playing:
            return
        self._playing = True
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._playing:
            return
        self._stop_evt.set()
        self._playing = False
        try:
            sd.stop()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  DETECTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════


class SleepDetector:

    LEVELS = ["NORMAL", "CANSADO", "SOMNOLIENTO", "ALERTA"]
    LEVEL_COLORS = {
        "NORMAL": (0, 200, 80),
        "CANSADO": (0, 200, 255),
        "SOMNOLIENTO": (0, 120, 255),
        "ALERTA": (0, 0, 255),
    }

    def __init__(self):
        _download_model()

        self.cap = None
        self._frame_bytes = None
        self._lock = threading.Lock()

        # ── Estado público
        self._status = {
            "face_detected": False,
            "level": "NORMAL",
            "fatigue_score": 0.0,
            "ear": 0.0,
            "ear_baseline": 0.0,
            "ear_threshold": EAR_FALLBACK,
            "eyes_closed": False,
            "mar": 0.0,
            "yawning": False,
            "yawns_per_min": 0,
            "blinks_per_min": 0,
            "head_pitch": 0.0,
            "head_nodding": False,
            "perclos": 0.0,
            "closed_time": 0.0,
            "alert": False,
            "calibrating": True,
            "calib_progress": 0.0,
        }

        # ── Estado interno
        self._frame_count = 0
        self._t_start = time.time()

        # EAR
        self._ear_buffer = deque(maxlen=SMOOTH_WINDOW)
        self._ear_baseline_samples = []
        self._ear_baseline = None
        self._ear_threshold = EAR_FALLBACK
        self._eyes_closed_since = None

        # PERCLOS — historial de (timestamp, eyes_closed)
        self._eye_history = deque()

        # MAR / bostezos
        self._yawn_start = None
        self._yawn_events = deque()  # timestamps de bostezos confirmados

        # Head pose
        self._nod_start = None

        # Parpadeos
        self._blink_start = None
        self._blink_events = deque()
        self._was_closed = False

        # Alarma
        self._alarm = AlarmPlayer()

        # MediaPipe
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._detector = FaceLandmarker.create_from_options(options)

    # ──────────────────────────────────────────────────────────
    #  Hilo de captura
    # ──────────────────────────────────────────────────────────
    def start(self):
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Modo poca luz: si la luminancia media es baja, aplicar CLAHE
            mean_lum = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            if mean_lum < 70:
                frame = apply_clahe(frame)

            self._frame_count += 1
            if self._frame_count % FRAME_SKIP == 0:
                self._process(frame)

            annotated = self._draw_hud(frame)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._frame_bytes = buf.tobytes()

    # ──────────────────────────────────────────────────────────
    #  Procesamiento
    # ──────────────────────────────────────────────────────────
    def _process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        now = time.time()

        if not result.face_landmarks:
            self._reset_face_lost()
            return

        lms = result.face_landmarks[0]

        # ── 1. EAR (con suavizado) ────────────────────────────
        ear_l = eye_aspect_ratio(lms, LEFT_EYE, w, h)
        ear_r = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
        ear_raw = (ear_l + ear_r) / 2.0
        self._ear_buffer.append(ear_raw)
        ear = float(np.mean(self._ear_buffer))

        # ── 2. Calibración inicial ─────────────────────────────
        elapsed_total = now - self._t_start
        calibrating = elapsed_total < CALIB_SECONDS
        if calibrating:
            self._ear_baseline_samples.append(ear_raw)
            self._status["calib_progress"] = float(elapsed_total / CALIB_SECONDS)
            self._status["calibrating"] = True
        else:
            if self._ear_baseline is None and self._ear_baseline_samples:
                self._ear_baseline = float(np.median(self._ear_baseline_samples))
                self._ear_threshold = max(0.15, self._ear_baseline * EAR_RATIO_CLOSED)
                self._status["calibrating"] = False
                self._status["calib_progress"] = 1.0
                print(
                    f"✔  Calibración completa. "
                    f"EAR baseline={self._ear_baseline:.3f}, "
                    f"umbral={self._ear_threshold:.3f}"
                )

        eyes_closed = ear < self._ear_threshold

        # ── 3. Cronómetro ojos cerrados continuos ──────────────
        if eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            closed_time = now - self._eyes_closed_since
        else:
            self._eyes_closed_since = None
            closed_time = 0.0

        # ── 4. Parpadeos ───────────────────────────────────────
        if eyes_closed and not self._was_closed:
            self._blink_start = now
        elif not eyes_closed and self._was_closed and self._blink_start:
            dur = now - self._blink_start
            if BLINK_MIN_DURATION <= dur <= BLINK_MAX_DURATION:
                self._blink_events.append(now)
            self._blink_start = None
        self._was_closed = eyes_closed
        self._prune_window(self._blink_events, BLINK_WINDOW_SEC, now)
        blinks_per_min = len(self._blink_events) * (60 / BLINK_WINDOW_SEC)

        # ── 5. PERCLOS ─────────────────────────────────────────
        self._eye_history.append((now, eyes_closed))
        cutoff = now - PERCLOS_WINDOW_SEC
        while self._eye_history and self._eye_history[0][0] < cutoff:
            self._eye_history.popleft()
        if self._eye_history:
            perclos = sum(1 for _, c in self._eye_history if c) / len(self._eye_history)
        else:
            perclos = 0.0

        # ── 6. MAR (bostezos) ──────────────────────────────────
        mar = mouth_aspect_ratio(lms, w, h)
        yawning = mar > MAR_YAWN_THRESHOLD
        if yawning:
            if self._yawn_start is None:
                self._yawn_start = now
            elif now - self._yawn_start >= YAWN_MIN_DURATION and (
                not self._yawn_events or now - self._yawn_events[-1] > 3
            ):
                self._yawn_events.append(now)
        else:
            self._yawn_start = None
        self._prune_window(self._yawn_events, YAWN_WINDOW_SEC, now)
        yawns_per_min = len(self._yawn_events) * (60 / YAWN_WINDOW_SEC)

        # ── 7. Head pose ───────────────────────────────────────
        pitch, _yaw, _roll = estimate_head_pose(lms, w, h)
        # cabeza inclinada hacia abajo = "nodding"
        if pitch > HEAD_PITCH_DOWN:
            if self._nod_start is None:
                self._nod_start = now
            head_nodding = (now - self._nod_start) >= HEAD_PITCH_WINDOW
        else:
            self._nod_start = None
            head_nodding = False

        # ── 8. Score de fatiga combinado (0-1) ─────────────────
        score = self._fatigue_score(
            perclos=perclos,
            closed_time=closed_time,
            yawns_per_min=yawns_per_min,
            head_nodding=head_nodding,
            blinks_per_min=blinks_per_min,
        )

        # ── 9. Nivel y alarma ──────────────────────────────────
        if closed_time >= CLOSED_SECONDS or score >= FATIGUE_ALERT:
            level = "ALERTA"
        elif score >= FATIGUE_DROWSY:
            level = "SOMNOLIENTO"
        elif score >= FATIGUE_TIRED:
            level = "CANSADO"
        else:
            level = "NORMAL"

        alert = level == "ALERTA"
        if alert:
            self._alarm.start()
        else:
            self._alarm.stop()

        # ── Publicar estado ────────────────────────────────────
        self._status = {
            "face_detected": True,
            "level": level,
            "fatigue_score": round(float(score), 3),
            "ear": round(float(ear), 3),
            "ear_baseline": round(float(self._ear_baseline or 0.0), 3),
            "ear_threshold": round(float(self._ear_threshold), 3),
            "eyes_closed": bool(eyes_closed),
            "mar": round(float(mar), 3),
            "yawning": bool(yawning),
            "yawns_per_min": int(round(yawns_per_min)),
            "blinks_per_min": int(round(blinks_per_min)),
            "head_pitch": round(float(pitch), 1),
            "head_nodding": bool(head_nodding),
            "perclos": round(float(perclos), 3),
            "closed_time": round(float(closed_time), 1),
            "alert": bool(alert),
            "calibrating": bool(calibrating),
            "calib_progress": round(self._status.get("calib_progress", 0.0), 2),
        }

    # ──────────────────────────────────────────────────────────
    #  Score de fatiga combinado
    # ──────────────────────────────────────────────────────────
    def _fatigue_score(
        self, perclos, closed_time, yawns_per_min, head_nodding, blinks_per_min
    ):
        """
        Fusión ponderada — el resultado [0, 1] indica nivel de fatiga.
        Los pesos están calibrados para que:
          • PERCLOS sea el dominante (es la métrica más validada en la industria)
          • un bostezo confirmado aporte ~20%
          • cabeceo sostenido aporte ~30%
          • parpadeo lento aporte un poco
        """
        # PERCLOS — el peso fuerte
        s_perclos = np.clip(perclos / PERCLOS_ALERT, 0, 1)  # 1.0 si perclos >= 40%

        # ojos cerrados de corrido — rampa rápida
        s_closed = np.clip(closed_time / CLOSED_SECONDS, 0, 1)  # 1.0 si llegamos a 2 s

        # bostezos — 3+ bostezos por minuto = fatiga clara
        s_yawn = np.clip(yawns_per_min / 3.0, 0, 1)

        # cabeceo — booleano amplificado
        s_nod = 1.0 if head_nodding else 0.0

        # frecuencia de parpadeo: la normal es 15-20/min;
        # > 30/min es señal de fatiga (parpadeo más frecuente y lento)
        s_blink = np.clip((blinks_per_min - 20) / 20.0, 0, 1)

        # Pesos (suman 1.0)
        score = (
            0.35 * s_perclos
            + 0.30 * s_closed
            + 0.15 * s_nod
            + 0.12 * s_yawn
            + 0.08 * s_blink
        )
        return float(np.clip(score, 0, 1))

    def _prune_window(self, dq, window_sec, now):
        cutoff = now - window_sec
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _reset_face_lost(self):
        self._eyes_closed_since = None
        self._yawn_start = None
        self._nod_start = None
        self._blink_start = None
        self._alarm.stop()
        self._status.update(
            {
                "face_detected": False,
                "level": "NORMAL",
                "fatigue_score": 0.0,
                "ear": 0.0,
                "eyes_closed": False,
                "alert": False,
                "closed_time": 0.0,
                "yawning": False,
                "head_nodding": False,
            }
        )

    # ──────────────────────────────────────────────────────────
    #  HUD
    # ──────────────────────────────────────────────────────────
    def _draw_hud(self, frame):
        out = frame.copy()
        s = self._status
        h, w = out.shape[:2]

        # Panel de fondo semi-transparente arriba
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (20, 25, 35), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # ── Calibración ───────────────────────────────────────
        if s.get("calibrating") and s["face_detected"]:
            prog = s.get("calib_progress", 0.0)
            cv2.putText(
                out,
                "CALIBRANDO...",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            cv2.rectangle(out, (10, 35), (210, 50), (60, 60, 60), -1)
            cv2.rectangle(out, (10, 35), (10 + int(200 * prog), 50), (0, 200, 255), -1)
            cv2.putText(
                out,
                "Mira a la cámara con ojos abiertos",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            return out

        if not s["face_detected"]:
            cv2.putText(
                out,
                "Buscando rostro...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )
            return out

        # ── Nivel actual ──────────────────────────────────────
        level = s["level"]
        col = self.LEVEL_COLORS.get(level, (200, 200, 200))
        cv2.putText(out, level, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 2)

        # Barra de score de fatiga
        score = s["fatigue_score"]
        cv2.rectangle(out, (10, 42), (260, 58), (60, 60, 60), -1)
        cv2.rectangle(out, (10, 42), (10 + int(250 * score), 58), col, -1)
        cv2.putText(
            out,
            f"Fatiga: {score:.2f}",
            (14, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # ── Métricas detalladas ───────────────────────────────
        y = 78
        line = lambda txt, color=(220, 220, 220): (
            cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        )

        ear_color = (0, 80, 255) if s["eyes_closed"] else (0, 220, 120)
        line(f"EAR: {s['ear']:.2f}  (umbral {s['ear_threshold']:.2f})", ear_color)
        y += 16
        line(
            f"PERCLOS: {s['perclos']*100:.0f}%",
            (0, 180, 255) if s["perclos"] > PERCLOS_WARN else (220, 220, 220),
        )
        y += 16
        line(
            f"MAR: {s['mar']:.2f}  {'(BOSTEZO)' if s['yawning'] else ''}",
            (0, 180, 255) if s["yawning"] else (220, 220, 220),
        )
        y += 16

        # Panel derecho — métricas secundarias
        right_x = w - 175
        cv2.rectangle(out, (right_x - 5, 5), (w - 5, 100), (20, 25, 35), -1)
        cv2.putText(
            out,
            f"Bostezos/min: {s['yawns_per_min']}",
            (right_x, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            out,
            f"Parpadeos/min: {s['blinks_per_min']}",
            (right_x, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            out,
            f"Pitch: {s['head_pitch']:+.0f}°",
            (right_x, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 80, 255) if s["head_nodding"] else (220, 220, 220),
            1,
        )
        if s["head_nodding"]:
            cv2.putText(
                out,
                "CABECEO",
                (right_x, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 80, 255),
                1,
            )

        # Cronómetro de ojos cerrados
        if s["closed_time"] > 0:
            cv2.putText(
                out,
                f"Cerrados: {s['closed_time']}s",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 120, 255),
                2,
            )

        # ── Overlay de ALERTA ─────────────────────────────────
        if s["alert"]:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
            txt = "¡DESPIERTA!"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.8, 4)
            cv2.putText(
                out,
                txt,
                ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                (0, 0, 255),
                4,
            )

        return out

    # ──────────────────────────────────────────────────────────
    #  API pública
    # ──────────────────────────────────────────────────────────
    def get_frame(self):
        with self._lock:
            return self._frame_bytes

    def get_status(self):
        return dict(self._status)

    def reset_alert(self):
        self._eyes_closed_since = None
        self._yawn_start = None
        self._nod_start = None
        self._eye_history.clear()
        self._yawn_events.clear()
        self._blink_events.clear()
        self._alarm.stop()
        self._status["alert"] = False
        self._status["closed_time"] = 0.0
        self._status["level"] = "NORMAL"
        self._status["fatigue_score"] = 0.0

    def recalibrate(self):
        """Forzar una nueva calibración del EAR baseline."""
        self._t_start = time.time()
        self._ear_baseline = None
        self._ear_baseline_samples = []
        self._ear_threshold = EAR_FALLBACK
        self._status["calibrating"] = True
        self._status["calib_progress"] = 0.0
