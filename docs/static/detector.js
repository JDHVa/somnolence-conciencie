// SleepDetectorJS — Pure computation drowsiness detector (no DOM dependencies)
// Port of stand_detector.py + detector.py to client-side JavaScript

const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

const MOUTH_OUTER_LEFT = 78;
const MOUTH_OUTER_RIGHT = 308;
const MOUTH_TOP_INNER = 13;
const MOUTH_BOTTOM_INNER = 14;
const MOUTH_TOP_L = 81;
const MOUTH_BOTTOM_L = 178;
const MOUTH_TOP_R = 311;
const MOUTH_BOTTOM_R = 402;

const HEAD_POSE_IDX = {
  nose_tip: 1, chin: 152, forehead: 10,
  left_eye_l: 263, right_eye_r: 33,
  mouth_l: 287, mouth_r: 57,
};

const EAR_RATIO_CLOSED = 0.75;
const EAR_FALLBACK = 0.22;
const CALIB_SECONDS = 3.0;
const SMOOTH_WINDOW = 5;
const PERCLOS_WINDOW_SEC = 30;
const MAR_YAWN_THRESHOLD = 0.55;
const HEAD_PITCH_DOWN = 18.0;
const HEAD_PITCH_WINDOW = 1.5;
const FATIGUE_TIRED = 0.35;
const FATIGUE_DROWSY = 0.60;
const FATIGUE_ALERT = 0.80;
const CLOSED_SECONDS = 2.0;
const HISTORY_SECONDS = 30;

function distance2D(p1, p2) {
  return Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function eyeAspectRatio(landmarks, indices, w, h) {
  const p = indices.map(i => [landmarks[i].x * w, landmarks[i].y * h]);
  const A = distance2D(p[1], p[5]);
  const B = distance2D(p[2], p[4]);
  const C = distance2D(p[0], p[3]);
  return C > 0 ? (A + B) / (2.0 * C) : 0.0;
}

function mouthAspectRatio(landmarks, w, h) {
  const px = (idx) => [landmarks[idx].x * w, landmarks[idx].y * h];
  const v = (
    distance2D(px(MOUTH_TOP_INNER), px(MOUTH_BOTTOM_INNER)) +
    distance2D(px(MOUTH_TOP_L), px(MOUTH_BOTTOM_L)) +
    distance2D(px(MOUTH_TOP_R), px(MOUTH_BOTTOM_R))
  ) / 3.0;
  const horiz = distance2D(px(MOUTH_OUTER_LEFT), px(MOUTH_OUTER_RIGHT));
  return horiz > 0 ? v / horiz : 0.0;
}

function estimateHeadPose(landmarks, w, h) {
  const px = (idx) => [landmarks[idx].x * w, landmarks[idx].y * h];
  const nose = px(HEAD_POSE_IDX.nose_tip);
  const chin = px(HEAD_POSE_IDX.chin);
  const forehead = px(HEAD_POSE_IDX.forehead);
  const leftEye = px(HEAD_POSE_IDX.left_eye_l);
  const rightEye = px(HEAD_POSE_IDX.right_eye_r);

  // Pitch: ratio of nose position between forehead and chin
  const faceH = chin[1] - forehead[1];
  let pitch = 0;
  if (faceH > 0) {
    const ratio = (nose[1] - forehead[1]) / faceH;
    pitch = (ratio - 0.38) * 130;
  }

  // Yaw: horizontal position of nose between eye corners
  const faceW = rightEye[0] - leftEye[0];
  let yaw = 0;
  if (Math.abs(faceW) > 0) {
    const ratio = (nose[0] - leftEye[0]) / faceW;
    yaw = (ratio - 0.5) * -90;
  }

  // Roll: angle of line between eyes
  const roll = Math.atan2(rightEye[1] - leftEye[1], rightEye[0] - leftEye[0]) * (180 / Math.PI);

  return [pitch, yaw, roll];
}

class SleepDetectorJS {
  constructor() {
    this._earBuffer = [];
    this._earBaselineSamples = [];
    this._earBaseline = null;
    this._earThreshold = EAR_FALLBACK;
    this._eyeHistory = [];
    this._eyesClosedSince = null;
    this._wasClosed = false;
    this._blinkStart = null;
    this._blinkEvents = [];
    this._yawnStart = null;
    this._yawnEvents = [];
    this._nodStart = null;
    this._tStart = Date.now() / 1000;

    this._histEar = [];
    this._histMar = [];
    this._histPerclos = [];
    this._histScore = [];

    this._state = {
      face_detected: false, ear: 0, ear_threshold: EAR_FALLBACK,
      mar: 0, perclos: 0, fatigue_score: 0, level: "NORMAL",
      eyes_closed: false, yawning: false, head_nodding: false,
      head_pitch: 0, head_yaw: 0, head_roll: 0,
      calibrating: true, calib_progress: 0, alert: false,
      closed_time: 0, blinks_per_min: 0, yawns_per_min: 0,
      landmarks: [],
      eye_left_pts: [], eye_right_pts: [], mouth_pts: [], head_pose_pts: [],
      ear_a_left: 0, ear_b_left: 0, ear_c_left: 0,
      ear_a_right: 0, ear_b_right: 0, ear_c_right: 0,
      frame_width: 640, frame_height: 480,
      history: { ear: [], mar: [], perclos: [], score: [] },
    };
  }

  update(landmarks, width, height) {
    const now = Date.now() / 1000;
    this._state.frame_width = width;
    this._state.frame_height = height;

    if (!landmarks || landmarks.length === 0) {
      this._state.face_detected = false;
      this._state.landmarks = [];
      this._state.eye_left_pts = [];
      this._state.eye_right_pts = [];
      this._state.mouth_pts = [];
      this._state.head_pose_pts = [];
      this._eyesClosedSince = null;
      return this._state;
    }

    const lms = landmarks;
    const w = width, h = height;

    // EAR
    const earL = eyeAspectRatio(lms, LEFT_EYE, w, h);
    const earR = eyeAspectRatio(lms, RIGHT_EYE, w, h);
    const earRaw = (earL + earR) / 2.0;
    this._earBuffer.push(earRaw);
    if (this._earBuffer.length > SMOOTH_WINDOW) this._earBuffer.shift();
    const ear = this._earBuffer.reduce((a, b) => a + b, 0) / this._earBuffer.length;

    // Calibration
    const elapsedTotal = now - this._tStart;
    const calibrating = elapsedTotal < CALIB_SECONDS;
    if (calibrating) {
      this._earBaselineSamples.push(earRaw);
      this._state.calib_progress = elapsedTotal / CALIB_SECONDS;
    } else if (this._earBaseline === null && this._earBaselineSamples.length > 0) {
      const sorted = [...this._earBaselineSamples].sort((a, b) => a - b);
      this._earBaseline = sorted[Math.floor(sorted.length / 2)];
      this._earThreshold = Math.max(0.15, this._earBaseline * EAR_RATIO_CLOSED);
      this._state.calib_progress = 1.0;
    }

    const eyesClosed = ear < this._earThreshold;

    // Closed time
    let closedTime = 0;
    if (eyesClosed) {
      if (this._eyesClosedSince === null) this._eyesClosedSince = now;
      closedTime = now - this._eyesClosedSince;
    } else {
      this._eyesClosedSince = null;
    }

    // Blinks
    if (eyesClosed && !this._wasClosed) {
      this._blinkStart = now;
    } else if (!eyesClosed && this._wasClosed && this._blinkStart) {
      const dur = now - this._blinkStart;
      if (dur >= 0.08 && dur <= 0.50) this._blinkEvents.push(now);
      this._blinkStart = null;
    }
    this._wasClosed = eyesClosed;
    const cut60 = now - 60;
    while (this._blinkEvents.length && this._blinkEvents[0] < cut60) this._blinkEvents.shift();
    const blinksPerMin = this._blinkEvents.length;

    // PERCLOS
    this._eyeHistory.push([now, eyesClosed]);
    const cutoff = now - PERCLOS_WINDOW_SEC;
    while (this._eyeHistory.length && this._eyeHistory[0][0] < cutoff) this._eyeHistory.shift();
    const perclos = this._eyeHistory.length > 0
      ? this._eyeHistory.filter(e => e[1]).length / this._eyeHistory.length
      : 0;

    // MAR
    const mar = mouthAspectRatio(lms, w, h);
    const yawning = mar > MAR_YAWN_THRESHOLD;
    if (yawning) {
      if (this._yawnStart === null) this._yawnStart = now;
      else if (now - this._yawnStart >= 1.5 &&
               (!this._yawnEvents.length || now - this._yawnEvents[this._yawnEvents.length - 1] > 3)) {
        this._yawnEvents.push(now);
      }
    } else {
      this._yawnStart = null;
    }
    while (this._yawnEvents.length && this._yawnEvents[0] < cut60) this._yawnEvents.shift();
    const yawnsPerMin = this._yawnEvents.length;

    // Head pose
    const [pitch, yaw, roll] = estimateHeadPose(lms, w, h);
    let headNodding = false;
    if (pitch > HEAD_PITCH_DOWN) {
      if (this._nodStart === null) this._nodStart = now;
      headNodding = (now - this._nodStart) >= HEAD_PITCH_WINDOW;
    } else {
      this._nodStart = null;
    }

    // Fatigue score
    const sPerclos = clamp(perclos / 0.40, 0, 1);
    const sClosed = clamp(closedTime / CLOSED_SECONDS, 0, 1);
    const sYawn = yawning ? 1.0 : 0.0;
    const sNod = headNodding ? 1.0 : 0.0;
    const sBlink = clamp((blinksPerMin - 20) / 20.0, 0, 1);
    const score = clamp(
      0.35 * sPerclos + 0.30 * sClosed + 0.15 * sNod + 0.12 * sYawn + 0.08 * sBlink, 0, 1);

    let level;
    if (closedTime >= CLOSED_SECONDS || score >= FATIGUE_ALERT) level = "ALERTA";
    else if (score >= FATIGUE_DROWSY) level = "SOMNOLIENTO";
    else if (score >= FATIGUE_TIRED) level = "CANSADO";
    else level = "NORMAL";

    // Landmark pixel coords for visualization
    const landmarksNorm = lms.map(lm => [lm.x, lm.y]);
    const eyeLeftPts = LEFT_EYE.map(i => [lms[i].x * w, lms[i].y * h]);
    const eyeRightPts = RIGHT_EYE.map(i => [lms[i].x * w, lms[i].y * h]);
    const mouthPts = [
      [lms[MOUTH_OUTER_LEFT].x * w, lms[MOUTH_OUTER_LEFT].y * h],
      [lms[MOUTH_OUTER_RIGHT].x * w, lms[MOUTH_OUTER_RIGHT].y * h],
      [lms[MOUTH_TOP_INNER].x * w, lms[MOUTH_TOP_INNER].y * h],
      [lms[MOUTH_BOTTOM_INNER].x * w, lms[MOUTH_BOTTOM_INNER].y * h],
    ];
    const headPosePts = [
      HEAD_POSE_IDX.nose_tip, HEAD_POSE_IDX.chin,
      HEAD_POSE_IDX.left_eye_l, HEAD_POSE_IDX.right_eye_r,
      HEAD_POSE_IDX.mouth_l, HEAD_POSE_IDX.mouth_r,
    ].map(i => [lms[i].x * w, lms[i].y * h]);

    // History
    const tRel = now - this._tStart;
    this._histEar.push([tRel, ear]);
    this._histMar.push([tRel, mar]);
    this._histPerclos.push([tRel, perclos]);
    this._histScore.push([tRel, score]);
    const histCut = tRel - HISTORY_SECONDS;
    for (const arr of [this._histEar, this._histMar, this._histPerclos, this._histScore]) {
      while (arr.length && arr[0][0] < histCut) arr.shift();
    }

    const dist = (a, b) => distance2D(a, b);

    this._state = {
      face_detected: true,
      ear: Math.round(ear * 1000) / 1000,
      ear_threshold: Math.round(this._earThreshold * 1000) / 1000,
      mar: Math.round(mar * 1000) / 1000,
      perclos: Math.round(perclos * 1000) / 1000,
      fatigue_score: Math.round(score * 1000) / 1000,
      level, eyes_closed: eyesClosed, yawning, head_nodding: headNodding,
      head_pitch: Math.round(pitch * 10) / 10,
      head_yaw: Math.round(yaw * 10) / 10,
      head_roll: Math.round(roll * 10) / 10,
      calibrating, calib_progress: this._state.calib_progress,
      alert: level === "ALERTA",
      closed_time: Math.round(closedTime * 10) / 10,
      blinks_per_min: blinksPerMin, yawns_per_min: yawnsPerMin,
      landmarks: landmarksNorm,
      eye_left_pts: eyeLeftPts, eye_right_pts: eyeRightPts,
      mouth_pts: mouthPts, head_pose_pts: headPosePts,
      ear_a_left: Math.round(dist(eyeLeftPts[1], eyeLeftPts[5]) * 10) / 10,
      ear_b_left: Math.round(dist(eyeLeftPts[2], eyeLeftPts[4]) * 10) / 10,
      ear_c_left: Math.round(dist(eyeLeftPts[0], eyeLeftPts[3]) * 10) / 10,
      ear_a_right: Math.round(dist(eyeRightPts[1], eyeRightPts[5]) * 10) / 10,
      ear_b_right: Math.round(dist(eyeRightPts[2], eyeRightPts[4]) * 10) / 10,
      ear_c_right: Math.round(dist(eyeRightPts[0], eyeRightPts[3]) * 10) / 10,
      frame_width: width, frame_height: height,
      history: {
        ear: [...this._histEar],
        mar: [...this._histMar],
        perclos: [...this._histPerclos],
        score: [...this._histScore],
      },
    };

    return this._state;
  }

  getState() { return this._state; }
}

// Alarm via Web Audio API
class AlarmPlayerJS {
  constructor() {
    this._ctx = null;
    this._interval = null;
    this._playing = false;
  }

  start() {
    if (this._playing) return;
    this._playing = true;
    this._ctx = new (window.AudioContext || window.webkitAudioContext)();
    this._interval = setInterval(() => {
      if (!this._ctx) return;
      const osc = this._ctx.createOscillator();
      const gain = this._ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(900, this._ctx.currentTime);
      gain.gain.setValueAtTime(0.3, this._ctx.currentTime);
      osc.connect(gain);
      gain.connect(this._ctx.destination);
      osc.start();
      osc.stop(this._ctx.currentTime + 0.15);
    }, 300);
  }

  stop() {
    if (!this._playing) return;
    this._playing = false;
    if (this._interval) { clearInterval(this._interval); this._interval = null; }
  }

  get playing() { return this._playing; }
}

// Export to global scope
window.SleepDetectorJS = SleepDetectorJS;
window.AlarmPlayerJS = AlarmPlayerJS;
window.LEFT_EYE = LEFT_EYE;
window.RIGHT_EYE = RIGHT_EYE;
window.HEAD_POSE_IDX = HEAD_POSE_IDX;
