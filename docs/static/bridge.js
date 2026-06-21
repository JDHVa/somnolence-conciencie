// bridge.js — ES Module: webcam + MediaPipe + BroadcastChannel
// Each page is autonomous: becomes master if no other tab is detecting,
// otherwise stays as consumer receiving state via BroadcastChannel.

const CHANNEL_NAME = "detector_state";
const NEGOTIATE_MS = 1500;

export async function initDetector({ onState, onVideo, onReady }) {
  const channel = new BroadcastChannel(CHANNEL_NAME);
  let isMaster = false;
  let lastReceived = 0;

  channel.onmessage = (e) => {
    if (e.data?.type === "heartbeat") { lastReceived = Date.now(); return; }
    if (e.data?.type === "state") {
      lastReceived = Date.now();
      if (!isMaster) onState(e.data.state);
    }
  };

  // Wait briefly to see if a master is already broadcasting
  await new Promise(r => setTimeout(r, NEGOTIATE_MS));

  if (Date.now() - lastReceived < NEGOTIATE_MS) {
    // Master exists — stay as consumer, open camera only for display if requested
    if (onVideo) {
      try {
        const video = await openCamera();
        onVideo(video);
      } catch (e) {}
    }
    if (onReady) onReady(false);
    return { isMaster: false, channel };
  }

  // No master found — become master
  isMaster = true;

  const video = await openCamera();
  if (onVideo) onVideo(video);

  const { FaceLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm"
  );
  const resolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  const landmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minFacePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });

  const detector = new SleepDetectorJS();

  // Heartbeat so other tabs know a master is active
  setInterval(() => channel.postMessage({ type: "heartbeat" }), 500);

  function processFrame() {
    if (video.readyState >= 2) {
      const result = landmarker.detectForVideo(video, performance.now());
      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      const landmarks = result.faceLandmarks?.length > 0
        ? result.faceLandmarks[0] : null;
      const state = detector.update(landmarks, w, h);
      onState(state);
      channel.postMessage({ type: "state", state });
    }
    requestAnimationFrame(processFrame);
  }
  processFrame();

  if (onReady) onReady(true);
  return { isMaster: true, channel };
}

async function openCamera() {
  const video = document.createElement("video");
  video.setAttribute("playsinline", "");
  video.setAttribute("autoplay", "");
  video.muted = true;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
  });
  video.srcObject = stream;
  await video.play();
  return video;
}
