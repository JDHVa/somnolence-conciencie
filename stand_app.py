"""
stand_app.py — 4 vistas para el stand (puerto 5001)
  /xray        — Rayos X (landmarks)
  /conv        — Pipeline de convoluciones
  /monitor     — Monitor del piloto (graficas)
  /sim         — Simulacion de auto en ciudad
  /video_feed  — stream MJPEG
  /state       — JSON completo
"""

from flask import Flask, Response, render_template, jsonify
from stand_detector import StandDetector
import threading
import os

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

app = Flask(__name__)
detector = StandDetector(camera_index=CAMERA_INDEX)


def gen_frames():
    while True:
        frame = detector.get_raw_frame()
        if frame is None:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def home():
    return render_template("stand_home.html")


@app.route("/xray")
def xray():
    return render_template("stand_xray.html")


@app.route("/conv")
def conv():
    return render_template("stand_conv.html")


@app.route("/monitor")
def monitor():
    return render_template("stand_monitor.html")


@app.route("/sim")
def sim():
    return render_template("stand_sim.html")


@app.route("/neural")
def neural():
    return render_template("stand_neural.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def state():
    return jsonify(detector.get_state())


if __name__ == "__main__":
    cam_thread = threading.Thread(target=detector.start, daemon=True)
    cam_thread.start()

    print()
    print("  STAND VISUAL v2")
    print(f"  Camara: indice {CAMERA_INDEX} (cambiar con CAMERA_INDEX=0)")
    print()
    print("  Vistas:")
    print("    http://127.0.0.1:5001/xray     Rayos X")
    print("    http://127.0.0.1:5001/conv     Convoluciones")
    print("    http://127.0.0.1:5001/monitor  Monitor del Piloto")
    print("    http://127.0.0.1:5001/sim      Simulacion Auto")
    print("    http://127.0.0.1:5001/neural   Red Neuronal")
    print()

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
