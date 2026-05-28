"""
app.py — Servidor Flask principal
Sistema de Detección de Somnolencia en Tiempo Real (versión avanzada)
"""

from flask import Flask, Response, render_template, jsonify
from detector import SleepDetector
import threading

app = Flask(__name__)

detector = SleepDetector()


def gen_frames():
    """Generador de frames para el stream de video (MJPEG)."""
    while True:
        frame = detector.get_frame()
        if frame is None:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream de video en vivo."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    """Estado actual del detector (todas las métricas)."""
    return jsonify(detector.get_status())


@app.route("/reset")
def reset():
    """Reinicia la alerta y los contadores."""
    detector.reset_alert()
    return jsonify({"ok": True})


@app.route("/recalibrate")
def recalibrate():
    """Forzar una nueva calibración del EAR baseline."""
    detector.recalibrate()
    return jsonify(
        {"ok": True, "message": "Recalibrando, mira a la cámara con ojos abiertos"}
    )


if __name__ == "__main__":
    cam_thread = threading.Thread(target=detector.start, daemon=True)
    cam_thread.start()

    print("  Abre tu navegador en  http://127.0.0.1:5000")
    app.run(debug=False, threaded=True)
