#!/usr/bin/env python3
import argparse
import base64
import binascii
import json
import sqlite3
import time
from pathlib import Path
import shutil
import threading

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "known"
MODELS_DIR = BASE_DIR / "models"
LABELS_PATH = MODELS_DIR / "labels.json"
MODEL_PATH = MODELS_DIR / "lbph.yml"
SNAP_DIR = BASE_DIR / "data" / "snapshots"
DB_PATH = BASE_DIR / "data" / "auth_logs.db"

FACE_SIZE = (200, 200)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                snapshot TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_auth_result(result: dict) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO auth_logs (name, status, confidence, snapshot, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result.get("name", "Unknown"),
                result.get("status", "denied"),
                result.get("confidence"),
                result.get("snapshot"),
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        conn.commit()


def face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")
    return detector


def detect_largest_face(detector: cv2.CascadeClassifier, gray: np.ndarray):
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def crop_face(gray: np.ndarray, rect) -> np.ndarray:
    x, y, w, h = rect
    face = gray[y : y + h, x : x + w]
    return cv2.resize(face, FACE_SIZE)


def enroll(
    name: str, count: int, camera_index: int, delay: float, show_window: bool = True
) -> int:
    ensure_dirs()
    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    captured = 0
    last_capture = 0.0

    start_time = time.time()
    try:
        while captured < count:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rect = detect_largest_face(detector, gray)

            if rect is not None:
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                now = time.time()
                if now - last_capture >= delay:
                    face = crop_face(gray, rect)
                    filename = person_dir / f"{captured:03d}.png"
                    cv2.imwrite(str(filename), face)
                    captured += 1
                    last_capture = now

            if show_window:
                elapsed = int(time.time() - start_time)
                cv2.putText(
                    frame,
                    f"Capturing {name}: {captured}/{count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Camera time: {elapsed}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Enroll", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
    print(f"Saved {captured} face images to {person_dir}")
    return captured


def load_training_data():
    detector = face_detector()
    samples = []
    labels = []
    label_map = {}
    current_label = 0

    for person_dir in sorted(DATA_DIR.glob("*")):
        if not person_dir.is_dir():
            continue
        label_map[current_label] = person_dir.name

        for image_path in sorted(person_dir.glob("*")):
            if not image_path.is_file():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            rect = detect_largest_face(detector, image)
            if rect is None:
                continue
            face = crop_face(image, rect)

            samples.append(face)
            labels.append(current_label)

        current_label += 1

    return samples, np.array(labels, dtype=np.int32), label_map


def train() -> None:
    ensure_dirs()
    samples, labels, label_map = load_training_data()

    if len(samples) == 0:
        raise RuntimeError("No training images found. Run enroll first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, labels)
    recognizer.save(str(MODEL_PATH))

    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Trained on {len(samples)} images. Model saved to {MODEL_PATH}")


def clear_known_faces() -> dict:
    ensure_dirs()
    removed_people = []
    for person_dir in DATA_DIR.glob("*"):
        if person_dir.is_dir():
            shutil.rmtree(person_dir)
            removed_people.append(person_dir.name)
        elif person_dir.is_file():
            person_dir.unlink()

    removed_files = []
    for path in (MODEL_PATH, LABELS_PATH):
        if path.exists():
            path.unlink()
            removed_files.append(path.name)

    removed_snapshots = []
    for snap in SNAP_DIR.glob("*"):
        if snap.is_file():
            snap.unlink()
            removed_snapshots.append(snap.name)

    cleared_logs = 0
    if DB_PATH.exists():
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            cleared_logs = conn.execute("DELETE FROM auth_logs").rowcount
            conn.commit()

    return {
        "status": "cleared",
        "removed_people": removed_people,
        "removed_files": removed_files,
        "removed_snapshots": removed_snapshots,
        "cleared_logs": cleared_logs,
    }


def recognize(camera_index: int, threshold: float) -> None:
    ensure_dirs()
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Model not found. Run train first.")

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    detector = face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_log_time = 0.0
    log_interval = 1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        now = time.time()
        if len(faces) > 0 and now - last_log_time >= log_interval:
            print(f"Face detected: {len(faces)}")
            last_log_time = now

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
            label, confidence = recognizer.predict(face)
            if confidence <= threshold:
                name = label_map.get(label, "Unknown")
                color = (0, 255, 0)
                text = f"{name} ({confidence:.1f})"
            else:
                color = (0, 0, 255)
                text = f"Unknown ({confidence:.1f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Recognize", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def authenticate_frame(frame: np.ndarray, threshold: float):
    ensure_dirs()
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Model not found. Run train first.")

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    detector = face_detector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    best_match = None
    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
        label, confidence = recognizer.predict(face)
        if best_match is None or confidence < best_match["confidence"]:
            best_match = {
                "label": label,
                "confidence": float(confidence),
                "box": (int(x), int(y), int(w), int(h)),
            }

    timestamp = int(time.time() * 1000)
    snapshot_path = SNAP_DIR / f"auth_{timestamp}.jpg"
    if best_match and best_match.get("box"):
        x, y, w, h = best_match["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(str(snapshot_path), frame)

    if best_match and best_match["confidence"] <= threshold:
        name = label_map.get(best_match["label"], "Unknown")
        return {
            "status": "granted",
            "name": name,
            "confidence": best_match["confidence"],
            "snapshot": str(snapshot_path),
        }

    return {
        "status": "denied",
        "name": "Unknown",
        "confidence": best_match["confidence"] if best_match else None,
        "snapshot": str(snapshot_path),
    }


def authenticate_once(
    camera_index: int, threshold: float, timeout: float, show_window: bool = True
):
    ensure_dirs()
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Model not found. Run train first.")

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    detector = face_detector()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    start_time = time.time()
    deadline = start_time + timeout
    best_match = None
    last_frame = None

    try:
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
                label, confidence = recognizer.predict(face)
                if best_match is None or confidence < best_match["confidence"]:
                    best_match = {
                        "label": label,
                        "confidence": float(confidence),
                        "box": (int(x), int(y), int(w), int(h)),
                    }
                if confidence <= threshold:
                    break

            if show_window:
                elapsed = int(time.time() - start_time)
                cv2.putText(
                    frame,
                    f"Camera time: {elapsed}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Authenticate", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if best_match and best_match["confidence"] <= threshold:
                break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    timestamp = int(time.time() * 1000)
    snapshot_path = SNAP_DIR / f"auth_{timestamp}.jpg"
    if last_frame is not None:
        if best_match and best_match.get("box"):
            x, y, w, h = best_match["box"]
            cv2.rectangle(last_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(snapshot_path), last_frame)

    if best_match and best_match["confidence"] <= threshold:
        name = label_map.get(best_match["label"], "Unknown")
        return {
            "status": "granted",
            "name": name,
            "confidence": best_match["confidence"],
            "snapshot": str(snapshot_path),
        }

    return {
        "status": "denied",
        "name": "Unknown",
        "confidence": best_match["confidence"] if best_match else None,
        "snapshot": str(snapshot_path),
    }


def serve(host: str, port: int, camera_index: int, threshold: float, timeout: float) -> None:
    from flask import Flask, abort, jsonify, request, send_from_directory, url_for
    from flasgger import Swagger

    ensure_dirs()
    init_db()
    app = Flask(__name__)
    Swagger(app, template={"info": {"title": "Face Detector API", "version": "1.0.0"}})
    camera_lock = threading.Lock()

    @app.post("/authenticate")
    def authenticate():
        """Authenticate a face once using the configured camera.
        ---
        tags:
          - Auth
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
                properties:
                  consent:
                    type: boolean
                required:
                  - consent
        responses:
          200:
            description: Match granted
          403:
            description: Match denied
          404:
            description: No trained model
          409:
            description: Camera busy
        """
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            return jsonify({"error": "user not found"}), 404
        if not camera_lock.acquire(blocking=False):
            return jsonify({"error": "camera is busy"}), 409
        try:
            payload = request.get_json(silent=True) or {}
            if payload.get("consent") is not True:
                return jsonify({"error": "consent is required to access the camera"}), 400

            result = authenticate_once(camera_index, threshold, timeout, show_window=False)
            log_auth_result(result)
            status_code = 200 if result["status"] == "granted" else 403
            return jsonify(result), status_code
        finally:
            camera_lock.release()

    @app.post("/authenticate-frame")
    def authenticate_frame_endpoint():
        """Authenticate from a client-provided image.
        ---
        tags:
          - Auth
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
                properties:
                  image:
                    type: string
                    description: Base64 string or data URL of a JPEG/PNG image.
                  consent:
                    type: boolean
                required:
                  - image
                  - consent
        responses:
          200:
            description: Match granted
          400:
            description: Invalid image or consent missing
          404:
            description: No trained model
          403:
            description: Match denied
        """
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            return jsonify({"error": "user not found"}), 404

        payload = request.get_json(silent=True) or {}
        if payload.get("consent") is not True:
            return jsonify({"error": "consent is required to access the camera"}), 400

        raw_image = (payload.get("image") or "").strip()
        if not raw_image:
            return jsonify({"error": "image is required"}), 400

        if raw_image.startswith("data:"):
            _, raw_image = raw_image.split(",", 1)

        try:
            image_bytes = base64.b64decode(raw_image, validate=True)
        except (ValueError, binascii.Error):
            return jsonify({"error": "invalid image data"}), 400

        data = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "invalid image data"}), 400

        result = authenticate_frame(frame, threshold)
        log_auth_result(result)
        status_code = 200 if result["status"] == "granted" else 403
        return jsonify(result), status_code

    @app.post("/signup")
    def signup():
        """Enroll a new person and retrain the model.
        ---
        tags:
          - Enrollment
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
                properties:
                  name:
                    type: string
                  count:
                    type: integer
                  delay:
                    type: number
                  camera:
                    type: integer
                  consent:
                    type: boolean
                required:
                  - name
                  - consent
        responses:
          200:
            description: Enrolled
          400:
            description: Invalid request
        """
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        if payload.get("consent") is not True:
            return jsonify({"error": "consent is required to access the camera"}), 400

        count = int(payload.get("count", 20))
        delay = float(payload.get("delay", 0.4))
        cam = int(payload.get("camera", camera_index))

        captured = enroll(name, count, cam, delay, show_window=False)
        train()
        return jsonify(
            {
                "status": "enrolled",
                "name": name,
                "captured": captured,
                "path": str((DATA_DIR / name).resolve()),
            }
        )

    @app.post("/deletefaces")
    def deletefaces():
        """Delete all known faces, models, snapshots, and auth logs.
        ---
        tags:
          - Maintenance
        responses:
          200:
            description: Cleared
        """
        result = clear_known_faces()
        return jsonify(result)

    @app.get("/known/<person>/<filename>")
    def known_image(person: str, filename: str):
        """Serve a stored face image.
        ---
        tags:
          - Assets
        parameters:
          - name: person
            in: path
            required: true
            schema:
              type: string
          - name: filename
            in: path
            required: true
            schema:
              type: string
        responses:
          200:
            description: Image file
          400:
            description: Invalid filename
          404:
            description: Not found
        """
        person_dir = DATA_DIR / person
        if not person_dir.is_dir():
            abort(404)
        if Path(filename).name != filename:
            abort(400)
        return send_from_directory(person_dir, filename)

    @app.get("/ui")
    def ui():
        """Simple HTML UI for known faces and activity stats.
        ---
        tags:
          - UI
        responses:
          200:
            description: HTML page
        """
        init_db()
        stats = {}
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT name, COUNT(*) AS total, MAX(created_at) AS last_seen
                FROM auth_logs
                GROUP BY name
                """
            ).fetchall()
            for name, total, last_seen in rows:
                stats[name] = {"total": total, "last_seen": last_seen}

        people = []
        for person_dir in sorted(DATA_DIR.glob("*")):
            if not person_dir.is_dir():
                continue
            image_path = None
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                candidates = sorted(person_dir.glob(pattern))
                if candidates:
                    image_path = candidates[0]
                    break
            people.append(
                {
                    "name": person_dir.name,
                    "image": image_path.name if image_path else None,
                    "total": stats.get(person_dir.name, {}).get("total", 0),
                    "last_seen": stats.get(person_dir.name, {}).get("last_seen"),
                }
            )

        rows = []
        for person in people:
            if person["image"]:
                img_url = url_for("known_image", person=person["name"], filename=person["image"])
                img_tag = f'<img class="avatar" src="{img_url}" alt="{person["name"]}">'
            else:
                img_tag = '<div class="placeholder">No image</div>'
            last_seen = person["last_seen"] or "—"
            rows.append(
                f"""
                <div class="row">
                  {img_tag}
                  <div class="name">{person["name"]}</div>
                  <div class="meta">
                    <div class="label">Auth count</div>
                    <div class="value">{person["total"]}</div>
                  </div>
                  <div class="meta">
                    <div class="label">Last seen</div>
                    <div class="value">{last_seen}</div>
                  </div>
                </div>
                """
            )

        body = "\n".join(rows) if rows else '<p class="empty">No enrolled people yet.</p>'
        return f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Known Faces</title>
            <style>
              :root {{
                color-scheme: light;
              }}
              body {{
                margin: 0;
                padding: 24px;
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f5f0;
                color: #1f1f1f;
              }}
              h1 {{
                margin: 0 0 8px;
                font-size: 24px;
              }}
              .page-header {{
                margin-bottom: 16px;
              }}
              .subtitle {{
                margin: 0;
                color: #6b6150;
                font-size: 14px;
              }}
              .table {{
                display: flex;
                flex-direction: column;
                gap: 10px;
              }}
              .row {{
                display: flex;
                align-items: center;
                gap: 16px;
                background: #fff8e6;
                border: 1px solid #e6dcc2;
                border-radius: 12px;
                padding: 10px 14px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
              }}
              .name {{
                font-weight: 600;
                font-size: 16px;
                min-width: 140px;
              }}
              .meta {{
                display: flex;
                flex-direction: column;
                min-width: 120px;
                gap: 2px;
              }}
              .label {{
                font-size: 12px;
                color: #8b7e64;
                text-transform: uppercase;
                letter-spacing: 0.04em;
              }}
              .value {{
                font-size: 14px;
                color: #2b2b2b;
              }}
              .avatar {{
                width: 96px;
                height: 120px;
                object-fit: cover;
                border-radius: 8px;
                border: 1px solid #e0d5b7;
                background: #fff;
                display: block;
              }}
              .placeholder {{
                width: 96px;
                height: 120px;
                border-radius: 8px;
                border: 1px dashed #d3c7a5;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #7a6f5a;
                background: #fff3d6;
              }}
              .empty {{
                color: #6b6150;
              }}
            </style>
          </head>
          <body>
            <header class="page-header">
              <h1>Known Faces</h1>
              <p class="subtitle">Live overview of enrolled people and recent authentication activity.</p>
            </header>
            <div class="table">
              {body}
            </div>
          </body>
        </html>
        """

    @app.get("/client")
    def client():
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Client Capture</title>
            <style>
              :root {
                color-scheme: light;
                --ink: #1f1a16;
                --muted: #6b6150;
                --card: #fff8e6;
                --border: #e6dcc2;
                --accent: #2b2a27;
              }
              body {
                margin: 0;
                padding: 24px;
                font-family: "Trebuchet MS", "Gill Sans", "Lucida Grande", sans-serif;
                background: radial-gradient(circle at top, #fff6db 0%, #f5f5f0 55%, #efe8d8 100%);
                color: var(--ink);
              }
              .shell {
                max-width: 720px;
                margin: 0 auto;
              }
              h1 {
                margin: 0 0 6px;
                font-size: 24px;
              }
              p {
                margin: 0 0 16px;
                color: var(--muted);
              }
              .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 16px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.06);
              }
              .consent {
                margin: 16px 0;
                font-size: 14px;
              }
              .actions {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 16px;
              }
              button {
                border: 0;
                padding: 10px 14px;
                border-radius: 10px;
                background: var(--accent);
                color: #fff;
                cursor: pointer;
                font-weight: 600;
              }
              button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
              }
              .frame {
                margin-bottom: 16px;
              }
              .frame-grid {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
              }
              .frame-card {
                flex: 1 1 280px;
              }
              .frame h2 {
                margin: 0 0 10px;
                font-size: 16px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
              }
              video, img {
                width: 100%;
                max-width: 520px;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: #fff;
                display: block;
              }
              .status {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-top: 10px;
                font-size: 13px;
                color: var(--muted);
              }
              .badge {
                background: #1f1a16;
                color: #fff;
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.08em;
              }
              .hidden { display: none; }
              .result { margin-top: 12px; font-size: 13px; color: var(--muted); }
            </style>
          </head>
          <body>
            <div class="shell">
              <h1>Client Capture</h1>
              <p>Capture a frame locally and send it to the API for authentication.</p>
              <div class="card consent">
                Your camera will be accessed and a photo will be taken to validate authentication.
              </div>
              <div class="actions">
                <button id="start">Start Camera</button>
                <button id="snap" disabled>Capture and Authenticate</button>
              </div>
              <div class="frame-grid">
                <div class="frame frame-card card">
                  <h2>Live View</h2>
                  <video id="video" autoplay playsinline class="hidden"></video>
                </div>
                <div class="frame frame-card card hidden" id="captured">
                  <h2>Snapshot</h2>
                  <img id="capturedImg" alt="Captured frame">
                  <div class="status">
                    <span class="badge">OK</span>
                    <span>Completed</span>
                  </div>
                </div>
              </div>
              <canvas id="canvas" width="640" height="480" hidden></canvas>
              <div class="result" id="result"></div>
            </div>
            <script>
              const video = document.getElementById("video");
              const canvas = document.getElementById("canvas");
              const result = document.getElementById("result");
              const start = document.getElementById("start");
              const snap = document.getElementById("snap");
              const captured = document.getElementById("captured");
              const capturedImg = document.getElementById("capturedImg");
              start.onclick = async () => {
                result.textContent = "";
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.classList.remove("hidden");
                snap.disabled = false;
              };
              snap.onclick = async () => {
                if (!video.srcObject) {
                  result.textContent = "Start the camera first.";
                  return;
                }
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
                capturedImg.src = dataUrl;
                captured.classList.remove("hidden");
                const response = await fetch("/authenticate-frame", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ image: dataUrl, consent: true })
                });
                const payload = await response.json();
                result.textContent = JSON.stringify(payload);
              };
            </script>
          </body>
        </html>
        """

    app.run(host=host, port=port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple face recognition demo using OpenCV")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll_parser = sub.add_parser("enroll", help="Capture face images for a person")
    enroll_parser.add_argument("name", help="Person name")
    enroll_parser.add_argument("--count", type=int, default=20, help="Number of images to capture")
    enroll_parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    enroll_parser.add_argument("--delay", type=float, default=0.4, help="Seconds between captures")

    sub.add_parser("train", help="Train the recognizer from stored faces")

    recognize_parser = sub.add_parser("recognize", help="Run live recognition")
    recognize_parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    recognize_parser.add_argument("--threshold", type=float, default=60.0, help="LBPH confidence threshold")

    serve_parser = sub.add_parser("serve", help="Run a local auth REST server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    serve_parser.add_argument("--threshold", type=float, default=60.0, help="LBPH confidence threshold")
    serve_parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for a match")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "enroll":
        enroll(args.name, args.count, args.camera, args.delay)
    elif args.command == "train":
        train()
    elif args.command == "recognize":
        recognize(args.camera, args.threshold)
    elif args.command == "serve":
        serve(args.host, args.port, args.camera, args.threshold, args.timeout)


if __name__ == "__main__":
    main()
