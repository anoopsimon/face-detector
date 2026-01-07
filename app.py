#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "known"
MODELS_DIR = BASE_DIR / "models"
LABELS_PATH = MODELS_DIR / "labels.json"
MODEL_PATH = MODELS_DIR / "lbph.yml"
SNAP_DIR = BASE_DIR / "data" / "snapshots"

FACE_SIZE = (200, 200)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)


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


def enroll(name: str, count: int, camera_index: int, delay: float) -> None:
    ensure_dirs()
    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    captured = 0
    last_capture = 0.0

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

        cv2.putText(
            frame,
            f"Capturing {name}: {captured}/{count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Enroll", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {captured} face images to {person_dir}")


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


def authenticate_once(camera_index: int, threshold: float, timeout: float):
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

    deadline = time.time() + timeout
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

            if best_match and best_match["confidence"] <= threshold:
                break
    finally:
        cap.release()

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
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.post("/authenticate")
    def authenticate():
        result = authenticate_once(camera_index, threshold, timeout)
        status_code = 200 if result["status"] == "granted" else 403
        return jsonify(result), status_code

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
