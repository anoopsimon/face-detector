import base64
import binascii
import json
import sqlite3
import time
from pathlib import Path
import shutil

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
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


def detect_frame(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    detector = face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def decode_base64_image(raw_image: str):
    raw_image = (raw_image or "").strip()
    if not raw_image:
        return None, "image is required"

    if raw_image.startswith("data:"):
        _, raw_image = raw_image.split(",", 1)

    try:
        image_bytes = base64.b64decode(raw_image, validate=True)
    except (ValueError, binascii.Error):
        return None, "invalid image data"

    data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return None, "invalid image data"

    return frame, None


def enroll_from_frames(name: str, frames: list[np.ndarray]) -> int:
    ensure_dirs()
    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = face_detector()
    existing = len([p for p in person_dir.glob("*.png") if p.is_file()])
    captured = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = detect_largest_face(detector, gray)
        if rect is None:
            continue
        face = crop_face(gray, rect)
        filename = person_dir / f"{existing + captured:03d}.png"
        cv2.imwrite(str(filename), face)
        captured += 1

    return captured


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
