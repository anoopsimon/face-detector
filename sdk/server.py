import sqlite3
import threading
from typing import Optional

from flask import Flask, abort, jsonify, make_response, request, url_for
from flasgger import Swagger

from .core import (
    DATA_DIR,
    DB_PATH,
    LABELS_PATH,
    MODEL_PATH,
    authenticate_frame,
    authenticate_once,
    clear_known_faces,
    detect_frame,
    decode_base64_image,
    enroll,
    enroll_from_frames,
    ensure_dirs,
    init_db,
    log_auth_result,
    train,
)


def create_app(
    camera_index: int, threshold: float, timeout: float, cors_origin: Optional[str] = None
) -> Flask:
    ensure_dirs()
    init_db()
    app = Flask(__name__)
    Swagger(app, template={"info": {"title": "Face Detector API", "version": "1.0.0"}})
    camera_lock = threading.Lock()

    if cors_origin:
        @app.before_request
        def handle_preflight():
            if request.method == "OPTIONS":
                return make_response("", 204)

        @app.after_request
        def add_cors_headers(response):
            origin = cors_origin.strip()
            if origin == "auto":
                origin = request.headers.get("Origin", "*")
                response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

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

        frame, error = decode_base64_image(payload.get("image"))
        if error:
            return jsonify({"error": error}), 400

        result = authenticate_frame(frame, threshold)
        log_auth_result(result)
        status_code = 200 if result["status"] == "granted" else 403
        return jsonify(result), status_code

    @app.post("/detect-frame")
    def detect_frame_endpoint():
        """Detect faces from a client-provided image (no auth decision).
        ---
        tags:
          - Detection
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
            description: Detected faces
          400:
            description: Invalid image or consent missing
        """
        payload = request.get_json(silent=True) or {}
        if payload.get("consent") is not True:
            return jsonify({"error": "consent is required to access the camera"}), 400

        frame, error = decode_base64_image(payload.get("image"))
        if error:
            return jsonify({"error": error}), 400

        faces = detect_frame(frame)
        return jsonify(
            {
                "count": len(faces),
                "faces": [{"box": [x, y, w, h]} for (x, y, w, h) in faces],
            }
        )

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

    @app.post("/signup-frame")
    def signup_frame():
        """Enroll a new person from client-provided images.
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
                  images:
                    type: array
                    items:
                      type: string
                    description: Base64 strings or data URLs of images.
                  consent:
                    type: boolean
                required:
                  - name
                  - images
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

        images = payload.get("images") or []
        if not isinstance(images, list) or not images:
            return jsonify({"error": "images are required"}), 400

        frames = []
        for idx, raw_image in enumerate(images):
            frame, error = decode_base64_image(raw_image)
            if error:
                return jsonify({"error": f"image {idx + 1}: {error}"}), 400
            frames.append(frame)

        captured = enroll_from_frames(name, frames)
        if captured == 0:
            return jsonify({"error": "no faces detected in provided images"}), 400

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

    return app
