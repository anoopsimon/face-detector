# Face Detector (OpenCV)

## 1. What is it
A small local face-recognition demo that lets you enroll faces, train an LBPH model, and authenticate using a webcam.

### 1.a What problem does it solve
It provides a simple, offline way to learn face enrollment and recognition flows without cloud services.

### 1.b Simple learning project
This is a small learning project to understand basic face datasets, training, and camera-based inference.

## 2. Tool stack and purpose

| Tool | Purpose |
| --- | --- |
| Python | Application runtime |
| OpenCV | Face detection + LBPH recognition |
| Flask | REST API and UI server |
| SQLite | Local auth logs |
| Flasgger | Swagger UI for API docs |

## 3. Available endpoints

- `POST /signup` enroll a person and retrain
- `POST /authenticate` authenticate once from camera
- `POST /authenticate-frame` authenticate from a client-captured image
- `POST /deletefaces` clear all known faces, models, snapshots, and logs
- `GET /ui` simple HTML dashboard
- `GET /known/<person>/<filename>` serve stored face images
- `GET /client` browser capture demo (client-side camera)
- `GET /apidocs` Swagger UI

Swagger UI: `http://127.0.0.1:8000/apidocs`

## 4. How to build and run

Build (install dependencies):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run (server):

```bash
python app.py serve
```

Client capture (browser):

```text
http://127.0.0.1:8000/client
```
