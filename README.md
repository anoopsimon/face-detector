# Face Recognition (OpenCV)

This is a small, local face-recognition demo using OpenCV. It works by storing a few face images per person, training an LBPH model, and then matching live webcam faces against that model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1) Capture faces for a person:

```bash
python app.py enroll alice --count 25
```

2) Train the model:

```bash
python app.py train
```

3) Run live recognition:

```bash
python app.py recognize
```

Press `q` to exit the camera windows.

## Auth API

Run a local REST server:

```bash
python app.py serve
```

Trigger authentication (returns `200` on match, `403` on unknown):

```bash
curl -X POST http://127.0.0.1:8000/authenticate
```

Response example:

```json
{"status":"granted","name":"anoop","confidence":51.2,"snapshot":"data/snapshots/auth_1736345790123.jpg"}
```

## Project Layout

- `app.py` - CLI for enrolling, training, and recognizing faces.
- `data/known/<person>/` - Stored face images per person.
- `models/` - Trained model and label map.

## Notes

- LBPH is simple and fast but sensitive to lighting and camera quality.
- The `--threshold` value controls how strict matching is; lower is stricter.
