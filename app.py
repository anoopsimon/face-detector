#!/usr/bin/env python3
import argparse

from sdk import enroll, recognize, train
from sdk.server import create_app


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
    serve_parser.add_argument(
        "--cors",
        default=None,
        help="Set Access-Control-Allow-Origin for separate UI hosting",
    )

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
        app = create_app(args.camera, args.threshold, args.timeout, cors_origin=args.cors)
        app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
