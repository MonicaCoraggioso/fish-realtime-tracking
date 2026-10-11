#!/usr/bin/env python3
"""
Evaluate the trained YOLOv8 fish detector on a labelled split.

    python evaluate.py                                # test split of dataset/
    python evaluate.py --split val --weights path/to/best.pt

Reports the standard detection metrics computed by Ultralytics (precision,
recall, mAP@0.5, mAP@0.5:0.95) and, for the real-time use case, the detection
rate and the error on the fish centre in pixels at the 128x128 YOLO resolution.
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def centre_errors(model, split_dir, conf):
    errors, detected = [], 0
    images = sorted((split_dir / "images").glob("*.jpg"))
    for img in images:
        _, xc, yc, _, _ = map(float, (split_dir / "labels" / f"{img.stem}.txt").read_text().split())
        truth = np.array([xc, yc]) * 128
        boxes = model(str(img), imgsz=128, conf=conf, verbose=False)[0].boxes
        if len(boxes) == 0:
            continue
        detected += 1
        best = boxes.xyxy.cpu().numpy()[int(boxes.conf.argmax())]
        centre = np.array([(best[0] + best[2]) / 2, (best[1] + best[3]) / 2])
        errors.append(np.linalg.norm(centre - truth))
    return detected, len(images), np.array(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=HERE / "yolo8m_live_tracking.pt", type=Path)
    parser.add_argument("--data", default=HERE / "dataset" / "data.yaml", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", default=0.25, type=float, help="confidence threshold for the detection rate")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split=args.split, imgsz=128, verbose=False)
    print(f"\n== Ultralytics metrics ({args.split}) ==")
    print(f"precision      {metrics.box.mp:.3f}")
    print(f"recall         {metrics.box.mr:.3f}")
    print(f"mAP@0.5        {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95   {metrics.box.map:.3f}")

    split_dir = args.data.parent / {"val": "valid"}.get(args.split, args.split)
    detected, total, err = centre_errors(model, split_dir, args.conf)
    print(f"\n== Real-time view (conf ≥ {args.conf}) ==")
    print(f"detection rate {detected}/{total} = {detected / total:.1%}")
    if len(err):
        print(f"centre error   median {np.median(err):.2f} px, max {err.max():.2f} px (128×128 input)")


if __name__ == "__main__":
    main()
