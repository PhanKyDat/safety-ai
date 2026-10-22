from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------
# CONFIGURATION - edit only these paths if your project folders differ
# ---------------------------------------------------------------------
ROOT = Path(r"D:\safety-ai")
DATASET_ROOT = ROOT / "data" / "ppe_v1_expanded_val_reviewed_construction" / "val"
IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"
MODEL_PATH = ROOT / "runs" / "ppe_v1_round2_train" / "finetune_v1" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "runs" / "expanded_val_eval" / "classwise_conf_no_vest"

IMAGE_SIZE = 512
BATCH_SIZE = 4  # fits the RTX 3050 Laptop GPU with 4 GB VRAM in this project
DEVICE = 0
LOWEST_INFERENCE_CONF = 0.001
NMS_IOU = 0.70
MATCH_IOU = 0.50
FIXED_THRESHOLD_OTHER_CLASSES = 0.25
NO_VEST_THRESHOLDS = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
CLASS_NAMES = {0: "person", 1: "helmet", 2: "no_helmet", 3: "vest", 4: "no_vest"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

Box = Tuple[float, float, float, float]  # normalized xywh
Pred = Tuple[int, float, Box]            # class, confidence, normalized xywh


def die(message: str, code: int = 1) -> None:
    print(f"\nERROR: {message}")
    raise SystemExit(code)


def load_gt(label_path: Path) -> List[Tuple[int, Box]]:
    if not label_path.is_file():
        die(f"Missing label file: {label_path}")
    text = label_path.read_text(encoding="utf-8-sig").strip()
    rows: List[Tuple[int, Box]] = []
    if not text:
        return rows
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            die(f"Invalid YOLO row at {label_path.name}:{line_no}; expected 5 columns")
        try:
            cls_float = float(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            die(f"Non-numeric value at {label_path.name}:{line_no}")
        if not cls_float.is_integer() or int(cls_float) not in CLASS_NAMES:
            die(f"Class ID invalid at {label_path.name}:{line_no}")
        if any(not (0.0 <= v <= 1.0) for v in vals):
            die(f"Coordinate outside [0,1] at {label_path.name}:{line_no}")
        xc, yc, w, h = vals
        if w <= 0 or h <= 0 or xc - w / 2 < -1e-5 or xc + w / 2 > 1.00001 or yc - h / 2 < -1e-5 or yc + h / 2 > 1.00001:
            die(f"Invalid bbox at {label_path.name}:{line_no}")
        rows.append((int(cls_float), (xc, yc, w, h)))
    return rows


def xywh_to_xyxy(box: Box) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def iou_xywh(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_predictions(
    gt_by_image: Dict[str, List[Tuple[int, Box]]],
    pred_by_image: Dict[str, List[Pred]],
    thresholds: Dict[int, float],
    match_iou: float,
) -> Dict[int, Dict[str, float]]:
    counts = {c: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    for stem, rows in gt_by_image.items():
        for c, _ in rows:
            counts[c]["gt"] += 1
        selected = [p for p in pred_by_image.get(stem, []) if p[1] >= thresholds[p[0]]]
        for c in CLASS_NAMES:
            gt_boxes = [box for cls, box in rows if cls == c]
            preds = sorted([p for p in selected if p[0] == c], key=lambda p: p[1], reverse=True)
            used_gt = set()
            counts[c]["pred"] += len(preds)
            for _, conf, pbox in preds:
                best_idx, best_iou = -1, 0.0
                # Match to the highest-IoU GT that has not already been assigned.
                for gi, gt_box in enumerate(gt_boxes):
                    if gi in used_gt:
                        continue
                    overlap = iou_xywh(pbox, gt_box)
                    if overlap > best_iou:
                        best_iou, best_idx = overlap, gi
                if best_idx >= 0 and best_iou >= match_iou:
                    used_gt.add(best_idx)
                    counts[c]["tp"] += 1
                else:
                    counts[c]["fp"] += 1
            counts[c]["fn"] += len(gt_boxes) - len(used_gt)

    metrics: Dict[int, Dict[str, float]] = {}
    for c, v in counts.items():
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        r = v["tp"] / v["gt"] if v["gt"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        metrics[c] = {**{k: float(val) for k, val in v.items()}, "precision": p, "recall": r, "f1": f1}
    return metrics


def main() -> None:
    print("=" * 72)
    print("CLASS-WISE CONFIDENCE TRIAL: no_vest threshold, other classes fixed")
    print("=" * 72)
    print(f"Images:     {IMAGE_DIR}")
    print(f"Labels:     {LABEL_DIR}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"imgsz={IMAGE_SIZE}, inference conf floor={LOWEST_INFERENCE_CONF}, batch={BATCH_SIZE}")

    if not IMAGE_DIR.is_dir():
        die(f"Image folder not found: {IMAGE_DIR}")
    if not LABEL_DIR.is_dir():
        die(f"Label folder not found: {LABEL_DIR}")
    if not MODEL_PATH.is_file():
        die(f"Checkpoint not found: {MODEL_PATH}")

    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        die("No images found")
    image_stems = {p.stem.lower() for p in images}
    label_map = {p.stem.lower(): p for p in LABEL_DIR.glob("*.txt")}
    missing_labels = sorted(image_stems - set(label_map))
    orphan_labels = sorted(set(label_map) - image_stems)
    if missing_labels or orphan_labels:
        print("Missing labels:", missing_labels[:20])
        print("Labels without images:", orphan_labels[:20])
        die("Image/label pairs are not complete; fix before evaluation")

    gt_by_image: Dict[str, List[Tuple[int, Box]]] = {}
    gt_counts = defaultdict(int)
    for im in images:
        rows = load_gt(label_map[im.stem.lower()])
        gt_by_image[im.stem.lower()] = rows
        for cls, _ in rows:
            gt_counts[cls] += 1
    print(f"Images paired: {len(images)}")
    print("Ground-truth boxes:")
    for c, name in CLASS_NAMES.items():
        print(f"  {c} {name:<10}: {gt_counts[c]}")
    if gt_counts[4] == 0:
        die("No ground-truth no_vest boxes found; cannot test this class")

    print("\nLoading model and collecting predictions once at confidence floor...", flush=True)
    model = YOLO(str(MODEL_PATH))
    model_names = model.names
    if isinstance(model_names, list):
        model_names = {i: n for i, n in enumerate(model_names)}
    actual_names = {int(k): str(v).strip().lower() for k, v in model_names.items()}
    for c, expected in CLASS_NAMES.items():
        if actual_names.get(c) != expected:
            die(f"Checkpoint class order mismatch at {c}: expected {expected}, got {actual_names.get(c)}")

    pred_by_image: Dict[str, List[Pred]] = {im.stem.lower(): [] for im in images}
    # Folder inference is batched for speed; results are consumed as a stream to limit memory.
    results = model.predict(
        source=str(IMAGE_DIR),
        stream=True,
        conf=LOWEST_INFERENCE_CONF,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=0,
        device=DEVICE,
        max_det=300,
        verbose=False,
        save=False,
    )
    done = 0
    for result in results:
        stem = Path(result.path).stem.lower()
        if stem not in pred_by_image:
            continue
        if result.boxes is not None and len(result.boxes) > 0:
            xywhn = result.boxes.xywhn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for cls, conf, box in zip(clss, confs, xywhn):
                if cls in CLASS_NAMES:
                    pred_by_image[stem].append((int(cls), float(conf), tuple(float(x) for x in box)))
        done += 1
        if done % 100 == 0 or done == len(images):
            print(f"Processed {done}/{len(images)}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "threshold_sweep.csv"
    per_class_fixed: Dict[int, float] = {c: FIXED_THRESHOLD_OTHER_CLASSES for c in CLASS_NAMES}
    rows_out = []
    metric_by_t = {}
    for t in NO_VEST_THRESHOLDS:
        thresholds = dict(per_class_fixed)
        thresholds[4] = t
        metrics = score_predictions(gt_by_image, pred_by_image, thresholds, MATCH_IOU)
        metric_by_t[t] = metrics
        nv = metrics[4]
        row = {
            "no_vest_conf_threshold": t,
            "other_classes_conf_threshold": FIXED_THRESHOLD_OTHER_CLASSES,
            "imgsz": IMAGE_SIZE,
            "match_iou": MATCH_IOU,
            "no_vest_gt": int(nv["gt"]),
            "no_vest_predictions": int(nv["pred"]),
            "no_vest_TP": int(nv["tp"]),
            "no_vest_FP": int(nv["fp"]),
            "no_vest_FN": int(nv["fn"]),
            "no_vest_precision": nv["precision"],
            "no_vest_recall": nv["recall"],
            "no_vest_f1": nv["f1"],
        }
        for c, name in CLASS_NAMES.items():
            m = metrics[c]
            row[f"{name}_P"] = m["precision"]
            row[f"{name}_R"] = m["recall"]
            row[f"{name}_F1"] = m["f1"]
            row[f"{name}_pred"] = int(m["pred"])
        rows_out.append(row)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    # A separate baseline row makes it clear what standard threshold 0.25 does.
    threshold_for_best = max(NO_VEST_THRESHOLDS, key=lambda t: metric_by_t[t][4]["f1"])
    best_nv = metric_by_t[threshold_for_best][4]
    report = {
        "checkpoint": str(MODEL_PATH),
        "dataset_root": str(DATASET_ROOT),
        "images": len(images),
        "imgsz": IMAGE_SIZE,
        "inference_floor": LOWEST_INFERENCE_CONF,
        "nms_iou": NMS_IOU,
        "match_iou": MATCH_IOU,
        "other_class_thresholds": {CLASS_NAMES[c]: FIXED_THRESHOLD_OTHER_CLASSES for c in CLASS_NAMES if c != 4},
        "no_vest_thresholds_tested": NO_VEST_THRESHOLDS,
        "best_f1_threshold_among_tested": threshold_for_best,
        "no_vest_metrics_at_best_f1": best_nv,
        "note": "This is a custom operating-point P/R/F1 evaluation, NOT mAP. Confidence threshold tuning does not retrain the model. Select a threshold using validation data and an acceptable false-alarm/false-negative tradeoff; confirm on a separate untouched test set before deployment."
    }
    report_path = OUTPUT_DIR / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nNO_VEST THRESHOLD SWEEP (other 4 classes fixed at 0.25)")
    print("conf   preds   TP   FP   FN     P       R      F1")
    for t in NO_VEST_THRESHOLDS:
        m = metric_by_t[t][4]
        print(f"{t:0.4f} {int(m['pred']):6d} {int(m['tp']):4d} {int(m['fp']):4d} {int(m['fn']):4d} {m['precision']:0.3f} {m['recall']:0.3f} {m['f1']:0.3f}")
    print(f"\nHighest F1 among tested thresholds: conf={threshold_for_best} F1={best_nv['f1']:.3f}; P={best_nv['precision']:.3f}; R={best_nv['recall']:.3f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {report_path}")
    print("\nIMPORTANT: Do not treat highest F1 as the automatic deployment threshold. Review FP examples and select the tradeoff required by the safety system.")


if __name__ == "__main__":
    main()
