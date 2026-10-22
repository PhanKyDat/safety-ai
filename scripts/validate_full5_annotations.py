from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\safety-ai\data\ppe_full5_eval_162")
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels_final"
OUT_JSON = ROOT / "validation_precheck_report.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {0: "person", 1: "helmet", 2: "no_helmet", 3: "vest", 4: "no_vest"}


def main() -> int:
    if not IMAGE_DIR.is_dir():
        print(f"ERROR: Missing images folder: {IMAGE_DIR}")
        return 2
    if not LABEL_DIR.is_dir():
        print(f"ERROR: Missing manually reviewed labels folder: {LABEL_DIR}")
        print("Open annotate_ppe5_gui.py and save reviewed labels to labels_final first.")
        return 2

    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(LABEL_DIR.glob("*.txt"))
    image_stems = {p.stem.lower() for p in images}
    label_stems = {p.stem.lower() for p in labels}
    missing = sorted(image_stems - label_stems)
    orphan = sorted(label_stems - image_stems)
    box_counts = Counter()
    image_counts = Counter()
    errors = []
    empty_label_files = 0
    total_boxes = 0

    if len(images) != 162:
        errors.append(f"Expected 162 images but found {len(images)}")
    if missing:
        errors.append(f"Images missing final label files: {len(missing)}")
    if orphan:
        errors.append(f"Label files without matching image: {len(orphan)}")

    for path in labels:
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            empty_label_files += 1
            continue
        for line_num, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{path.name}:{line_num}: expected 5 YOLO columns")
                continue
            try:
                cfloat = float(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{path.name}:{line_num}: non-numeric field")
                continue
            if not cfloat.is_integer() or int(cfloat) not in CLASS_NAMES:
                errors.append(f"{path.name}:{line_num}: class ID must be integer 0..4")
                continue
            cid = int(cfloat)
            if any(v < 0 or v > 1 for v in vals):
                errors.append(f"{path.name}:{line_num}: coordinate outside [0,1]")
                continue
            xc, yc, w, h = vals
            if w <= 0 or h <= 0:
                errors.append(f"{path.name}:{line_num}: width/height must be positive")
                continue
            if xc - w / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc - h / 2 < -1e-6 or yc + h / 2 > 1 + 1e-6:
                errors.append(f"{path.name}:{line_num}: bbox exceeds image bounds")
                continue
            box_counts[cid] += 1
            image_counts[cid] += 0
            total_boxes += 1

    # Count the number of images containing each class.
    for path in labels:
        present = set()
        text = path.read_text(encoding="utf-8-sig").strip()
        for line in text.splitlines() if text else []:
            p = line.split()
            if len(p) == 5:
                try:
                    cid = int(float(p[0]))
                    if cid in CLASS_NAMES:
                        present.add(cid)
                except ValueError:
                    pass
        for cid in present:
            image_counts[cid] += 1

    report = {
        "images": len(images),
        "final_label_files": len(labels),
        "missing_label_images": missing,
        "orphan_label_files": orphan,
        "empty_label_files": empty_label_files,
        "total_valid_boxes": total_boxes,
        "bbox_count_by_class": {str(c): box_counts[c] for c in CLASS_NAMES},
        "images_containing_class": {str(c): image_counts[c] for c in CLASS_NAMES},
        "format_errors": errors,
        "format_precheck_pass": len(errors) == 0 and len(images) == 162 and len(labels) == 162,
        "semantic_annotation_verified": False,
        "warning": "A format pass does NOT prove every visible PPE object was annotated correctly. Complete visual audit is still required before evaluation or promotion into the main val split."
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 62)
    print("EXPANDED VALIDATION LABEL PRECHECK")
    print("=" * 62)
    print(f"Images: {len(images)} | final labels: {len(labels)} | empty label files: {empty_label_files}")
    print(f"Valid boxes: {total_boxes}")
    for cid, name in CLASS_NAMES.items():
        print(f"{cid} {name:10s}: {box_counts[cid]:5d} boxes in {image_counts[cid]:3d} images")
    print(f"Format errors: {len(errors)}")
    for err in errors[:30]:
        print("ERROR:", err)
    if len(errors) > 30:
        print(f"... plus {len(errors)-30} more format errors")
    print(f"Report: {OUT_JSON}")
    if report["format_precheck_pass"]:
        print("FORMAT PRECHECK: PASS")
        print("SEMANTIC LABEL AUDIT: STILL REQUIRED")
        return 0
    print("FORMAT PRECHECK: FAIL — fix errors before evaluation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
