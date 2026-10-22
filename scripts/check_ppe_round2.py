
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "ppe_v1_round2"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
}

all_stems = {}
total_counts = Counter()
total_images = 0
total_labels = 0
empty_labels = 0
invalid_lines = []

print("=" * 65)
print("KIEM TRA DATASET PPE V1 ROUND 2")
print("=" * 65)

for split in SPLITS:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    labels = list(label_dir.rglob("*.txt"))

    image_stems = {p.stem.lower() for p in images}
    label_stems = {p.stem.lower() for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    counts = Counter()
    split_invalid = []
    split_empty = 0

    for label_path in labels:
        lines = label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        valid_box_count = 0

        for line_no, line in enumerate(lines, 1):
            parts = line.split()

            if not parts:
                continue

            try:
                if len(parts) != 5:
                    raise ValueError("Can phai co 5 gia tri")

                cls = int(parts[0])
                values = [float(x) for x in parts[1:]]

                if cls not in CLASS_NAMES:
                    raise ValueError("Class ID khong hop le")

                if not all(0 <= x <= 1 for x in values):
                    raise ValueError("Toa do nam ngoai [0,1]")

                _, _, w, h = values
                if w <= 0 or h <= 0:
                    raise ValueError("Width/height khong duong")

                counts[cls] += 1
                valid_box_count += 1

            except ValueError as exc:
                split_invalid.append(
                    f"{label_path}:{line_no}: {exc}"
                )

        if valid_box_count == 0:
            split_empty += 1

    print(f"\n[{split.upper()}]")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images thieu label:", len(missing_labels))
    print("Labels thieu image:", len(missing_images))
    print("Label rong/khong co bbox hop le:", split_empty)
    print("Dong nhan loi:", len(split_invalid))

    for cls, name in CLASS_NAMES.items():
        print(f"  {cls} {name}: {counts[cls]} bbox")

    total_images += len(images)
    total_labels += len(labels)
    total_counts.update(counts)
    empty_labels += split_empty
    invalid_lines.extend(split_invalid)

    for image in images:
        all_stems.setdefault(image.stem.lower(), []).append(split)

print("\n" + "=" * 65)
print("TONG KET")
print("=" * 65)
print("Total images:", total_images)
print("Total labels:", total_labels)
print("Label rong/khong co bbox hop le:", empty_labels)
print("Dong nhan loi:", len(invalid_lines))

for cls, name in CLASS_NAMES.items():
    print(f"  {cls} {name}: {total_counts[cls]} bbox")

cross_split_stems = {
    stem: splits
    for stem, splits in all_stems.items()
    if len(set(splits)) > 1
}

print("Trung ten anh giua cac split:", len(cross_split_stems))

if invalid_lines:
    print("\nMot so dong nhan loi:")
    for item in invalid_lines[:20]:
        print(item)

if cross_split_stems:
    print("\nMot so ten anh trung giua split:")
    for stem, splits in list(cross_split_stems.items())[:20]:
        print(stem, splits)

print("\nKET THUC KIEM TRA")
