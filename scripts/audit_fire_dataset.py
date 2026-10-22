
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "fire": ROOT / "data" / "fire",
    "fire_clean": ROOT / "data" / "fire_clean",
}

SPLITS = ["train", "val", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def get_labels(folder):
    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )


def inspect_label(path):
    result = {
        "boxes": Counter(),
        "invalid_lines": 0,
        "out_of_range": 0,
        "empty_file": path.stat().st_size == 0,
        "blank_lines": 0,
        "nonempty_lines": 0,
    }

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except (OSError, UnicodeError):
        result["invalid_lines"] += 1
        return result

    for line in lines:
        parts = line.strip().split()

        if not parts:
            result["blank_lines"] += 1
            continue

        result["nonempty_lines"] += 1

        # YOLO bbox: class x_center y_center width height
        if len(parts) != 5:
            result["invalid_lines"] += 1
            continue

        try:
            cls_float = float(parts[0])
            values = [float(x) for x in parts[1:]]

            if not cls_float.is_integer():
                result["invalid_lines"] += 1
                continue

            cls = int(cls_float)
            x, y, w, h = values

            if not all(map(lambda n: abs(n) < float("inf"), values)):
                result["invalid_lines"] += 1
                continue

            if w <= 0 or h <= 0:
                result["invalid_lines"] += 1
                continue

        except (ValueError, OverflowError):
            result["invalid_lines"] += 1
            continue

        result["boxes"][cls] += 1

        # YOLO normalized coordinates should lie within [0, 1].
        # Also check the box stays inside the image.
        if not all(0 <= v <= 1 for v in values):
            result["out_of_range"] += 1
        elif (
            x - w / 2 < -1e-6
            or y - h / 2 < -1e-6
            or x + w / 2 > 1 + 1e-6
            or y + h / 2 > 1 + 1e-6
        ):
            result["out_of_range"] += 1

    return result


def inspect_split(dataset_name, dataset_dir, split):
    split_dir = dataset_dir / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    if not image_dir.is_dir() and not label_dir.is_dir():
        return None

    print(f"\n--- {dataset_name}/{split} ---")

    if not image_dir.is_dir():
        print("THIEU THU MUC images:", image_dir)
        images = []
    else:
        images = get_images(image_dir)

    if not label_dir.is_dir():
        print("THIEU THU MUC labels:", label_dir)
        labels = []
    else:
        labels = get_labels(label_dir)

    image_map = {p.stem.lower(): p for p in images}
    label_map = {p.stem.lower(): p for p in labels}

    missing_labels = sorted(set(image_map) - set(label_map))
    orphan_labels = sorted(set(label_map) - set(image_map))

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images khong co label file:", len(missing_labels))
    print("Label khong co image:", len(orphan_labels))

    if missing_labels:
        print("Vi du image thieu label:")
        for name in missing_labels[:10]:
            print("  ", image_map[name].name)

    if orphan_labels:
        print("Vi du label orphan:")
        for name in orphan_labels[:10]:
            print("  ", label_map[name].name)

    all_box_counts = Counter()
    images_per_class = Counter()
    empty_labels = 0
    zero_box_labels = 0
    invalid_lines = 0
    out_of_range = 0

    for label in labels:
        stats = inspect_label(label)

        if stats["empty_file"]:
            empty_labels += 1

        if stats["nonempty_lines"] == 0:
            zero_box_labels += 1

        invalid_lines += stats["invalid_lines"]
        out_of_range += stats["out_of_range"]

        for cls, count in stats["boxes"].items():
            all_box_counts[cls] += count
            if count > 0:
                images_per_class[cls] += 1

    print("File label rong (0 byte):", empty_labels)
    print("Label khong co bbox hop le:", zero_box_labels)
    print("Dong label loi:", invalid_lines)
    print("BBox co toa do ngoai khoang chuan:", out_of_range)

    print("\nThong ke class ID:")
    if not all_box_counts:
        print("  Khong tim thay bbox hop le.")
    else:
        for cls in sorted(all_box_counts):
            print(
                f"  class {cls}: {all_box_counts[cls]} bbox, "
                f"xuat hien trong {images_per_class[cls]} file label"
            )

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "boxes": all_box_counts,
    }


def main():
    print("=" * 65)
    print("AUDIT DATASET FIRE / SMOKE")
    print("Chi doc file, khong sua dataset.")
    print("=" * 65)

    for name, dataset_dir in DATASETS.items():
        print(f"\n\n{'=' * 65}")
        print("DATASET:", name)
        print("PATH:", dataset_dir)
        print("=" * 65)

        if not dataset_dir.is_dir():
            print("Khong ton tai thu muc nay.")
            continue

        found_split = False

        for split in SPLITS:
            result = inspect_split(name, dataset_dir, split)
            if result is not None:
                found_split = True

        if not found_split:
            print("Khong tim thay split chua images/labels.")

        print("\nFile metadata o thu muc goc:")
        metadata_exts = {".yaml", ".yml", ".md", ".json", ".csv"}
        metadata = [
            p for p in dataset_dir.iterdir()
            if p.is_file()
            and (
                p.suffix.lower() in metadata_exts
                or p.name.lower().startswith("readme")
            )
        ]

        if metadata:
            for path in metadata:
                print("  ", path.name)
        else:
            print("  Khong tim thay README/YAML/JSON/CSV o thu muc goc.")

    print("\n\nHOAN TAT AUDIT.")
    print("Chua tao YAML, chua chia split, chua train model.")


if __name__ == "__main__":
    main()
