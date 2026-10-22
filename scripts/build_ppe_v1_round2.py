
from pathlib import Path
import csv
import os
import re
import shutil
import sys


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data" / "ppe_v1_final"
OUTPUT = ROOT / "data" / "ppe_v1_round2"

REVIEW_DIR = ROOT / "reports" / "ppe_v1_nohelmet_review"
ROUND1_CSV = REVIEW_DIR / "review_status.csv"
ROUND2_CSV = REVIEW_DIR / "review_status_round2.csv"

FINAL_DECISIONS_CSV = REVIEW_DIR / "round2_final_decisions.csv"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
}

# So lieu da xac nhan sau hai vong review.
EXPECTED_REMOVE_IMAGES = 327
EXPECTED_REMOVE_BOXES = 590
EXPECTED_KEEP_IMAGES = 64
EXPECTED_KEEP_BOXES = 137
EXPECTED_SOURCE_NO_HELMET_BOXES = 727


# ============================================================
# READ AND VALIDATE CSV
# ============================================================

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay file:\n{path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_relative_path(value):
    path = Path(value.replace("\\", "/"))

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Duong dan khong an toan: {value}")

    return path


def read_label_lines(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return f.readlines()


def class_id(line):
    parts = line.strip().split()

    if not parts:
        return None

    try:
        return int(parts[0])
    except ValueError:
        return None


def count_class_in_lines(lines):
    counts = {}

    for line in lines:
        cls = class_id(line)

        if cls is not None:
            counts[cls] = counts.get(cls, 0) + 1

    return counts


def load_final_decisions():
    rows1 = read_csv(ROUND1_CSV)
    rows2 = read_csv(ROUND2_CSV)

    round1 = {}

    for row in rows1:
        key = (row.get("key") or "").strip()
        status = (row.get("status") or "").strip().upper()

        if not key:
            raise ValueError("CSV vong 1 co dong thieu key.")

        if key in round1:
            raise ValueError(f"Key trung trong CSV vong 1: {key}")

        if status not in {"KEEP", "REMOVE", "UNSURE"}:
            raise ValueError(
                f"Trang thai vong 1 khong hop le: {key}: {status}"
            )

        try:
            boxes = int(row.get("no_helmet_boxes") or 0)
        except ValueError:
            raise ValueError(f"So bbox khong hop le: {key}")

        image_rel = safe_relative_path(
            (row.get("image") or "").strip()
        )
        label_rel = safe_relative_path(
            (row.get("label") or "").strip()
        )

        split = (row.get("split") or "").strip().lower()

        if split not in SPLITS:
            raise ValueError(f"Split khong hop le: {key}: {split}")

        if image_rel.parts[0] != split or label_rel.parts[0] != split:
            raise ValueError(
                f"Split va duong dan khong khop trong CSV: {key}"
            )

        round1[key] = {
            "key": key,
            "split": split,
            "image": image_rel.as_posix(),
            "label": label_rel.as_posix(),
            "boxes": boxes,
            "round1_status": status,
        }

    round2 = {}

    for row in rows2:
        key = (row.get("key") or "").strip()
        status = (
            (row.get("round2_status") or "").strip().upper()
        )

        if not key:
            raise ValueError("CSV vong 2 co dong thieu key.")

        if key in round2:
            raise ValueError(f"Key trung trong CSV vong 2: {key}")

        if key not in round1:
            raise ValueError(
                f"Anh vong 2 khong co trong vong 1: {key}"
            )

        if round1[key]["round1_status"] not in {"KEEP", "UNSURE"}:
            raise ValueError(
                f"Anh REMOVE o vong 1 khong nen co trong vong 2: {key}"
            )

        if status not in {"KEEP", "REMOVE", "UNSURE"}:
            raise ValueError(
                f"Anh chua duoc review hoac trang thai sai: {key}: {status}"
            )

        round2[key] = status

    expected_round2_keys = {
        key for key, item in round1.items()
        if item["round1_status"] in {"KEEP", "UNSURE"}
    }

    if set(round2) != expected_round2_keys:
        missing = expected_round2_keys - set(round2)
        extra = set(round2) - expected_round2_keys

        raise ValueError(
            "CSV vong 2 khong khop danh sach anh can review.\n"
            f"Thieu: {len(missing)}\n"
            f"Thua: {len(extra)}"
        )

    final_rows = []

    for key, item in round1.items():
        if item["round1_status"] == "REMOVE":
            final_status = "REMOVE"
        else:
            final_status = round2[key]

        final = dict(item)
        final["round2_status"] = round2.get(key, "")
        final["final_status"] = final_status
        final_rows.append(final)

    unresolved = [
        row for row in final_rows
        if row["final_status"] in {"UNSURE", "PENDING"}
    ]

    if unresolved:
        raise ValueError(
            f"Van con {len(unresolved)} anh UNSURE/PENDING. "
            "Khong tao dataset."
        )

    remove_rows = [
        row for row in final_rows
        if row["final_status"] == "REMOVE"
    ]

    keep_rows = [
        row for row in final_rows
        if row["final_status"] == "KEEP"
    ]

    remove_boxes = sum(row["boxes"] for row in remove_rows)
    keep_boxes = sum(row["boxes"] for row in keep_rows)

    # Dung neu CSV da thay doi so voi ket qua da xac nhan.
    if len(remove_rows) != EXPECTED_REMOVE_IMAGES:
        raise ValueError(
            f"So anh REMOVE khong dung: {len(remove_rows)}; "
            f"du kien {EXPECTED_REMOVE_IMAGES}"
        )

    if remove_boxes != EXPECTED_REMOVE_BOXES:
        raise ValueError(
            f"So bbox REMOVE khong dung: {remove_boxes}; "
            f"du kien {EXPECTED_REMOVE_BOXES}"
        )

    if len(keep_rows) != EXPECTED_KEEP_IMAGES:
        raise ValueError(
            f"So anh KEEP khong dung: {len(keep_rows)}; "
            f"du kien {EXPECTED_KEEP_IMAGES}"
        )

    if keep_boxes != EXPECTED_KEEP_BOXES:
        raise ValueError(
            f"So bbox KEEP khong dung: {keep_boxes}; "
            f"du kien {EXPECTED_KEEP_BOXES}"
        )

    if sum(row["boxes"] for row in final_rows) != (
        EXPECTED_SOURCE_NO_HELMET_BOXES
    ):
        raise ValueError("Tong bbox vong 1 + vong 2 khong khop 727.")

    return final_rows


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_stats(dataset):
    counts = {}
    label_files = 0
    image_counts = {}

    for split in SPLITS:
        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset thieu images/labels:\n{images_dir}\n{labels_dir}"
            )

        image_counts[split] = sum(
            1 for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        for path in labels_dir.glob("*.txt"):
            label_files += 1
            file_counts = count_class_in_lines(read_label_lines(path))

            for cls, n in file_counts.items():
                counts[cls] = counts.get(cls, 0) + n

    return counts, label_files, image_counts


# ============================================================
# CREATE OUTPUT
# Hard-link image/label files to save disk space.
# Labels changed in OUTPUT will be detached from source first.
# ============================================================

def create_linked_copy():
    for current, dirs, files in os.walk(SOURCE):
        source_dir = Path(current)
        relative_dir = source_dir.relative_to(SOURCE)
        output_dir = OUTPUT / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            src = source_dir / filename
            rel = src.relative_to(SOURCE)
            dst = OUTPUT / rel

            # Do not copy YOLO cache files; output will create its own.
            if src.name.lower().endswith(".cache"):
                continue

            if dst.exists():
                raise FileExistsError(f"File output da ton tai: {dst}")

            # Images and YOLO labels use hard links.
            # All other files are copied as independent files.
            is_image = (
                "images" in rel.parts
                and src.suffix.lower() in IMAGE_EXTS
            )
            is_label = (
                "labels" in rel.parts
                and src.suffix.lower() == ".txt"
            )

            if is_image or is_label:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)


def write_independent_label(label_path, lines):
    # Replace the output hard link with an independent label file.
    # os.replace replaces only the output directory entry.
    temp_path = label_path.with_name(label_path.name + ".tmp")

    with temp_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(lines)

    temp_path.replace(label_path)


def update_output_yaml():
    yaml_path = OUTPUT / "data.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Thieu data.yaml: {yaml_path}")

    text = yaml_path.read_text(encoding="utf-8-sig")

    replacement = f"path: {OUTPUT.as_posix()}"
    pattern = r"(?m)^\s*path\s*:\s*.*$"

    text, number = re.subn(pattern, replacement, text, count=1)

    if number == 0:
        text = replacement + "\n" + text

    # data.yaml is an independent copy in OUTPUT, not a hard link.
    yaml_path.write_text(text, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Source :", SOURCE)
    print("Output :", OUTPUT)
    print("Review :", REVIEW_DIR)

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Khong tim thay dataset goc: {SOURCE}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output da ton tai:\n{OUTPUT}\n"
            "Khong tu dong xoa hoac ghi de."
        )

    final_rows = load_final_decisions()

    remove_rows = [
        row for row in final_rows
        if row["final_status"] == "REMOVE"
    ]

    # Map label relative path -> expected number of class-2 boxes to remove.
    remove_by_label = {}

    for row in remove_rows:
        label_rel = row["label"]

        if label_rel in remove_by_label:
            raise ValueError(f"Label bi lap trong danh sach REMOVE: {label_rel}")

        source_label = SOURCE / label_rel

        if not source_label.is_file():
            raise FileNotFoundError(
                f"Khong tim thay label goc: {source_label}"
            )

        lines = read_label_lines(source_label)
        class_counts = count_class_in_lines(lines)
        actual_no_helmet = class_counts.get(2, 0)

        if actual_no_helmet != row["boxes"]:
            raise ValueError(
                f"So bbox class 2 khong khop CSV:\n{source_label}\n"
                f"CSV: {row['boxes']}; label goc: {actual_no_helmet}"
            )

        remove_by_label[label_rel] = row["boxes"]

    before_counts, before_label_files, before_image_counts = (
        dataset_stats(SOURCE)
    )

    if before_counts.get(2, 0) != EXPECTED_SOURCE_NO_HELMET_BOXES:
        raise ValueError(
            "Dataset goc co so bbox no_helmet khong khop.\n"
            f"Thuc te: {before_counts.get(2, 0)}; "
            f"du kien: {EXPECTED_SOURCE_NO_HELMET_BOXES}"
        )

    print("\nQuyet dinh da xac nhan:")
    print("  Anh REMOVE:", len(remove_rows))
    print(
        "  Bbox no_helmet can loai:",
        sum(remove_by_label.values())
    )
    print("  Anh KEEP:", EXPECTED_KEEP_IMAGES)
    print("  Bbox no_helmet se giu:", EXPECTED_KEEP_BOXES)

    print("\nDang tao dataset bang hard link de tiet kiem dung luong...")
    create_linked_copy()

    removed_boxes = 0
    changed_label_files = 0

    try:
        for label_rel, expected_count in remove_by_label.items():
            output_label = OUTPUT / label_rel

            lines = read_label_lines(output_label)
            original_counts = count_class_in_lines(lines)

            actual_count = original_counts.get(2, 0)

            if actual_count != expected_count:
                raise ValueError(
                    f"So bbox class 2 trong output khong khop: "
                    f"{output_label}"
                )

            kept_lines = [
                line for line in lines
                if class_id(line) != 2
            ]

            number_removed = len(lines) - len(kept_lines)

            # Verify based on class count, because blank lines do not count.
            retained_counts = count_class_in_lines(kept_lines)

            if number_removed != expected_count:
                actual_removed = actual_count
            else:
                actual_removed = number_removed

            if actual_removed != expected_count:
                raise ValueError(
                    f"So bbox loai khong khop: {output_label}"
                )

            if retained_counts.get(0, 0) != original_counts.get(0, 0):
                raise ValueError(f"Class person bi thay doi: {output_label}")

            if retained_counts.get(1, 0) != original_counts.get(1, 0):
                raise ValueError(f"Class helmet bi thay doi: {output_label}")

            if retained_counts.get(3, 0) != original_counts.get(3, 0):
                raise ValueError(f"Class vest bi thay doi: {output_label}")

            if retained_counts.get(4, 0) != original_counts.get(4, 0):
                raise ValueError(f"Class no_vest bi thay doi: {output_label}")

            write_independent_label(output_label, kept_lines)

            removed_boxes += actual_removed
            changed_label_files += 1

        update_output_yaml()

        after_counts, after_label_files, after_image_counts = (
            dataset_stats(OUTPUT)
        )

        # The split membership and image counts must be unchanged.
        if before_image_counts != after_image_counts:
            raise ValueError(
                f"So anh theo split thay doi: "
                f"{before_image_counts} -> {after_image_counts}"
            )

        if before_label_files != after_label_files:
            raise ValueError(
                f"So label files thay doi: "
                f"{before_label_files} -> {after_label_files}"
            )

        # Only class 2 is supposed to change.
        for cls in [0, 1, 3, 4]:
            if before_counts.get(cls, 0) != after_counts.get(cls, 0):
                raise ValueError(
                    f"So bbox class {cls} thay doi bat thuong: "
                    f"{before_counts.get(cls, 0)} -> "
                    f"{after_counts.get(cls, 0)}"
                )

        expected_after_class2 = (
            before_counts.get(2, 0) - EXPECTED_REMOVE_BOXES
        )

        if after_counts.get(2, 0) != expected_after_class2:
            raise ValueError(
                f"So bbox no_helmet sau loc khong dung: "
                f"{after_counts.get(2, 0)}; "
                f"du kien {expected_after_class2}"
            )

        if removed_boxes != EXPECTED_REMOVE_BOXES:
            raise ValueError(
                f"So bbox da loai khong dung: {removed_boxes}"
            )

        # Save an audit CSV of combined decisions.
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        with FINAL_DECISIONS_CSV.open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "key",
                    "split",
                    "image",
                    "label",
                    "no_helmet_boxes",
                    "round1_status",
                    "round2_status",
                    "final_status",
                ],
            )
            writer.writeheader()

            for row in final_rows:
                writer.writerow({
                    "key": row["key"],
                    "split": row["split"],
                    "image": row["image"],
                    "label": row["label"],
                    "no_helmet_boxes": row["boxes"],
                    "round1_status": row["round1_status"],
                    "round2_status": row["round2_status"],
                    "final_status": row["final_status"],
                })

    except Exception:
        print(
            "\nLOI khi tao/kiem tra dataset.\n"
            "Dataset goc khong bi sua.\n"
            f"Output dang nam tai: {OUTPUT}\n"
            "Khong tu dong xoa output; hay gui lai thong bao loi."
        )
        raise

    print("\n" + "=" * 65)
    print("HOAN TAT PPE V1 ROUND 2")
    print("=" * 65)
    print("Output:", OUTPUT)
    print("Label files:", after_label_files)
    print("Image counts by split:", after_image_counts)
    print("Label files co bbox class 2 bi loai:", changed_label_files)
    print("BBox class 2 da loai:", removed_boxes)

    print("\nSo bbox truoc -> sau:")
    for cls in [0, 1, 2, 3, 4]:
        old = before_counts.get(cls, 0)
        new = after_counts.get(cls, 0)
        name = CLASS_NAMES[cls]
        print(f"  {cls} {name}: {old} -> {new}")

    print("\nData YAML:", OUTPUT / "data.yaml")
    print("CSV tong hop:", FINAL_DECISIONS_CSV)
    print("Dataset goc ppe_v1_final khong bi sua.")
    print("Cac file nhan khong thay doi van la hard link.")
    print(
        "Khong sua truc tiep anh trong dataset output; "
        "anh duoc hard link voi dataset goc."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nDUNG LAI DO LOI: {exc}")
        sys.exit(1)
