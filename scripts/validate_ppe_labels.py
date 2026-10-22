
from pathlib import Path

ROOT = Path(r"D:\safety-ai\data\ppe_v1_round2")
VALID_CLASSES = set(range(5))

total_boxes = 0
bad_lines = []
empty_labels = 0

for split in ["train", "val", "test"]:
    label_dir = ROOT / split / "labels"

    if not label_dir.is_dir():
        print(f"ERROR: Khong tim thay {label_dir}")
        raise SystemExit(1)

    for file in label_dir.glob("*.txt"):
        lines = file.read_text(encoding="utf-8-sig").splitlines()
        nonempty_lines = [line for line in lines if line.strip()]

        if not nonempty_lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if not parts:
                continue

            try:
                if len(parts) != 5:
                    raise ValueError("Can dung 5 truong")

                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]

                if class_id not in VALID_CLASSES:
                    raise ValueError("Class ID phai tu 0 den 4")

                if not all(0.0 <= x <= 1.0 for x in values):
                    raise ValueError("Toa do nam ngoai [0, 1]")

                x_center, y_center, width, height = values

                if width <= 0 or height <= 0:
                    raise ValueError("Width/height phai lon hon 0")

                x_min = x_center - width / 2
                x_max = x_center + width / 2
                y_min = y_center - height / 2
                y_max = y_center + height / 2

                if (
                    x_min < -1e-6
                    or x_max > 1.0 + 1e-6
                    or y_min < -1e-6
                    or y_max > 1.0 + 1e-6
                ):
                    raise ValueError("Bounding box vuot bien anh")

                total_boxes += 1

            except (ValueError, OverflowError) as error:
                bad_lines.append(
                    f"{split}/{file.name}, dong {line_number}: "
                    f"{error} | {line}"
                )

print("=" * 50)
print("KIEM TRA NHAN YOLO PPE")
print("=" * 50)
print("Tong bounding box hop le:", total_boxes)
print("So dong nhan khong hop le:", len(bad_lines))
print("So file nhan rong:", empty_labels)

if bad_lines:
    print("\nToi da 20 loi dau tien:")
    for error in bad_lines[:20]:
        print(error)
else:
    print("\nKhong phat hien dong nhan sai dinh dang.")

print("=" * 50)
