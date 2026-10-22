
from pathlib import Path

label_dir = Path("D:/safety-ai/data/ppe_v1_round2/val/labels")

names = ["person", "helmet", "no_helmet", "vest", "no_vest"]
box_counts = [0] * 5
image_counts = [0] * 5
empty_count = 0

if not label_dir.exists():
    print("LOI: Khong tim thay thu muc:", label_dir)
else:
    files = list(label_dir.glob("*.txt"))

    for file in files:
        content = file.read_text(encoding="utf-8-sig").strip()

        if not content:
            empty_count += 1
            continue

        found_classes = set()

        for line in content.splitlines():
            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue

            if 0 <= cls < 5:
                box_counts[cls] += 1
                found_classes.add(cls)

        for cls in found_classes:
            image_counts[cls] += 1

    print("Tong file nhan:", len(files))
    print("So nhan rong:", empty_count)
    print("\nThong ke tung lop:")

    for i in range(5):
        print(
            names[i],
            "- so doi tuong:", box_counts[i],
            "- so anh co lop nay:", image_counts[i]
        )

input("\nNhan Enter de dong...")
