
from pathlib import Path

root = Path(r"D:\safety-ai\data\ppe_v1_round2")

for split in ["train", "val", "test"]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"

    images = [
        p for p in image_dir.iterdir() if p.is_file()
    ] if image_dir.exists() else []

    labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []

    empty_labels = 0
    for label in labels:
        try:
            if not label.read_text(encoding="utf-8").strip():
                empty_labels += 1
        except Exception as e:
            print(f"Khong doc duoc {label}: {e}")

    print(f"\n[{split.upper()}]")
    print(f"So anh: {len(images)}")
    print(f"So file nhan: {len(labels)}")
    print(f"So nhan rong: {empty_labels}")

input("\nNhan Enter de ket thuc...")
