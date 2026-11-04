r"""Viết đáp án nhan.csv cho bộ FIRESENSE để scripts/evaluate_videos.py chấm được.

FIRESENSE (https://zenodo.org/records/836749, CC BY 4.0) chỉ chia video thành pos/ (có lửa
hoặc khói) và neg/ (cảnh dễ nhầm, không có), không ghi lửa xuất hiện từ giây nào. Vì vậy
video pos/ được ghi là có `nhan` từ đầu tới cuối video, video neg/ để trống nhãn.

Giải nén fire_videos.1406.zip và smoke_videos.1407.zip vào hai thư mục riêng rồi:

    .venv\Scripts\python.exe scripts/prep_firesense.py data/public/firesense_fire --nhan fire
    .venv\Scripts\python.exe scripts/prep_firesense.py data/public/firesense_smoke --nhan smoke
    .venv\Scripts\python.exe scripts/evaluate_videos.py data/public/firesense_smoke/pos/*.avi ^
        data/public/firesense_smoke/neg/*.avi --model models/fire_v1_yolo26s.pt

Hệ quả cần nhớ khi đọc số: F1 theo sự cố là chấm theo video (video pos có báo = đúng,
video neg có báo = nhầm), còn AUROC/AUPRC theo đoạn hơi bi quan vì mấy giây đầu video
pos có thể chưa thấy lửa/khói mà vẫn tính là "có".
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def duration(path: Path) -> float:
    """Độ dài video tính bằng giây, đọc hết khung vì vài tệp .avi báo sai số khung."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n / fps


def main() -> int:
    ap = argparse.ArgumentParser(description="Viết nhan.csv cho FIRESENSE")
    ap.add_argument("folder", type=Path, help="thư mục có pos/ và neg/")
    ap.add_argument("--nhan", required=True, choices=("fire", "smoke"))
    args = ap.parse_args()
    for sub, positive in (("pos", True), ("neg", False)):
        folder = args.folder / sub
        videos = sorted(folder.glob("*.avi"))
        rows = [["tep", "nhan", "bat_dau_giay", "ket_thuc_giay", "ghi_chu"]]
        for v in videos:
            if positive:
                rows.append([v.name, args.nhan, "0", f"{duration(v):.1f}", "FIRESENSE pos, ca video"])
            else:
                rows.append([v.name, "", "", "", "FIRESENSE neg, canh de nham"])
        with open(folder / "nhan.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"{folder / 'nhan.csv'}: {len(videos)} video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
