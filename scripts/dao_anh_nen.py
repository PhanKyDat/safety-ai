r"""Đào ảnh nền khó cho mô hình cháy từ một bộ ảnh KHÔNG có lửa/khói (vd ExDark: ảnh chụp trong tối).

Chạy mô hình cháy trên từng ảnh, giữ lại ảnh nó báo lửa/khói (là báo nhầm, vì bộ ảnh không có lửa),
ghi tệp nhãn RỖNG để thêm vào tập train như ảnh nền: dạy mô hình "đèn xe, đèn đường, đèn trần không
phải lửa". Lỗi báo nhầm đo được trên FIRESENSE (04/11) đều do đèn.

    .venv\Scripts\python.exe scripts\dao_anh_nen.py data\public\exdark\ExDark --out data\gan_nhan\exdark_nen

Kết quả: <out>/images, <out>/labels (rỗng), <out>/xem_truoc (khoanh chỗ báo nhầm), <out>/danh_sach.csv.
Phải xem lại xem_truoc: bộ ảnh "không có lửa" đôi khi vẫn có nến, lửa trại, bếp — ảnh đó phải bỏ ra
(liệt kê tên vào --bo hoặc xoá tay), không được đưa vào như ảnh nền.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Đào ảnh nền khó cho mô hình cháy")
    ap.add_argument("folder", type=Path, help="thư mục ảnh không có lửa/khói (duyệt cả thư mục con)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="models/fire_v1_yolo26s.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng như module dùng khi chạy thật")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--bo", type=Path, default=None, help="tệp liệt kê tên ảnh cần bỏ (có lửa thật)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    drop = set(args.bo.read_text(encoding="utf-8").split()) if args.bo else set()
    for sub in ("images", "labels", "xem_truoc"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.folder.rglob("*") if p.suffix in EXTS)
    rows, kept = [], 0
    for i, p in enumerate(files):
        name = f"{p.parent.name}_{p.stem}".replace(" ", "_")
        if name in drop or p.stem in drop:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device)[0]
        if not len(r.boxes):
            continue
        kept += 1
        shutil.copy2(p, args.out / "images" / f"{name}{p.suffix.lower()}")
        (args.out / "labels" / f"{name}.txt").write_text("")
        vis = img.copy()
        best = 0.0
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"nham {r.names[int(c)]} {s:.2f}", (x1, max(14, y1 - 5)), 0, 0.6, (0, 0, 255), 2)
            best = max(best, s)
        h, w = vis.shape[:2]
        k = min(1.0, 900 / max(h, w))
        cv2.imwrite(str(args.out / "xem_truoc" / f"{name}.jpg"), cv2.resize(vis, (int(w * k), int(h * k))))
        rows.append([f"{name}{p.suffix.lower()}", p.parent.name,
                     ";".join(sorted({r.names[int(c)] for c in r.boxes.cls.tolist()})), f"{best:.3f}"])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)} ảnh, giữ {kept}")
    with open(args.out / "danh_sach.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["anh", "thu_muc_goc", "mo_hinh_bao", "diem_cao_nhat"]] + rows)
    print(f"Duyệt {len(files)} ảnh, mô hình báo nhầm trên {kept} ảnh ({100 * kept / max(1, len(files)):.1f}%).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
