r"""Chuyển bộ SHWD (Safety Helmet Wearing Dataset) sang nhãn YOLO 5 lớp của mô hình đồ bảo hộ.

SHWD: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (repo giấy phép MIT; ảnh gom
từ Google/Baidu và bộ SCUT-HEAD — ghi nguồn khi dùng). 7581 ảnh, nhãn Pascal VOC chỉ khoanh ĐẦU:
"hat" = đầu đội mũ bảo hộ, "person" = đầu không đội mũ bảo hộ (kể cả đầu đội mũ thường).

Lớp đích (đúng thứ tự mô hình nhóm): 0 person, 1 helmet, 2 no_helmet, 3 vest, 4 no_vest
  hat    -> 1 helmet     (nhãn gốc, người gán tay)
  person -> 2 no_helmet  (nhãn gốc, người gán tay)
  0 person, 3 vest       -> TỰ GÁN bằng mô hình nhóm (--gan-them): SHWD không khoanh người và áo,
                            để trống thì mô hình mới học sai rằng người trong ảnh "không phải người".
  4 no_vest              -> không gán (không có căn cứ đủ chắc).

Bước 1 (máy Đạt, có GPU):
    .venv\Scripts\python.exe scripts\chuyen_shwd.py data\public\shwd\VOC2028 --out data\gan_nhan\shwd_yolo --gan-them
Bước 2 (máy Trường, sau khi tự tải VOC2028.zip từ link chính thức và giải nén):
    python scripts\chuyen_shwd.py D:\VOCdevkit\VOC2028 --out <thư mục nhãn đã nhận> --chi-xep-anh
  -> chép ảnh vào images/train, images/val khớp với labels/train, labels/val để Ultralytics đọc.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest"]
VOC_TO_ID = {"hat": 1, "person": 2}
COLORS = {0: (255, 160, 0), 1: (0, 200, 0), 2: (0, 0, 255), 3: (0, 220, 220)}


def read_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    boxes, other = [], []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        if name in VOC_TO_ID and x2 > x1 and y2 > y1:
            boxes.append((VOC_TO_ID[name], x1, y1, x2, y2))
        else:
            other.append(name)
    return boxes, other


def yolo(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def split_lists(voc: Path) -> dict[str, list[str]]:
    sets = voc / "ImageSets" / "Main"
    out = {}
    for split, fname in (("train", "trainval.txt"), ("val", "test.txt")):
        f = sets / fname
        out[split] = [s.strip() for s in f.read_text().split() if s.strip()] if f.is_file() else []
    if not any(out.values()):     # không có danh sách chia thì tự chia 90/10 theo tên
        names = sorted(p.stem for p in (voc / "Annotations").glob("*.xml"))
        out = {"train": [n for i, n in enumerate(names) if i % 10], "val": names[::10]}
    return out


def find_image(voc: Path, name: str) -> Path | None:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = voc / "JPEGImages" / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def xep_anh(voc: Path, out: Path) -> int:
    n = 0
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        for lab in (out / "labels" / split).glob("*.txt"):
            src = find_image(voc, lab.stem)
            if src is not None:
                shutil.copy2(src, out / "images" / split / f"{lab.stem}{src.suffix.lower()}")
                n += 1
    print(f"Đã chép {n} ảnh vào {out / 'images'}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # cửa sổ lệnh Windows mặc định cp1252
    ap = argparse.ArgumentParser(description="SHWD (VOC) -> nhãn YOLO 5 lớp đồ bảo hộ")
    ap.add_argument("voc", type=Path, help="thư mục VOC2028 (có Annotations, JPEGImages, ImageSets)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gan-them", action="store_true", help="tự gán person (mô hình tư thế) và vest (mô hình PPE)")
    ap.add_argument("--chi-xep-anh", action="store_true", help="chỉ chép ảnh khớp với nhãn đã có trong --out")
    ap.add_argument("--person-conf", type=float, default=0.5)
    ap.add_argument("--vest-conf", type=float, default=0.5)
    ap.add_argument("--ppe-model", default="models/ppe_v1_finetune_best.pt")
    ap.add_argument("--xem-truoc", type=int, default=60, help="số ảnh vẽ sẵn khung để soát nhanh")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if args.chi_xep_anh:
        return xep_anh(args.voc, args.out)

    import cv2

    pose = ppe = None
    if args.gan_them:
        from ultralytics import YOLO

        pose = YOLO("yolo26n-pose.pt")
        ppe = YOLO(args.ppe_model)
        vest_id = [k for k, v in ppe.names.items() if v == "vest"][0]

    counts: Counter = Counter()
    skipped: Counter = Counter()
    previews = 0
    started = time.time()
    for split, names in split_lists(args.voc).items():
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            xml = args.voc / "Annotations" / f"{name}.xml"
            img_path = find_image(args.voc, name)
            if not xml.is_file() or img_path is None:
                skipped["thieu_tep"] += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped["anh_hong"] += 1
                continue
            h, w = img.shape[:2]          # không tin size trong xml: SHWD có tệp ghi 0
            boxes, other = read_voc(xml)
            for o in other:
                skipped[f"lop_{o}"] += 1
            lines = [yolo(c, *b, w, h) for c, *b in boxes]
            drawn = [(c, *b) for c, *b in boxes]
            if args.gan_them:
                rp = pose.predict(img, conf=args.person_conf, verbose=False, device=args.device)[0]
                for b in rp.boxes.xyxy.tolist():
                    lines.append(yolo(0, *b, w, h))
                    drawn.append((0, *b))
                rv = ppe.predict(img, conf=args.vest_conf, imgsz=416, verbose=False, device=args.device)[0]
                for b, c in zip(rv.boxes.xyxy.tolist(), rv.boxes.cls.tolist()):
                    if int(c) == vest_id:
                        lines.append(yolo(3, *b, w, h))
                        drawn.append((3, *b))
            for ln in lines:
                counts[CLASSES[int(ln.split()[0])]] += 1
            (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            if previews < args.xem_truoc and i % 97 == 0:
                vis = img.copy()
                for c, x1, y1, x2, y2 in drawn:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[c], 2)
                    cv2.putText(vis, CLASSES[c], (int(x1), max(12, int(y1) - 4)), 0, 0.45, COLORS[c], 1)
                (args.out / "xem_truoc").mkdir(exist_ok=True)
                cv2.imwrite(str(args.out / "xem_truoc" / f"{split}_{name}.jpg"), vis)
                previews += 1
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{len(names)} ảnh, {time.time() - started:.0f}s")
        print(f"{split}: {len(names)} ảnh")

    n_train = len(list((args.out / "labels" / "train").glob("*.txt")))
    n_val = len(list((args.out / "labels" / "val").glob("*.txt")))
    (args.out / "data.yaml").write_text(
        "# Bộ SHWD đã chuyển sang 5 lớp của mô hình đồ bảo hộ. Sửa path cho đúng máy.\n"
        "path: D:/safety-ai/data/shwd_yolo\ntrain: images/train\nval: images/val\n"
        f"names: {dict(enumerate(CLASSES))}\n", encoding="utf-8")
    (args.out / "README.txt").write_text(
        f"SHWD -> YOLO 5 lớp, tạo {time.strftime('%d/%m/%Y')} bằng scripts/chuyen_shwd.py (repo safety-ai).\n"
        "Nguồn: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset (repo MIT; ảnh từ Google/Baidu\n"
        "và SCUT-HEAD). Ghi nguồn trong báo cáo. Ảnh KHÔNG kèm ở đây: tải VOC2028.zip (1,0 GB) theo link\n"
        "Google Drive trong README của repo đó, giải nén, rồi chạy\n"
        "  python scripts/chuyen_shwd.py <VOC2028> --out <thư mục này> --chi-xep-anh\n\n"
        f"Lớp: {', '.join(f'{i} {c}' for i, c in enumerate(CLASSES))}\n"
        "helmet/no_helmet: nhãn gốc của SHWD (người gán tay, khoanh ĐẦU; no_helmet gồm cả đầu đội mũ thường).\n"
        + ("person/vest: TỰ GÁN bằng mô hình tư thế (ngưỡng "
           f"{args.person_conf}) và ppe_v1_finetune_best (ngưỡng {args.vest_conf}) — là nhãn nháp, soát ở xem_truoc/.\n"
           if args.gan_them else "person/vest: KHÔNG gán (chạy lại với --gan-them).\n")
        + "no_vest: không gán.\n"
        f"Ảnh: train {n_train}, val {n_val} (theo ImageSets/Main trainval/test của SHWD).\n"
        f"Số khung: {dict(counts)}. Bỏ qua: {dict(skipped)}.\n"
        "Gợi ý: trộn vào tập train của mô hình PPE (ảnh SCUT-HEAD là lớp học, nhiều đầu nhỏ — có thể chỉ lấy một phần).\n",
        encoding="utf-8")
    print(f"Xong: {dict(counts)}, bỏ qua {dict(skipped)}, {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
