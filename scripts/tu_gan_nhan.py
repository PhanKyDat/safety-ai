r"""Tự gán nhãn từ video train_ để bổ sung dữ liệu huấn luyện (nhãn nháp, người phải soát lại).

Hai chế độ:

  ppe  (mặc định): lấy mẫu khung có người, ghi nhãn YOLO đủ 5 lớp của mô hình đồ bảo hộ
       (0 person, 1 helmet, 2 no_helmet, 3 vest, 4 no_vest). person/helmet/vest lấy từ mô hình
       hiện có; no_helmet/no_vest lấy từ luật vắng mặt (thấy rõ đầu mà không có mũ thì khoanh
       đầu, thấy rõ thân mà không có áo thì khoanh thân) — đúng hai lớp mô hình hiện chưa ra
       được lần nào vì thiếu ảnh.

  nen: video KHÔNG có lửa/khói (đèn pha, đèn đường, đèn trần, nắng, lau sậy...). Lấy các khung
       mà mô hình cháy báo nhầm, ghi tệp nhãn RỖNG (ảnh nền) để dạy mô hình "đây không phải lửa".

    .venv\Scripts\python.exe scripts\tu_gan_nhan.py data\train\train_khong_mu_01_dem.mp4 --out data\gan_nhan\ppe
    .venv\Scripts\python.exe scripts\tu_gan_nhan.py data\train\train_bay_den_vang_01_dem.mp4 --che-do nen --out data\gan_nhan\nen

Từ chối video test_*: video chấm điểm mà đưa vào huấn luyện thì mọi số đo sau đó đẹp giả.
Kết quả: <out>/images, <out>/labels (định dạng YOLO), <out>/xem_truoc (ảnh vẽ sẵn khung để soát
nhanh), <out>/README.txt. Nhãn là NHÁP: mô hình còn sót mũ/áo thì luật sẽ khoanh nhầm no_helmet/
no_vest, nên phải xem lại từng ảnh (CVAT, Roboflow, LabelImg...) trước khi train.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from safety.contracts import BBox, DetectClass, Detection, FrameResult
from safety.pipeline import SafetyPipeline
from safety.rules import PPEConfig, PPERule

PPE_CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest"]
CLASS_ID = {DetectClass.PERSON: 0, DetectClass.HELMET: 1, DetectClass.NO_HELMET: 2,
            DetectClass.VEST: 3, DetectClass.NO_VEST: 4}
COLORS = {0: (255, 160, 0), 1: (0, 200, 0), 2: (0, 0, 255), 3: (0, 220, 220), 4: (255, 0, 255)}


def head_box(person: Detection, width: int, height: int) -> BBox | None:
    """Khung đầu từ 5 điểm mặt/tai, nới theo bề ngang vai để trùm cả tóc/mũ."""
    k = person.keypoints
    pts = [p for p in k[:5] if p.conf >= 0.5]
    if len(pts) < 2:
        return None
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    shoulder = abs(k[5].x - k[6].x) if k[5].conf >= 0.5 and k[6].conf >= 0.5 else 0.0
    size = max(max(xs) - min(xs), shoulder * 0.55, person.bbox.width * 0.25)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = size * 0.75
    return BBox(max(0, cx - half), max(0, cy - half * 1.2), min(width, cx + half), min(height, cy + half * 0.8))


def torso_box(person: Detection, width: int, height: int) -> BBox | None:
    """Khung thân từ 2 vai + 2 hông, nới một chút ra hai bên như áo phản quang."""
    k = person.keypoints
    pts = [k[i] for i in (5, 6, 11, 12)]
    if any(p.conf < 0.5 for p in pts):
        return None
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    pad_x = (max(xs) - min(xs)) * 0.15
    pad_y = (max(ys) - min(ys)) * 0.08
    return BBox(max(0, min(xs) - pad_x), max(0, min(ys) - pad_y),
                min(width, max(xs) + pad_x), min(height, max(ys) + pad_y))


def yolo_line(cls: int, b: BBox, w: int, h: int) -> str:
    cx, cy = (b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h
    return f"{cls} {cx:.6f} {cy:.6f} {(b.x2 - b.x1) / w:.6f} {(b.y2 - b.y1) / h:.6f}"


def ppe_labels(result: FrameResult, rule: PPERule, min_conf: float) -> list[tuple[int, BBox]]:
    out: list[tuple[int, BBox]] = []
    w, h = result.width, result.height
    for d in result.detections:
        if d.cls in (DetectClass.HELMET, DetectClass.VEST) and d.conf >= min_conf:
            out.append((CLASS_ID[d.cls], d.bbox))
    for p in result.detections:
        if p.cls is not DetectClass.PERSON or p.conf < 0.5:
            continue
        out.append((0, p.bbox))
        if not p.keypoints:
            continue
        # Đúng logic luật đang chạy: trả về người = vi phạm do vắng mặt, False = có đồ bảo hộ.
        if rule._violating(p, result, DetectClass.HELMET, DetectClass.NO_HELMET, True) is p:
            box = head_box(p, w, h)
            if box is not None:
                out.append((2, box))
        if rule._violating(p, result, DetectClass.VEST, DetectClass.NO_VEST, False) is p:
            box = torso_box(p, w, h)
            if box is not None:
                out.append((4, box))
    return out


def draw(frame, labels):
    img = frame.copy()
    for cls, b in labels:
        c = COLORS[cls]
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, 2)
        cv2.putText(img, PPE_CLASSES[cls], (int(b.x1), max(14, int(b.y1) - 5)), 0, 0.55, c, 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Tự gán nhãn nháp từ video train_")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--che-do", choices=("ppe", "nen"), default="ppe")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--moi-giay", type=float, default=1.0, help="lấy 1 khung mỗi bấy nhiêu giây")
    ap.add_argument("--min-conf", type=float, default=0.4, help="ngưỡng giữ khung mũ/áo của mô hình")
    ap.add_argument("--ppe-model", default="models/ppe_v1_finetune_best.pt")
    ap.add_argument("--fire-model", default="models/fire_v1_yolo26s.pt")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    bad = [v for v in args.videos if v.name.startswith("test_")]
    if bad:
        print("Từ chối video test_ (chỉ để chấm, không được đưa vào huấn luyện): " + ", ".join(v.name for v in bad))
        return 1
    if args.che_do == "ppe":
        pipe = SafetyPipeline(model="yolo26n-pose.pt", camera="gan_nhan", device=args.device,
                              extra_models=[args.ppe_model])
    else:
        pipe = SafetyPipeline(model=args.fire_model, camera="gan_nhan", device=args.device, conf=0.2)
    for sub in ("images", "labels", "xem_truoc"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    counts = [0] * len(PPE_CLASSES)
    saved = 0
    for vi, video in enumerate(args.videos):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps * args.moi_giay))
        rule = PPERule(f"v{vi}", PPEConfig())
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                (result,) = pipe.process_batch([(f"v{vi}", idx / fps, frame)])
                name = f"{video.stem}_{idx:06d}"
                if args.che_do == "ppe":
                    labels = ppe_labels(result, rule, args.min_conf)
                    if any(c == 0 for c, _ in labels):           # chỉ giữ khung có người
                        cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), frame)
                        (args.out / "labels" / f"{name}.txt").write_text(
                            "\n".join(yolo_line(c, b, result.width, result.height) for c, b in labels) + "\n")
                        cv2.imwrite(str(args.out / "xem_truoc" / f"{name}.jpg"), draw(frame, labels))
                        for c, _ in labels:
                            counts[c] += 1
                        saved += 1
                else:
                    hits = [d for d in result.detections if d.cls in (DetectClass.FIRE, DetectClass.SMOKE)]
                    if hits:                                       # mô hình báo nhầm: ảnh nền khó
                        cv2.imwrite(str(args.out / "images" / f"{name}.jpg"), frame)
                        (args.out / "labels" / f"{name}.txt").write_text("")
                        vis = frame.copy()
                        for d in hits:
                            b = d.bbox
                            cv2.rectangle(vis, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 2)
                            cv2.putText(vis, f"nham {d.cls.value} {d.conf:.2f}", (int(b.x1), max(14, int(b.y1) - 5)),
                                        0, 0.55, (0, 0, 255), 2)
                        cv2.imwrite(str(args.out / "xem_truoc" / f"{name}.jpg"), vis)
                        saved += 1
            idx += 1
        cap.release()
        print(f"{video.name}: {idx / fps:.0f}s, đã lưu tổng {saved} ảnh")

    if args.che_do == "ppe":
        summary = ", ".join(f"{n} {c}" for n, c in zip(PPE_CLASSES, counts))
        readme = (f"Nhãn NHÁP tự gán {time.strftime('%d/%m/%Y')} bằng scripts/tu_gan_nhan.py (chế độ ppe).\n"
                  f"Lớp (đúng thứ tự mô hình đồ bảo hộ): {', '.join(f'{i} {n}' for i, n in enumerate(PPE_CLASSES))}\n"
                  f"person/helmet/vest: mô hình {args.ppe_model} (ngưỡng {args.min_conf}).\n"
                  "no_helmet/no_vest: luật vắng mặt (thấy rõ đầu/thân bằng điểm khớp mà không có mũ/áo).\n"
                  "PHẢI SOÁT LẠI: mô hình sót mũ/áo thì luật khoanh nhầm no_helmet/no_vest; mũ lưỡi trai bị nhận\n"
                  "là helmet thì phải sửa thành no_helmet. Xem nhanh ở xem_truoc/.\n"
                  f"Số khung: {summary}. Ảnh: {saved}.\n")
    else:
        readme = (f"Ảnh NỀN KHÓ tự lấy {time.strftime('%d/%m/%Y')} bằng scripts/tu_gan_nhan.py (chế độ nen).\n"
                  f"Video không có lửa/khói; đây là các khung mô hình {args.fire_model} báo nhầm (ngưỡng 0,2).\n"
                  "Tệp nhãn RỖNG = ảnh nền, thêm thẳng vào tập train của mô hình cháy. Xem khung báo nhầm ở xem_truoc/.\n"
                  f"Ảnh: {saved}.\n")
    (args.out / "README.txt").write_text(readme, encoding="utf-8")
    print(readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
