r"""Đo luật đồ bảo hộ với mô hình PPE thật trên một tệp video.

    .venv\Scripts\python.exe scripts/measure_ppe_video.py data/samples/xuong_han.mp4 ^
        --extra-model models/ppe_v1_finetune_best.pt
    ... --save-frames runs/ppe_check     # lưu vài khung có vẽ hộp để xem bằng mắt

Bảng số của mô hình (precision/recall trên tập ảnh) chưa trả lời được câu hỏi của hệ
thống: luật có báo được người thiếu mũ/áo không. Luật chỉ đếm khi lớp vi phạm nằm trong
vùng đầu (mũ) hoặc khung người (áo) và kéo dài 3 giây, nên script đo đúng thứ luật nhìn
thấy: với mỗi người (track_id), tỉ lệ khung có đồ bảo hộ, khung vi phạm, và khung không
kết luận được, rồi chạy chính PPERule để đếm sự cố.

Đọc tệp trực tiếp và lấy mẫu 5 khung/giây theo thời gian trong video, mỗi tệp một
pipeline mới (bẫy đo số 1 và số 3 trong lịch sử nhóm: dùng chung tracker giữa các lần
chạy, và lấy mốc thời gian từ đồng hồ máy).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from safety.contracts import DetectClass
from safety.pipeline import SafetyPipeline
from safety.rules import PPERule, _CHECKS

FPS = 5.0
COLORS = {
    DetectClass.PERSON: (200, 200, 200),
    DetectClass.HELMET: (80, 175, 76),
    DetectClass.VEST: (80, 175, 76),
    DetectClass.NO_HELMET: (40, 40, 230),
    DetectClass.NO_VEST: (40, 40, 230),
}


def draw(frame, result):
    out = frame.copy()
    for d in result.detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox.to_list())
        color = COLORS.get(d.cls, (0, 200, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{d.cls.value}{'#' + str(d.track_id) if d.track_id is not None else ''} {d.conf:.2f}"
        cv2.putText(out, tag, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def measure(path: str, args) -> None:
    pipeline = SafetyPipeline(model=args.model, camera="xuong_han", device=args.device,
                              conf=args.conf, extra_models=args.extra_model)
    rule = PPERule("xuong_han")
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / FPS))

    # stats[track][label] = [có đồ bảo hộ, vi phạm, không kết luận]
    stats: dict[int, dict] = defaultdict(lambda: {label: [0, 0, 0] for label, *_ in _CHECKS})
    class_frames: dict[DetectClass, int] = defaultdict(int)
    incidents = []
    saved = 0
    idx = sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        ts = idx / src_fps
        idx += 1
        sampled += 1
        result = pipeline.process(ts, frame)
        for c in {d.cls for d in result.detections}:
            class_frames[c] += 1
        for person in result.detections:
            if person.cls is not DetectClass.PERSON or person.track_id is None:
                continue
            for label, ok_cls, bad_cls, head_only in _CHECKS:
                v = rule._violating(person, result, ok_cls, bad_cls, head_only)
                stats[person.track_id][label][0 if v is False else 2 if v is None else 1] += 1
        for inc in rule.update(result):
            incidents.append((ts, inc.label.value, inc.sub_label, inc.score))
        if args.save_frames and sampled % args.every == 0 and saved < args.max_saved:
            out_dir = Path(args.save_frames)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{Path(path).stem}_{ts:05.1f}s.jpg"), draw(frame, result))
            saved += 1
    cap.release()

    print(f"\n=== {path}: {sampled} khung lấy mẫu ({FPS:g}/giây)")
    print("Số khung có mỗi lớp: " + ", ".join(f"{c.value} {n}" for c, n in sorted(class_frames.items(), key=lambda x: x[0].value)))
    print(f"{'người':>7} {'khung':>6} | {'mũ':>5} {'KHÔNG mũ':>9} {'?':>5} | {'áo':>5} {'KHÔNG áo':>9} {'?':>5}")
    for tid in sorted(stats, key=lambda t: -sum(stats[t][_CHECKS[0][0]])):
        row = stats[tid]
        n = sum(row[_CHECKS[0][0]])
        if n < args.min_frames:
            continue
        h, v = row[_CHECKS[0][0]], row[_CHECKS[1][0]]
        print(f"{'#' + str(tid):>7} {n:>6} | {h[0]:>5} {h[1]:>9} {h[2]:>5} | {v[0]:>5} {v[1]:>9} {v[2]:>5}")
    print(f"Sự cố luật phát ra: {len(incidents)}")
    for ts, label, sub, score in incidents:
        print(f"  {ts:5.1f}s  {label:<10} {sub}  score {score:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--extra-model", action="append", default=[])
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-frames", type=int, default=5, help="bỏ người xuất hiện ít hơn số khung này")
    ap.add_argument("--save-frames", default=None, help="thư mục lưu khung có vẽ hộp")
    ap.add_argument("--every", type=int, default=10, help="lưu 1 khung mỗi N khung lấy mẫu")
    ap.add_argument("--max-saved", type=int, default=8)
    args = ap.parse_args()
    for path in args.videos:
        measure(path, args)


if __name__ == "__main__":
    main()
