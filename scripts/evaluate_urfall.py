r"""Chấm luật ngã trên bộ công khai UR Fall Detection (Kwolek & Kępski, ĐH Rzeszów).

Bộ này có 30 video ngã và 40 video sinh hoạt (ngồi xuống, cúi nhặt đồ, nằm lên giường...),
camera 0 đặt ngang người, kèm nhãn TỪNG KHUNG của tác giả: -1 không nằm, 0 đang ngã
(tư thế chuyển tiếp, tác giả bỏ ra khi chấm), 1 nằm dưới đất. Giấy phép CC BY-NC-SA 4.0:
dùng cho học tập được, không đưa video lên repo.

Tải về (khoảng 94 MB) vào data/public/urfall/:
    fall-01..30-cam0.mp4, adl-01..40-cam0.mp4, urfall-cam0-falls.csv, urfall-cam0-adls.csv
    từ https://fenix.ur.edu.pl/~mkepski/ds/data/

    .venv\Scripts\python.exe scripts/evaluate_urfall.py data/public/urfall

Mỗi video mp4 của tác giả ghép ảnh độ sâu (nửa trái) với ảnh màu (nửa phải); script chỉ
lấy nửa phải, 320x240.

Hai mức chấm:

1. Từng khung, tư thế nằm: luật có nhận ra người đang nằm không (FallRule.lying), so với
   nhãn tác giả. Ra P/R/F1 của quyết định nằm, và AUROC/AUPRC của điểm nghiêng (góc/90,
   cùng điểm mà evaluate_videos.py dùng). Khung nhãn 0 bị bỏ, như tác giả làm.

2. Từng video, có báo sự cố không: video ngã có báo là đúng, video sinh hoạt có báo là
   nhầm. Vướng: video của bộ này dừng ngay sau cú ngã (người nằm 0,2–2,1 giây rồi hết),
   còn luật đòi nằm liên tục 3 giây nên trên video gốc không bao giờ kịp báo. Vì vậy mặc
   định giữ khung cuối thêm --giu-cuoi giây (4): mô phỏng người ngã nằm yên tại chỗ, đúng
   tình huống nguy hiểm nhất (bất tỉnh). Mọi video, kể cả sinh hoạt, đều được giữ như
   nhau. --giu-cuoi 0 là chấm trên video gốc.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from safety.contracts import DetectClass
from safety.evaluation import auroc, average_precision
from safety.pipeline import SafetyPipeline
from safety.rules import FallConfig, FallRule, body_tilt


def load_labels(folder: Path) -> dict[str, dict[int, int]]:
    labels: dict[str, dict[int, int]] = collections.defaultdict(dict)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        with open(folder / name, newline="") as f:
            for row in csv.reader(f):
                labels[row[0]][int(row[1])] = int(row[2])
    return labels


def frame_state(rule: FallRule, result) -> tuple[bool, float, bool, float | None, float | None]:
    """(có ai đang nằm theo luật, điểm nghiêng cao nhất, có thấy người không,
    góc thân, góc chân của người nghiêng nhất) của một khung."""
    lying, score, seen = False, 0.0, False
    angles: tuple[float | None, float | None] = (None, None)
    for d in result.detections:
        if d.cls is not DetectClass.PERSON:
            continue
        seen = True
        if rule.lying(d, result):
            lying = True
        torso, legs = body_tilt(d, rule.config.kp_conf)
        if torso is not None:
            s = (min(torso, legs) if legs is not None else torso) / 90.0
            if s >= score:
                score, angles = s, (torso, legs)
    return lying, score, seen, *angles


def run_clip(path: Path, cam: str, pipeline: SafetyPipeline, labels: dict[int, int],
             fps: float, hold: float, config: FallConfig):
    rule = FallRule(cam, config)
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    # (khung, nhãn tác giả, luật nói nằm, điểm, thấy người, góc thân, góc chân)
    frames: list[tuple] = []
    alerts: list[float] = []
    idx, last, ts = 0, None, 0.0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        idx += 1                                  # nhãn tác giả đánh số khung từ 1
        if (idx - 1) % step:
            continue
        frame = raw[:, raw.shape[1] // 2:].copy()
        last, ts = frame, (idx - 1) / src_fps
        (result,) = pipeline.process_batch([(cam, ts, frame)])
        alerts += [inc.ts for inc in rule.update(result)]
        if idx in labels:
            frames.append((idx, labels[idx], *frame_state(rule, result)))
    cap.release()
    end = ts
    k = 0
    while last is not None and k * (1 / fps) < hold:
        k += 1
        t = end + k / fps
        (result,) = pipeline.process_batch([(cam, t, last)])
        alerts += [inc.ts for inc in rule.update(result)]
    return frames, alerts, idx / src_fps


def main() -> int:
    ap = argparse.ArgumentParser(description="Chấm luật ngã trên UR Fall Detection")
    ap.add_argument("folder", type=Path, help="thư mục chứa fall-NN-cam0.mp4, adl-NN-cam0.mp4 và 2 tệp nhãn")
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps", type=float, default=10.0, help="số khung lấy mỗi giây (video gốc 30)")
    ap.add_argument("--giu-cuoi", type=float, default=4.0, dest="hold",
                    help="giữ khung cuối thêm bấy nhiêu giây (xem giải thích đầu tệp); 0 = video gốc")
    ap.add_argument("--duration", type=float, default=FallConfig.duration,
                    help="luật đòi nằm liên tục bấy nhiêu giây")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    labels = load_labels(args.folder)
    clips = sorted(args.folder.glob("fall-*-cam0.mp4")) + sorted(args.folder.glob("adl-*-cam0.mp4"))
    if not clips:
        print(f"Không thấy video nào trong {args.folder}")
        return 1
    out = args.out or Path("runs/eval") / ("urfall-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    pipeline = SafetyPipeline(model=args.model, camera="urfall", device=args.device)
    config = FallConfig(duration=args.duration)

    truth, scores, decided, seen = [], [], [], []
    clip_rows, frame_rows = [], []
    tp = fp = fn = tn = 0
    latencies = []
    for i, clip in enumerate(clips):
        name = clip.name.replace("-cam0.mp4", "")
        lab = labels.get(name, {})
        frames, alerts, duration = run_clip(clip, f"u{i}", pipeline, lab, args.fps, args.hold, config)
        for k, author, lying, score, person, torso, legs in frames:
            frame_rows.append([name, k, author, int(person), "" if torso is None else f"{torso:.1f}",
                               "" if legs is None else f"{legs:.1f}"])
            if author == 0:
                continue
            truth.append(int(author == 1))
            scores.append(score)
            decided.append(lying)
            seen.append(person)
        is_fall = name.startswith("fall")
        fell_at = min((f - 1) / 30.0 for f, l in lab.items() if l >= 0) if is_fall else None
        alerted = bool(alerts)
        if is_fall and alerted:
            tp += 1
            latencies.append(alerts[0] - fell_at)
        elif is_fall:
            fn += 1
        elif alerted:
            fp += 1
        else:
            tn += 1
        lying_frames = sum(1 for _, a, *_ in frames if a == 1)
        clip_rows.append([name, "nga" if is_fall else "sinh_hoat", f"{duration:.1f}",
                          "" if fell_at is None else f"{fell_at:.2f}", lying_frames,
                          sum(1 for _, _, ly, *_ in frames if ly), int(alerted),
                          "" if not alerts else f"{alerts[0]:.2f}"])
        print(f"{name}: {duration:.1f}s, {'NGÃ' if is_fall else 'sinh hoạt'}, "
              f"{'báo lúc %.1fs' % alerts[0] if alerts else 'không báo'}")

    # Mức 1: từng khung
    pos = sum(truth)
    ftp = sum(1 for t, d in zip(truth, decided) if t and d)
    ffp = sum(1 for t, d in zip(truth, decided) if not t and d)
    fp_, fr_ = (ftp / (ftp + ffp) if ftp + ffp else None), (ftp / pos if pos else None)
    ff1 = 2 * fp_ * fr_ / (fp_ + fr_) if fp_ and fr_ else None
    a, ap_ = auroc(truth, scores), average_precision(truth, scores)
    # Tách hai nguồn lỗi: mô hình không thấy người, hay thấy mà luật đoán sai tư thế
    lying_unseen = sum(1 for t, s in zip(truth, seen) if t and not s)
    st = [t for t, s in zip(truth, seen) if s]
    ss = [x for x, s in zip(scores, seen) if s]
    sd = [d for d, s in zip(decided, seen) if s]
    s_tp = sum(1 for t, d in zip(st, sd) if t and d)
    s_r = s_tp / sum(st) if sum(st) else None
    sa, sap = auroc(st, ss), average_precision(st, ss)
    # Mức 2: từng video
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else None

    def pct(x):
        return "–" if x is None else f"{100 * x:.1f}%"

    print(f"\nTừng khung ({len(truth)} khung, {pos} khung nằm theo tác giả, bỏ khung đang ngã):")
    print(f"  quyết định nằm của luật: P {pct(fp_)}  R {pct(fr_)}  F1 {pct(ff1)}")
    print(f"  điểm nghiêng: AUROC {a:.3f}  AUPRC {ap_:.3f}  (ngẫu nhiên {pos / len(truth):.3f})")
    print(f"  khung nằm mà mô hình không thấy người: {lying_unseen}/{pos} "
          f"({pct(lying_unseen / pos if pos else None)})")
    print(f"  chỉ tính khung thấy người ({len(st)} khung, {sum(st)} nằm): R {pct(s_r)}  "
          f"AUROC {sa:.3f}  AUPRC {sap:.3f}  (ngẫu nhiên {sum(st) / len(st):.3f})")
    print(f"\nTừng video (giữ khung cuối {args.hold:g}s, luật đòi nằm {args.duration:g}s):")
    print(f"  ngã: bắt {tp}/{tp + fn}; sinh hoạt: báo nhầm {fp}/{fp + tn}")
    print(f"  P {pct(p)}  R {pct(r)}  F1 {pct(f1)}", end="")
    if latencies:
        latencies.sort()
        print(f"  độ trễ trung vị từ lúc bắt đầu ngã {latencies[len(latencies) // 2]:.1f}s")
    else:
        print()

    with open(out / "tung_video.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["video", "loai", "dai_s", "bat_dau_nga_s", "khung_nam_tac_gia",
                                  "khung_luat_noi_nam", "co_bao", "bao_luc_s"]] + clip_rows)
    with open(out / "tung_khung.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["video", "khung", "nhan_tac_gia", "thay_nguoi", "goc_than", "goc_chan"]]
                                + frame_rows)
    with open(out / "tong_hop.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([
            ["muc", "so_mau", "duong_tinh", "precision", "recall", "f1", "auroc", "auprc", "auprc_ngau_nhien"],
            ["khung", len(truth), pos, fp_, fr_, ff1, a, ap_, pos / len(truth)],
            ["khung_thay_nguoi", len(st), sum(st), "", s_r, "", sa, sap, sum(st) / len(st)],
            ["video", len(clips), tp + fn, p, r, f1, "", "", ""],
        ])
    print(f"\nĐã ghi kết quả vào {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
