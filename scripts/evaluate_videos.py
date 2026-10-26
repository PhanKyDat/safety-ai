r"""Chấm module trên bộ video kiểm thử có đáp án: F1 theo sự cố, AUROC/AUPRC theo đoạn.

    .venv\Scripts\python.exe scripts/evaluate_videos.py data/test_videos/*.mp4 ^
        --extra-model models/ppe_v1_finetune_best.pt --extra-model models/fire_v1_yolo26s.pt

Mỗi video cần tệp đáp án cùng tên, đuôi .csv (xem docs/dap-an-video.md và
docs/mau-dap-an.csv). Kết quả in ra màn hình và ghi vào runs/eval/<thời điểm>/:
tong_hop.csv, su_co.csv (từng sự cố module báo), doan.csv (điểm từng đoạn) và
roc_pr.png. Mỗi bản mô hình mới chạy lại lệnh này là có số mới để so.

Đọc tệp trực tiếp, lấy mẫu 5 khung/giây theo thời gian trong video; mỗi video một tên
camera riêng nên bộ bám vết bắt đầu lại từ đầu (bẫy đo số 1 của nhóm: dùng chung tracker
giữa các lần chạy). Vùng nguy hiểm do Frigate xử lý nên không chấm ở đây.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from safety.evaluation import (
    LabelScore,
    auroc,
    average_precision,
    frame_scores,
    load_gt,
    match_events,
    pr_points,
    roc_points,
    window_truth,
)
from safety.incidents import IncidentProcessor
from safety.pipeline import SafetyPipeline

MODULE_LABELS = ("no_helmet", "no_vest", "fire", "smoke", "fall")


def fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def run_video(path: Path, cam: str, pipeline: SafetyPipeline, fps: float, window: float):
    processor = IncidentProcessor(cam, publisher=None, snapshot_dir=None)
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    alerts: list[tuple[str, float]] = []
    win_scores: dict[str, dict[int, float]] = defaultdict(dict)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ts = idx / src_fps
            (result,) = pipeline.process_batch([(cam, ts, frame)])
            for inc in processor.handle(frame, result):
                alerts.append((inc.label.value, ts))
            k = int(ts // window)
            for label, s in frame_scores(result, processor.ppe).items():
                win_scores[label][k] = max(win_scores[label].get(k, 0.0), s)
        idx += 1
    cap.release()
    return alerts, win_scores, idx / src_fps


def main() -> int:
    ap = argparse.ArgumentParser(description="Chấm module trên video có đáp án")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--extra-model", action="append", default=[])
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--window", type=float, default=2.0, help="độ dài một đoạn khi tính AUROC/AUPRC (giây)")
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="sự cố tới muộn tối đa bấy nhiêu giây sau khi hết khoảng vi phạm vẫn tính là đúng")
    ap.add_argument("--out", type=Path, default=None, help="thư mục ghi kết quả (mặc định runs/eval/<thời điểm>)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    missing = [v for v in args.videos if not v.with_suffix(".csv").is_file()]
    if missing:
        print("Thiếu tệp đáp án (.csv cùng tên) cho: " + ", ".join(map(str, missing)))
        return 1

    out = args.out or Path("runs/eval") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    pipeline = SafetyPipeline(model=args.model, camera="eval", device=args.device,
                              extra_models=args.extra_model)
    have = {c.value for c in pipeline.detect_classes}
    if getattr(pipeline.model, "task", None) == "pose":
        have.add("fall")  # luật ngã dùng điểm khớp của mô hình tư thế, không cần lớp riêng

    totals: dict[str, LabelScore] = defaultdict(LabelScore)
    truth: dict[str, list[int]] = defaultdict(list)
    scores: dict[str, list[float]] = defaultdict(list)
    alert_rows, window_rows = [], []
    total_seconds = 0.0
    for i, video in enumerate(args.videos):
        gt = load_gt(video.with_suffix(".csv"))
        skipped = sorted({e.label for e in gt if e.label not in MODULE_LABELS})
        gt = [e for e in gt if e.label in MODULE_LABELS]
        alerts, win, duration = run_video(video, f"v{i}", pipeline, args.fps, args.window)
        total_seconds += duration
        for label, s in match_events(gt, alerts, args.tolerance).items():
            totals[label].add(s)
        for label, ts in alerts:
            alert_rows.append([video.name, label, f"{ts:.1f}"])
        for label in MODULE_LABELS:
            y = window_truth(gt, label, duration, args.window)
            s = [win[label].get(k, 0.0) for k in range(len(y))]
            truth[label] += y
            scores[label] += s
            window_rows += [[video.name, label, k * args.window, t, f"{v:.4f}"] for k, (t, v) in enumerate(zip(y, s))]
        note = f"; bỏ qua nhãn Frigate tự xử lý: {skipped}" if skipped else ""
        print(f"{video.name}: {duration:.1f}s, {len(gt)} khoảng vi phạm, {len(alerts)} sự cố báo{note}")

    hours = total_seconds / 3600
    print(f"\nTổng {len(args.videos)} video, {total_seconds:.0f}s. Đoạn dài {args.window:g}s, "
          f"dung sai báo muộn {args.tolerance:g}s.\n")
    header = ["nhan", "mo_hinh_co_lop", "khoang_vi_pham", "bao_dung", "bao_nham", "bo_sot", "bao_lap",
              "precision", "recall", "f1", "do_tre_trung_vi_s", "bao_nham_moi_gio",
              "doan_co", "doan_khong", "auroc", "auprc", "auprc_ngau_nhien"]
    rows = []
    print(f"{'nhãn':<10} {'có lớp':>6} {'khoảng':>6} {'đúng':>5} {'nhầm':>5} {'sót':>4} {'lặp':>4} "
          f"{'P':>7} {'R':>7} {'F1':>7} {'trễ':>6} {'nhầm/h':>7} {'AUROC':>6} {'AUPRC':>6} {'(ngẫu nhiên)':>12}")
    for label in MODULE_LABELS:
        s = totals[label]
        y, sc = truth[label], scores[label]
        a, ap_ = auroc(y, sc), average_precision(y, sc)
        lat = statistics.median(s.latencies) if s.latencies else None
        fph = s.fp / hours if hours else None
        has = label in have
        base = sum(y) / len(y) if y and sum(y) else None
        rows.append([label, int(has), s.gt, s.tp, s.fp, s.fn, s.repeats, s.precision, s.recall, s.f1,
                     lat, fph, sum(y), len(y) - sum(y), a, ap_, base])
        print(f"{label:<10} {'có' if has else 'KHÔNG':>6} {s.gt:>6} {s.tp:>5} {s.fp:>5} {s.fn:>4} {s.repeats:>4} "
              f"{fmt(s.precision):>7} {fmt(s.recall):>7} {fmt(s.f1):>7} "
              f"{(f'{lat:.1f}s' if lat is not None else '–'):>6} {(f'{fph:.1f}' if fph is not None else '–'):>7} "
              f"{fmt(a, False):>6} {fmt(ap_, False):>6} {fmt(base, False):>12}")
    print("\n'–' = chưa tính được: không có khoảng vi phạm nào của nhãn đó, hoặc (AUROC) "
          "không có đoạn nào không vi phạm. 'KHÔNG' = chưa có mô hình nhận ra lớp này.\n"
          "AUROC 0,5 = đoán bừa. AUPRC phải so với cột 'ngẫu nhiên' (tỉ lệ đoạn có vi phạm): "
          "AUPRC bằng cột đó nghĩa là mô hình không phân biệt được gì.")

    with open(out / "tong_hop.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([header] + [[("" if v is None else v) for v in r] for r in rows])
    with open(out / "su_co.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["video", "nhan", "giay"]] + alert_rows)
    with open(out / "doan.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["video", "nhan", "bat_dau_s", "co_vi_pham", "diem"]] + window_rows)
    plot(truth, scores, out / "roc_pr.png")
    print(f"\nĐã ghi kết quả vào {out}")
    return 0


def plot(truth, scores, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=130)
    drew = False
    for label in MODULE_LABELS:
        y, s = truth[label], scores[label]
        a, ap_ = auroc(y, s), average_precision(y, s)
        if a is not None:
            fx, fy = zip(*roc_points(y, s))
            ax1.plot(fx, fy, label=f"{label} (AUROC {a:.2f})".replace(".", ","))
            drew = True
        if ap_ is not None:
            rx, ry = zip(*pr_points(y, s))
            ax2.plot(rx, ry, label=f"{label} (AUPRC {ap_:.2f})".replace(".", ","))
    ax1.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax1.set(xlabel="Tỉ lệ đoạn không vi phạm bị báo (FPR)", ylabel="Tỉ lệ đoạn vi phạm bắt được (TPR)",
            title="Đường cong ROC theo đoạn", xlim=(0, 1), ylim=(0, 1.02))
    ax2.set(xlabel="Recall", ylabel="Precision", title="Đường cong Precision–Recall theo đoạn",
            xlim=(0, 1), ylim=(0, 1.02))
    for ax in (ax1, ax2):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    if not drew:
        ax1.text(0.5, 0.5, "Chưa đủ đoạn có và không vi phạm", ha="center")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    raise SystemExit(main())
