"""Đo riêng từng khâu của vòng lặp module khi chạy cùng Frigate.

    python scripts/measure_stages.py --seconds 30
    python scripts/measure_stages.py --seconds 30 --extra-model models/fire_v1_yolo26s.pt
    python scripts/measure_stages.py --seconds 30 --no-model     # chỉ đọc luồng

Mỗi khung ghi lại: chờ khung (thời gian nằm trong FrameSource), nhận diện, luật +
chụp ảnh, và độ trễ của khung (từ lúc luồng đọc nhận khung tới lúc bắt đầu xử lý).
Song song đó đếm số khung luồng đọc nhận được từ Frigate mỗi giây, và đọc CPU của
các container bằng `docker stats` ở giữa lần đo.

Bỏ 5 khung đầu (GPU khởi động) và lấy trung vị, không lấy trung bình: bẫy thứ 4
trong lịch sử đo của nhóm là trung bình gộp cả khung khởi động.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from safety.incidents import IncidentProcessor
from safety.stream import FrameSource

WARMUP = 5


class CountingSource(FrameSource):
    """FrameSource đếm mỗi khung luồng đọc nhận được, để biết Frigate phát bao nhiêu khung/giây."""

    def __init__(self, *a, **kw):
        self.arrivals: list[float] = []
        super().__init__(*a, **kw)

    @property
    def _latest(self):
        return self.__dict__.get("_latest_v")

    @_latest.setter
    def _latest(self, value):
        if value is not None:
            self.arrivals.append(value[0])
        self.__dict__["_latest_v"] = value


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def docker_cpu() -> str:
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.CPUPerc}}"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(không đọc được docker stats: {e})"
    return ", ".join(out.split("\n")).strip(", ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="rtsp://127.0.0.1:8554/xuong_han")
    p.add_argument("--camera", default="xuong_han")
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--extra-model", action="append", default=[])
    p.add_argument("--no-model", action="store_true", help="Chỉ đọc luồng, không nhận diện")
    p.add_argument("--device", default="0")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--seconds", type=float, default=30.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    pipeline = None
    if not args.no_model:
        from safety.pipeline import SafetyPipeline

        pipeline = SafetyPipeline(model=args.model, camera=args.camera, device=args.device,
                                  extra_models=args.extra_model)
    processor = IncidentProcessor(args.camera, snapshot_dir=None)
    src = CountingSource(args.source, target_fps=args.fps)

    cpu: list[str] = []
    rows = []          # (yield_at, wait_ms, stale_ms, infer_ms, rules_ms)
    t_prev_end = None
    deadline = None
    for ts, frame in src:
        t0 = time.time()
        if deadline is None:
            deadline = t0 + args.seconds
            threading.Timer(args.seconds / 2, lambda: cpu.append(docker_cpu())).start()
        wait_ms = 1000 * (t0 - t_prev_end) if t_prev_end is not None else 0.0
        stale_ms = 1000 * (t0 - ts)

        t1 = time.perf_counter()
        result = pipeline.process(ts, frame) if pipeline is not None else None
        t2 = time.perf_counter()
        if result is not None:
            processor.handle(frame, result)
        t3 = time.perf_counter()

        rows.append((t0, wait_ms, stale_ms, 1000 * (t2 - t1), 1000 * (t3 - t2)))
        t_prev_end = time.time()
        if t_prev_end >= deadline:
            break

    steady = rows[WARMUP:]
    span = steady[-1][0] - steady[0][0]
    fps = (len(steady) - 1) / span if span > 0 else 0.0
    arrivals = [a for a in src.arrivals if steady[0][0] <= a <= steady[-1][0]]
    gaps = [1000 * (b[0] - a[0]) for a, b in zip(steady, steady[1:])]

    def line(name: str, i: int) -> str:
        xs = [r[i] for r in steady]
        return f"  {name:<28} trung vị {statistics.median(xs):6.1f} ms   95%: {pct(xs, 0.95):6.1f} ms   max {max(xs):6.1f} ms"

    mode = "chỉ đọc luồng" if pipeline is None else f"{1 + len(pipeline.extras)} mô hình"
    print(f"\n=== {mode}, mục tiêu {args.fps:g} khung/giây, đo {span:.1f}s (bỏ {WARMUP} khung đầu) ===")
    print(f"Frigate phát tới module : {len(arrivals) / span:.1f} khung/giây")
    print(f"Module xử lý            : {fps:.2f} khung/giây ({len(steady)} khung)")
    print(f"Khoảng cách giữa 2 khung: trung vị {statistics.median(gaps):.0f} ms, 95%: {pct(gaps, 0.95):.0f} ms, max {max(gaps):.0f} ms")
    print(line("chờ khung (trong FrameSource)", 1))
    print(line("độ trễ khung lúc bắt đầu xử lý", 2))
    print(line("nhận diện", 3))
    print(line("luật", 4))
    slow = sum(g > 1000 / args.fps * 1.5 for g in gaps)
    when = [f"{a[0] - steady[0][0]:.1f}s" for a, b in zip(steady, steady[1:])
            if 1000 * (b[0] - a[0]) > 1000 / args.fps * 1.5]
    print(f"  khoảng cách > {1.5 * 1000 / args.fps:.0f} ms: {slow}/{len(gaps)} lần, lúc {', '.join(when) or '-'}")
    print(f"CPU container giữa lần đo: {cpu[0] if cpu else '(chưa đọc kịp)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
