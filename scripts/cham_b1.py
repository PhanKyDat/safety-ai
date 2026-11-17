r"""Bài B1: chấm "không đội mũ" của Frigate (mô hình phân loại mu_bao_ho) trên đúng bộ video và
đúng cách chấm của module (safety.evaluation.match_events, dung sai báo muộn 5 s).

Cách đo (17/11, xem docs/doi-chieu.md mục 2):
  1. Ghép 12 video kiểm thử thành data/samples/b1_test.mp4: 15 s đen đầu (chờ Frigate nối
     luồng), 4 s đen giữa các đoạn (bộ bám vết bắt đầu lại); vị trí từng đoạn ở data/b1/doan.csv.
  2. Frigate có camera `b1` (detect 640x360, 5 khung/s, bật classification.custom.mu_bao_ho).
  3. Đẩy video một lượt vào rtsp://rtsp-sim:8554/b1 bằng ffmpeg -re, ghi giờ bắt đầu vào
     data/b1/t0.txt.
  4. Chạy script này: lấy sự kiện person của camera b1 kèm thuộc tính mu_bao_ho, đổi giờ sự
     kiện về giây trong từng video, rồi chấm như module.

    .venv\Scripts\python.exe scripts\cham_b1.py

Sự kiện Frigate gắn "khong_mu" được coi là một lần báo. Frigate gán thuộc tính cho cả lượt
bám vết chứ không cho biết lúc nào, nên thời điểm báo lấy là lúc lượt bám vết chồng lên khoảng
vi phạm sớm nhất (có lợi cho Frigate: không tính độ trễ).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC))
sys.path.insert(0, str(GOC / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from b1_doi_chung import LOP_KHONG_MU, nhan_phan_loai  # noqa: E402
from safety.evaluation import load_gt, load_gt_table, match_events  # noqa: E402


def dap_an(ten: str) -> list:
    rieng = GOC / "data" / "samples" / ten.replace(".mp4", ".csv")
    if rieng.is_file():
        return load_gt(rieng)
    return load_gt_table(GOC / "data" / "videos_test" / "nhan.csv").get(ten, [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", default="http://127.0.0.1:5000/api")
    ap.add_argument("--camera", default="b1")
    ap.add_argument("--doan", type=Path, default=GOC / "data" / "b1" / "doan.csv")
    ap.add_argument("--t0", type=Path, default=GOC / "data" / "b1" / "t0.txt")
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args()

    t0 = float(args.t0.read_text().strip())
    doan = [(r["video"], float(r["bat_dau_s"]), float(r["dai_s"]))
            for r in csv.DictReader(open(args.doan, encoding="utf-8"))]
    url = (f"{args.api}/events?cameras={args.camera}&labels=person&after={t0 - 5:.0f}"
           f"&limit=2000&include_attributes=1")
    with urllib.request.urlopen(url, timeout=20) as f:
        events = json.load(f)

    theo_video: dict[str, list[tuple[float, float, str | None]]] = {v: [] for v, _, _ in doan}
    for e in events:
        a = e["start_time"] - t0
        b = (e.get("end_time") or e["start_time"]) - t0
        for v, bd, dai in doan:
            if bd - 1 <= a <= bd + dai + 1:
                theo_video[v].append((max(0.0, a - bd), min(dai, b - bd), nhan_phan_loai(e)))
                break

    tong: dict[str, list[int]] = {}
    dem_lop: dict[str, int] = {}
    print(f"{len(events)} sự kiện person của camera {args.camera} từ T0={t0:.0f}\n")
    for v, _, _ in doan:
        gt = [g for g in dap_an(v) if g.label == "no_helmet"]
        bao = []
        for a, b, lop in theo_video[v]:
            dem_lop[str(lop)] = dem_lop.get(str(lop), 0) + 1
            if lop != LOP_KHONG_MU:
                continue
            trung = [g for g in gt if a <= g.end + args.tolerance and b >= g.start]
            bao.append(("no_helmet", max(a, trung[0].start) if trung else a))
        s = match_events(gt, bao, args.tolerance).get("no_helmet")
        luot = ", ".join(f"{a:.0f}-{b:.0f}s:{lop}" for a, b, lop in theo_video[v]) or "không có"
        print(f"{v:30s} khoảng={len(gt)} " + (f"đúng={s.tp} nhầm={s.fp} sót={s.fn} lặp={s.repeats}" if s else "")
              + f"  | lượt bám vết: {luot}")
        if s:
            t = tong.setdefault("no_helmet", [0, 0, 0, 0, 0])
            t[0] += s.gt; t[1] += s.tp; t[2] += s.fp; t[3] += s.fn; t[4] += s.repeats

    gt_n, tp, fp, fn, lap = tong.get("no_helmet", [0] * 5)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / gt_n if gt_n else None
    f1 = 2 * p * r / (p + r) if p and r else None
    pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    print(f"\nFrigate (mu_bao_ho) — thiếu mũ: khoảng {gt_n}, đúng {tp}, nhầm {fp}, sót {fn}, lặp {lap}"
          f"  P {pct(p)}  R {pct(r)}  F1 {pct(f1)}")
    print("Các lượt bám vết theo thuộc tính:", dem_lop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
