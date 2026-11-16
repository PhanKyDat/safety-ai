"""Soi luật ngã trên từng khung: góc thân, góc chân, độ tin cậy vai/hông, và luật kết luận gì.

    python scripts/soi_goc_nga.py data/videos_test/test_nga_01_dem.mp4 [...] --fps 5

Dùng khi luật ngã bỏ sót: cho biết trượt ở ngưỡng góc thân (`lying_angle`), góc chân
(`leg_angle`), hay ở chỗ không thấy rõ vai/hông (`kp_conf`) nên luật "chưa kết luận".
Đọc đáp án trong `nhan.csv` để tách khung lúc đang nằm với khung lúc còn đứng.
In bảng theo thời gian và tóm tắt mỗi video; không ghi gì ra ngoài.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import cv2

GOC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GOC))

from safety.contracts import BBox, DetectClass, Detection, Keypoint  # noqa: E402
from safety.rules import FallConfig, body_tilt  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def doc_dap_an(thu_muc: Path) -> dict[str, list[tuple[float, float]]]:
    ra: dict[str, list[tuple[float, float]]] = {}
    tep = thu_muc / "nhan.csv"
    if not tep.is_file():
        return ra
    with tep.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("nhan") == "fall" and r.get("bat_dau_giay"):
                ra.setdefault(r["tep"], []).append((float(r["bat_dau_giay"]), float(r["ket_thuc_giay"])))
    return ra


def main() -> int:
    p = argparse.ArgumentParser(description="Soi góc thân/chân của luật ngã theo từng khung")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--chi-tom-tat", action="store_true", help="không in từng khung")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    cfg = FallConfig()
    print(f"Ngưỡng hiện tại: thân ≥ {cfg.lying_angle}°, chân ≥ {cfg.leg_angle}°, "
          f"điểm khớp ≥ {cfg.kp_conf}, liên tục {cfg.duration}s")

    for video in args.videos:
        dap_an = doc_dap_an(video.parent).get(video.name, [])
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        buoc = max(1, round(fps / args.fps))
        print(f"\n=== {video.name} ({cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
              f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}), đáp án ngã: {dap_an or 'không có'}")
        if not args.chi_tom_tat:
            print(f"{'giây':>5} {'đáp án':>7} {'người':>6} {'thân°':>6} {'chân°':>6} "
                  f"{'vai':>5} {'hông':>5} {'w/h':>5}  luật")
        i = 0
        tk = {"nam": [], "dung": []}
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % buoc:
                i += 1
                continue
            t = i / fps
            i += 1
            nga = any(a <= t <= b for a, b in dap_an)
            r = model.predict(fr, conf=args.conf, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                dong = (None, None, None, None, None, "không thấy người")
            else:
                j = int(r.boxes.conf.argmax())
                x1, y1, x2, y2 = (float(v) for v in r.boxes.xyxy[j])
                kps = r.keypoints.data[j].cpu().numpy()
                det = Detection(track_id=1, cls=DetectClass.PERSON, bbox=BBox(x1, y1, x2, y2),
                                conf=float(r.boxes.conf[j]),
                                keypoints=tuple(Keypoint(float(a), float(b), float(c)) for a, b, c in kps))
                than, chan = body_tilt(det, cfg.kp_conf)
                vai = min(kps[5][2], kps[6][2])
                hong = min(kps[11][2], kps[12][2])
                if than is None:
                    ket = "chưa rõ (vai/hông mờ)"
                elif than < cfg.lying_angle:
                    ket = "đứng (thân thẳng)"
                elif chan is not None and chan < cfg.leg_angle:
                    ket = "cúi (chân thẳng)"
                else:
                    ket = "NẰM"
                dong = (det.conf, than, chan, vai, hong, ket, (x2 - x1) / max(1.0, y2 - y1))
            (tk["nam"] if nga else tk["dung"]).append(dong)
            if not args.chi_tom_tat:
                f = lambda v, d=0: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
                wh = dong[6] if len(dong) > 6 else None
                print(f"{t:5.1f} {'NGÃ' if nga else '':>7} {f(dong[0], 2):>6} {f(dong[1]):>6} {f(dong[2]):>6} "
                      f"{f(dong[3], 2):>5} {f(dong[4], 2):>5} {f(wh, 2):>5}  {dong[5]}")

        nam = tk["nam"]
        if nam:
            dem = {}
            for d in nam:
                dem[d[5]] = dem.get(d[5], 0) + 1
            goc = [d[1] for d in nam if d[1] is not None]
            print(f"  Lúc đang ngã/nằm ({len(nam)} khung): " + ", ".join(f"{k} {v}" for k, v in dem.items()))
            if goc:
                print(f"  Góc thân lúc nằm: trung vị {statistics.median(goc):.0f}°, "
                      f"nhỏ nhất {min(goc):.0f}°, lớn nhất {max(goc):.0f}°")
        dung = [d[1] for d in tk["dung"] if d[1] is not None]
        if dung:
            print(f"  Góc thân lúc chưa ngã: lớn nhất {max(dung):.0f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
