"""Bài B4: đo tài nguyên Frigate với 1 / 4 / 8 camera.

    python scripts/do_b4.py                      # cả 3 mức, mỗi mức 3 phút
    python scripts/do_b4.py --so-camera 1 4      # chỉ đo 1 và 4 camera
    python scripts/do_b4.py --phut 5

Script tự sinh `frigate/config.yml` cho từng mức camera, khởi động lại Frigate, chờ ổn
định rồi lấy mẫu CPU / RAM / FPS. Xong trả lại cấu hình cũ.

Đây là phần **Frigate thuần**. Muốn số "Frigate + module" thì mở thêm một cửa sổ chạy
`safety.cli` rồi chạy script này với `--ghi-chu "co module"` — script chỉ đo container,
không quan tâm bên ngoài chạy gì.

Camera kho dùng luồng chép thẳng (không mã hoá lại) nên máy giả lập không thành chỗ
nghẽn; xem docs/frigate-setup.md mục tuần 8.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOC = Path(__file__).resolve().parent.parent
CONFIG = GOC / "frigate" / "config.yml"
MAU = GOC / "frigate" / "config.example.yml"


def chay(lenh: list[str], timeout: float = 60) -> str:
    try:
        return subprocess.run(lenh, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def camera_chi_ghi_hinh(ten: str) -> dict:
    return {
        "ffmpeg": {"inputs": [{
            "path": f"rtsp://127.0.0.1:8554/{ten}",
            "input_args": "preset-rtsp-restream",
            "roles": ["record"],
        }]},
        "detect": {"enabled": False},
        "motion": {"enabled": False},
        "record": {"enabled": True},
    }


def sinh_config(so_camera: int) -> None:
    """Viết frigate/config.yml với đúng `so_camera` camera (xuong_han luôn có)."""
    goc = yaml.safe_load(MAU.read_text(encoding="utf-8"))
    cameras = {"xuong_han": goc["cameras"]["xuong_han"]}
    for i in range(1, so_camera):
        ten = f"kho_{i}"
        cameras[ten] = goc["cameras"].get(ten) or camera_chi_ghi_hinh(ten)
        goc.setdefault("go2rtc", {}).setdefault("streams", {})[ten] = f"rtsp://rtsp-sim:8554/{ten}"
    goc["cameras"] = cameras
    CONFIG.write_text(yaml.safe_dump(goc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def cho_frigate(han: float = 180) -> bool:
    het = time.monotonic() + han
    while time.monotonic() < het:
        tt = chay(["docker", "inspect", "-f", "{{.State.Health.Status}}", "frigate"]).strip()
        if tt == "healthy":
            return True
        time.sleep(3)
    return False


def mot_mau() -> dict | None:
    out = chay(["docker", "exec", "frigate", "curl", "-fsS", "http://127.0.0.1:5000/api/stats"])
    try:
        s = json.loads(out)
    except ValueError:
        return None

    tai = chay(["docker", "stats", "--no-stream", "--format",
                "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"])
    cpu = {}
    for dong in tai.strip().splitlines():
        phan = dong.split("\t")
        if len(phan) < 3:
            continue
        try:
            mem = float(re.match(r"([\d.]+)", phan[2]).group(1))
            if "GiB" in phan[2].split("/")[0]:
                mem *= 1024
            cpu[phan[0]] = (float(phan[1].rstrip("%")), mem)
        except (ValueError, AttributeError):
            continue

    fps = [c.get("camera_fps", 0) for c in s.get("cameras", {}).values()]
    return {
        "cpu_frigate": cpu.get("frigate", (0, 0))[0],
        "ram_frigate": cpu.get("frigate", (0, 0))[1],
        "cpu_sim": cpu.get("rtsp-sim", (0, 0))[0],
        "fps_trung_binh": sum(fps) / len(fps) if fps else 0,
        "fps_thap_nhat": min(fps) if fps else 0,
        "detector_ms": next(iter(s.get("detectors", {}).values()), {}).get("inference_speed", 0),
        "so_camera_chay": len(s.get("cameras", {})),
    }


def do_mot_muc(so_camera: int, phut: float, moi: int) -> dict | None:
    print(f"\n=== {so_camera} camera ===")
    sinh_config(so_camera)
    chay(["docker", "compose", "restart", "frigate"], timeout=180)
    if not cho_frigate():
        print("  Frigate không khởi động được — bỏ mức này")
        return None

    print("  chờ 60 giây cho ổn định...")
    time.sleep(60)

    mau = []
    het = time.monotonic() + phut * 60
    while time.monotonic() < het:
        m = mot_mau()
        if m:
            mau.append(m)
            print(f"  CPU {m['cpu_frigate']:5.1f}%  RAM {m['ram_frigate']:5.0f} MB  "
                  f"FPS tb {m['fps_trung_binh']:.1f}  detector {m['detector_ms']:.1f} ms")
        time.sleep(moi)

    if not mau:
        return None
    tb = lambda k: statistics.mean(m[k] for m in mau)  # noqa: E731
    return {
        "so_camera": so_camera,
        "so_mau": len(mau),
        "cpu_frigate_tb": round(tb("cpu_frigate"), 1),
        "cpu_frigate_dinh": round(max(m["cpu_frigate"] for m in mau), 1),
        "ram_frigate_tb": round(tb("ram_frigate")),
        "cpu_sim_tb": round(tb("cpu_sim"), 1),
        "fps_trung_binh": round(tb("fps_trung_binh"), 2),
        "fps_thap_nhat": round(min(m["fps_thap_nhat"] for m in mau), 2),
        "detector_ms": round(tb("detector_ms"), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Đo tài nguyên Frigate theo số camera")
    p.add_argument("--so-camera", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--phut", type=float, default=3.0, help="đo bao lâu ở mỗi mức")
    p.add_argument("--moi", type=int, default=15, help="bao nhiêu giây lấy một mẫu")
    p.add_argument("--ghi-chu", default="frigate thuan")
    p.add_argument("--ra", default=None)
    args = p.parse_args()

    if not MAU.is_file():
        print(f"Không thấy {MAU}")
        return 1

    luu = CONFIG.read_text(encoding="utf-8") if CONFIG.is_file() else None
    ket_qua = []
    try:
        for n in args.so_camera:
            kq = do_mot_muc(n, args.phut, args.moi)
            if kq:
                kq["ghi_chu"] = args.ghi_chu
                ket_qua.append(kq)
    except KeyboardInterrupt:
        print("\nDừng sớm.")
    finally:
        if luu is not None:
            CONFIG.write_text(luu, encoding="utf-8")
        else:
            shutil.copy(MAU, CONFIG)
        chay(["docker", "compose", "restart", "frigate"], timeout=180)
        print("\nĐã trả lại cấu hình Frigate như cũ.")

    if not ket_qua:
        return 1

    ra = Path(args.ra or f"data/b4_{datetime.now():%Y-%m-%d_%H%M}.csv")
    ra.parent.mkdir(parents=True, exist_ok=True)
    with ra.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ket_qua[0]))
        w.writeheader()
        w.writerows(ket_qua)

    print()
    print(f"{'Camera':<8}{'CPU Frigate':>13}{'đỉnh':>8}{'RAM (MB)':>10}"
          f"{'FPS tb':>9}{'FPS thấp':>10}{'Detector':>10}{'CPU sim':>9}")
    for k in ket_qua:
        print(f"{k['so_camera']:<8}{k['cpu_frigate_tb']:>12.1f}%{k['cpu_frigate_dinh']:>7.0f}%"
              f"{k['ram_frigate_tb']:>10}{k['fps_trung_binh']:>9.2f}{k['fps_thap_nhat']:>10.2f}"
              f"{k['detector_ms']:>9.1f}ms{k['cpu_sim_tb']:>8.1f}%")
    print(f"\nĐã ghi {ra}")
    print("GPU: Frigate đang dùng bộ phát hiện CPU nên không chạm tới GPU (xem detectors trong config).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
