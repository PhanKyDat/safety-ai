"""Bài B4: đo tài nguyên Frigate với 1 / 4 / 8 camera.

    python scripts/do_b4.py                      # cả 3 mức, mỗi mức 3 phút
    python scripts/do_b4.py --so-camera 1 4      # chỉ đo 1 và 4 camera
    python scripts/do_b4.py --phut 5

Script tự sinh `frigate/config.yml` cho từng mức camera, khởi động lại Frigate, chờ ổn
định rồi lấy mẫu CPU / RAM / FPS. Xong trả lại cấu hình cũ.

Mặc định đo **Frigate thuần**. Thêm `--module` để đo "Frigate + module": ở mỗi mức, script
tự chạy `safety.cli` đọc đúng chừng ấy camera, lấy mẫu thêm CPU/RAM của tiến trình module và
GPU/VRAM (nvidia-smi), cuối mức đọc nhịp xử lý mà module tự in ra:

    python scripts/do_b4.py --module --extra-model models/ppe_v2_best.pt --extra-model models/fire_v2_best.pt

Module ghi sự cố vào `data/incidents_b4.db` (riêng, không lẫn thống kê) và không gửi Telegram.

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


def gpu_mau() -> tuple[float, float]:
    """(% dùng GPU, VRAM đã dùng MB) của card đầu tiên; (0, 0) nếu không có nvidia-smi."""
    out = chay(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"], timeout=15)
    try:
        dung, vram = out.strip().splitlines()[0].split(",")
        return float(dung), float(vram)
    except (ValueError, IndexError):
        return 0.0, 0.0


_DA_GAP: dict[int, object] = {}


def module_mau(tien_trinh) -> tuple[float, float]:
    """(CPU %, RAM MB) của tiến trình module kèm tiến trình con. CPU 100% = 1 luồng, như Docker.

    Trên Windows, python.exe của venv chỉ là trình khởi chạy, Python thật là tiến trình con.
    psutil tính CPU theo chênh lệch giữa hai lần gọi trên CÙNG một đối tượng Process, nên phải
    giữ lại đối tượng của từng pid; tạo mới mỗi lần thì lúc nào cũng ra 0%.
    """
    if tien_trinh is None:
        return 0.0, 0.0
    cpu = ram = 0.0
    try:
        cac = [tien_trinh] + tien_trinh.children(recursive=True)
    except Exception:  # noqa: BLE001 - tiến trình vừa thoát
        return 0.0, 0.0
    for p in cac:
        p = _DA_GAP.setdefault(p.pid, p)
        try:
            cpu += p.cpu_percent(None)
            ram += p.memory_info().rss / 2**20
        except Exception:  # noqa: BLE001
            continue
    return cpu, ram


def chay_module(so_camera: int, args, so_luot: int):
    """Bật safety.cli đọc xuong_han + kho_1..; trả (Popen, psutil.Process, tệp log)."""
    import os

    import psutil

    cams = ["xuong_han"] + [f"kho_{i}" for i in range(1, so_camera)]
    lenh = [sys.executable, "-m", "safety.cli", "--max-frames", str(so_luot),
            "--db", args.module_db, "--frigate", args.frigate_api]
    for c in cams:
        lenh += ["--cam", f"{c}=rtsp://127.0.0.1:8554/{c}"]
    for m in args.extra_model:
        lenh += ["--extra-model", m]
    (GOC / "data").mkdir(exist_ok=True)
    log = (GOC / "data" / f"b4_module_{so_camera}cam.log").open("w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(lenh, cwd=GOC, stdout=log, stderr=subprocess.STDOUT, env=env)
    tt = psutil.Process(p.pid)
    tt.cpu_percent(None)  # mốc đầu cho lần đo sau
    return p, tt, log


def doc_ket_qua_module(duong_dan: Path) -> dict:
    """Lấy 'Nhịp xử lý: X khung/giây' và 'trung vị Y ms/lô' từ log của safety.cli."""
    chu = duong_dan.read_text(encoding="utf-8", errors="replace")
    fps = re.search(r"Nhịp xử lý: ([\d.]+) khung/giây", chu)
    ms = re.search(r"trung vị ([\d.]+) ms/", chu)
    return {"module_fps": float(fps.group(1)) if fps else 0.0,
            "module_ms_lo": float(ms.group(1)) if ms else 0.0}


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


def do_mot_muc(so_camera: int, phut: float, moi: int, args=None) -> dict | None:
    print(f"\n=== {so_camera} camera ===")
    sinh_config(so_camera)
    chay(["docker", "compose", "restart", "frigate"], timeout=180)
    if not cho_frigate():
        print("  Frigate không khởi động được — bỏ mức này")
        return None

    co_module = bool(args and args.module)
    p = tt = log = None
    if co_module:
        # Đủ lượt cho 60 s chờ + thời gian đo + 30 s dư, ở nhịp 5 khung/giây.
        so_luot = int((60 + phut * 60 + 30) * 5)
        p, tt, log = chay_module(so_camera, args, so_luot)
        print(f"  đã bật module (pid {p.pid}, {so_luot} lượt)")

    print("  chờ 60 giây cho ổn định...")
    time.sleep(60)
    if co_module:
        module_mau(tt)  # bỏ phần nạp mô hình khỏi số CPU

    mau = []
    het = time.monotonic() + phut * 60
    while time.monotonic() < het:
        m = mot_mau()
        if m:
            m["cpu_module"], m["ram_module"] = module_mau(tt)
            m["gpu_dung"], m["vram"] = gpu_mau()
            mau.append(m)
            them = (f"  | module CPU {m['cpu_module']:5.1f}%  RAM {m['ram_module']:5.0f} MB  "
                    f"GPU {m['gpu_dung']:3.0f}%  VRAM {m['vram']:5.0f} MB") if co_module else ""
            print(f"  CPU {m['cpu_frigate']:5.1f}%  RAM {m['ram_frigate']:5.0f} MB  "
                  f"FPS tb {m['fps_trung_binh']:.1f}  detector {m['detector_ms']:.1f} ms{them}")
        time.sleep(moi)

    kq_module = {}
    if co_module:
        try:
            p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            print("  module chưa tự dừng sau 3 phút — đã tắt, không có số nhịp xử lý")
        log.close()
        kq_module = doc_ket_qua_module(Path(log.name))
        if p.returncode:
            print(f"  module thoát mã {p.returncode}, xem {log.name}")

    if not mau:
        return None
    tb = lambda k: statistics.mean(m[k] for m in mau)  # noqa: E731
    them_kq = {}
    if co_module:
        them_kq = {
            "cpu_module_tb": round(tb("cpu_module"), 1),
            "ram_module_tb": round(tb("ram_module")),
            "gpu_dung_tb": round(tb("gpu_dung"), 1),
            "vram_tb": round(tb("vram")),
            "cpu_tong_tb": round(tb("cpu_frigate") + tb("cpu_module") + tb("cpu_sim"), 1),
            **kq_module,
        }
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
        **them_kq,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Đo tài nguyên Frigate theo số camera")
    p.add_argument("--so-camera", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--phut", type=float, default=3.0, help="đo bao lâu ở mỗi mức")
    p.add_argument("--moi", type=int, default=15, help="bao nhiêu giây lấy một mẫu")
    p.add_argument("--ghi-chu", default="frigate thuan")
    p.add_argument("--ra", default=None)
    p.add_argument("--module", action="store_true", help="chạy kèm safety.cli để đo Frigate + module")
    p.add_argument("--extra-model", action="append", default=[], help="truyền thẳng cho safety.cli")
    p.add_argument("--frigate-api", default="http://127.0.0.1:5050/api")
    p.add_argument("--module-db", default="data/incidents_b4.db")
    args = p.parse_args()
    if args.module and args.ghi_chu == "frigate thuan":
        args.ghi_chu = "frigate + module"

    if not MAU.is_file():
        print(f"Không thấy {MAU}")
        return 1

    luu = CONFIG.read_text(encoding="utf-8") if CONFIG.is_file() else None
    ket_qua = []
    try:
        for n in args.so_camera:
            kq = do_mot_muc(n, args.phut, args.moi, args)
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
    if args.module:
        print()
        print(f"{'Camera':<8}{'CPU module':>12}{'RAM (MB)':>10}{'GPU':>7}{'VRAM (MB)':>11}"
              f"{'Nhịp module':>13}{'ms/lô':>8}{'CPU tổng':>10}")
        for k in ket_qua:
            print(f"{k['so_camera']:<8}{k['cpu_module_tb']:>11.1f}%{k['ram_module_tb']:>10}"
                  f"{k['gpu_dung_tb']:>6.0f}%{k['vram_tb']:>11}{k['module_fps']:>13.2f}"
                  f"{k['module_ms_lo']:>8.0f}{k['cpu_tong_tb']:>9.1f}%")
        print("VRAM là của cả card (gồm màn hình); CPU tổng = Frigate + module + máy giả lập.")
    print(f"\nĐã ghi {ra}")
    print("GPU: Frigate đang dùng bộ phát hiện CPU nên không chạm tới GPU (xem detectors trong config).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
