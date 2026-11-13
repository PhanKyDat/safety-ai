"""Theo dõi hệ thống trong lúc chạy dài, phục vụ tiêu chí "chạy liên tục ≥ 8 giờ".

    python scripts/do_on_dinh.py --gio 8
    python scripts/do_on_dinh.py --gio 8 --moi 60 --ra data/on_dinh_2026-11-11.csv
    python scripts/do_on_dinh.py --tong-ket data/on_dinh_2026-11-11.csv   # đọc lại kết quả cũ

Cứ mỗi `--moi` giây ghi một dòng CSV: FPS từng camera, tốc độ bộ phát hiện, CPU/RAM
từng container, CPU/RAM tiến trình module (`safety.cli`) và cầu nối (`safety.mqtt_bridge`)
nếu đang chạy, GPU/VRAM, số sự cố trong DB, và số lần Frigate mất luồng phải nối lại.
Cuối buổi in tổng kết: chạy được bao lâu, FPS thấp nhất, RAM đầu buổi so với cuối buổi
(có rò bộ nhớ không), sau mỗi lần mất luồng có tự nối lại không.

`--thu-ngat-moi 2` cứ 2 giờ khởi động lại camera giả lập một lần, để có bằng chứng
"tự kết nối lại khi mất luồng" thay vì chờ mạng tự rớt.

Đo từ ngoài qua API và `docker stats` nên không đụng gì vào hệ thống đang chạy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Log Frigate ghi dòng này mỗi lần mất luồng rồi nối lại.
MAU_MAT_LUONG = re.compile(r"Unable to read frames|ffmpeg process crashed|Error opening input", re.I)


def chay(lenh: list[str], timeout: float = 20) -> str:
    try:
        r = subprocess.run(lenh, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def api_frigate() -> dict:
    """Lấy /api/stats qua docker exec: cổng 5000 không mở ra host."""
    out = chay(["docker", "exec", "frigate", "curl", "-fsS", "http://127.0.0.1:5000/api/stats"])
    try:
        return json.loads(out)
    except ValueError:
        return {}


def tai_container() -> dict[str, tuple[float, float]]:
    """CPU (%) và RAM (MB) của từng container."""
    out = chay(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"])
    ket_qua = {}
    for dong in out.strip().splitlines():
        phan = dong.split("\t")
        if len(phan) < 3:
            continue
        ten, cpu, mem = phan[0], phan[1], phan[2]
        try:
            cpu_so = float(cpu.rstrip("%"))
            mem_so = float(re.match(r"([\d.]+)", mem).group(1))
            if "GiB" in mem.split("/")[0]:
                mem_so *= 1024
        except (ValueError, AttributeError):
            continue
        ket_qua[ten] = (cpu_so, mem_so)
    return ket_qua


def dem_mat_luong(tu_giay: int) -> int:
    out = chay(["docker", "compose", "logs", "--since", f"{tu_giay}s", "frigate"], timeout=30)
    return sum(1 for d in out.splitlines() if MAU_MAT_LUONG.search(d))


def dem_su_co(db: Path) -> int:
    if not db.is_file():
        return 0
    try:
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=3)
        try:
            return con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return 0


_TIEN_TRINH: dict[int, object] = {}


def tien_trinh_python(tu_khoa: str) -> tuple[int, float, float]:
    """(số tiến trình, CPU %, RAM MB) của các tiến trình Python có `tu_khoa` trong lệnh chạy.

    python.exe của venv trên Windows là trình khởi chạy + một tiến trình con: cả hai đều
    mang lệnh `-m safety.cli`, nên cộng RAM thì chỉ lấy tiến trình nặng nhất mỗi cặp là đủ —
    ở đây lấy tổng nhưng trình khởi chạy chỉ ~5 MB, không làm lệch xu hướng.
    """
    try:
        import psutil
    except ImportError:
        return 0, 0.0, 0.0
    so = 0
    cpu = ram = 0.0
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if tu_khoa not in " ".join(p.info["cmdline"] or []):
                continue
            p = _TIEN_TRINH.setdefault(p.pid, p)  # giữ đối tượng cũ để cpu_percent có mốc
            so += 1
            cpu += p.cpu_percent(None)
            ram += p.memory_info().rss / 2**20
        except Exception:  # noqa: BLE001 - tiến trình vừa thoát / không có quyền
            continue
    return so, cpu, ram


def gpu() -> tuple[float, float]:
    out = chay(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"], timeout=15)
    try:
        dung, vram = out.strip().splitlines()[0].split(",")
        return float(dung), float(vram)
    except (ValueError, IndexError):
        return 0.0, 0.0


def mot_dong(db: Path, moi: int) -> dict:
    s = api_frigate()
    cameras = s.get("cameras", {})
    detectors = s.get("detectors", {})
    tai = tai_container()
    dong = {
        "luc": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ts": round(time.time(), 1),
        "frigate_song": int(bool(s)),
        "detector_ms": round(next(iter(detectors.values()), {}).get("inference_speed", 0) or 0, 2),
        "su_co": dem_su_co(db),
        "mat_luong": dem_mat_luong(moi),
        "thu_ngat": 0,
    }
    dong["module_song"], dong["cpu_module"], dong["ram_module"] = tien_trinh_python("safety.cli")
    dong["cau_noi_song"], _, dong["ram_cau_noi"] = tien_trinh_python("safety.mqtt_bridge")
    dong["cpu_module"] = round(dong["cpu_module"], 1)
    dong["ram_module"] = round(dong["ram_module"])
    dong["ram_cau_noi"] = round(dong["ram_cau_noi"])
    dong["gpu_dung"], dong["vram"] = gpu()
    for ten, c in sorted(cameras.items()):
        dong[f"fps_{ten}"] = c.get("camera_fps", 0)
        dong[f"detect_{ten}"] = c.get("detection_fps", 0)
    for ten, (cpu, mem) in sorted(tai.items()):
        dong[f"cpu_{ten}"] = cpu
        dong[f"ram_{ten}"] = round(mem)
    return dong


def tong_ket(duong_dan: Path) -> None:
    with duong_dan.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Chưa có dòng nào.")
        return

    so = lambda r, k: float(r.get(k) or 0)  # noqa: E731
    bat_dau, ket_thuc = so(rows[0], "ts"), so(rows[-1], "ts")
    gio = (ket_thuc - bat_dau) / 3600
    song = sum(int(so(r, "frigate_song")) for r in rows)
    mat_luong = sum(int(so(r, "mat_luong")) for r in rows)
    cameras = sorted(k[4:] for k in rows[0] if k.startswith("fps_"))

    print(f"Chạy từ {rows[0]['luc']} đến {rows[-1]['luc']} — {gio:.1f} giờ, {len(rows)} lần đo")
    print(f"Frigate trả lời API : {song}/{len(rows)} lần ({song / len(rows):.1%})")
    print(f"Số lần mất luồng    : {mat_luong}")
    print(f"Sự cố ghi được      : {int(so(rows[0], 'su_co'))} → {int(so(rows[-1], 'su_co'))}")
    print()
    print(f"{'Camera':<14}{'FPS nhỏ nhất':>14}{'FPS trung bình':>16}")
    for cam in cameras:
        gia_tri = [so(r, f"fps_{cam}") for r in rows]
        print(f"{cam:<14}{min(gia_tri):>14.1f}{sum(gia_tri) / len(gia_tri):>16.1f}")
    print()
    for ten in ("frigate", "rtsp-sim", "mosquitto"):
        cpu = [so(r, f"cpu_{ten}") for r in rows if r.get(f"cpu_{ten}")]
        ram = [so(r, f"ram_{ten}") for r in rows if r.get(f"ram_{ten}")]
        if cpu:
            print(f"{ten:<12} CPU trung bình {sum(cpu) / len(cpu):>6.1f}% (đỉnh {max(cpu):.1f}%)"
                  f"   RAM {sum(ram) / len(ram):>6.0f} MB (đỉnh {max(ram):.0f} MB)")

    co_module = any(so(r, "module_song") for r in rows)
    if co_module:
        cpu = [so(r, "cpu_module") for r in rows]
        vram = [so(r, "vram") for r in rows]
        print(f"{'module':<12} CPU trung bình {sum(cpu) / len(cpu):>6.1f}%"
              f"            VRAM cả card {sum(vram) / len(vram):>6.0f} MB (đỉnh {max(vram):.0f} MB)")

    # RAM có tăng dần không: so trung bình 10% số dòng đầu với 10% cuối (bỏ 10 phút khởi động).
    print()
    print("RAM đầu buổi → cuối buổi (trung bình 10% số lần đo đầu / cuối):")
    bo = min(10, len(rows) // 10)
    phan = max(1, (len(rows) - bo) // 10)
    tang_nhieu = False
    for cot, ten in (("ram_frigate", "frigate"), ("ram_module", "module"), ("ram_cau_noi", "cầu nối"),
                     ("vram", "VRAM")):
        gia_tri = [so(r, cot) for r in rows[bo:] if r.get(cot) not in (None, "", "0")]
        if len(gia_tri) < 2:
            continue
        dau = sum(gia_tri[:phan]) / len(gia_tri[:phan])
        cuoi = sum(gia_tri[-phan:]) / len(gia_tri[-phan:])
        doi = (cuoi - dau) / dau if dau else 0
        if doi > 0.2:
            tang_nhieu = True
        print(f"  {ten:<9} {dau:>7.0f} MB → {cuoi:>7.0f} MB  ({doi:+.1%})")

    # Tự nối lại: sau mỗi lần mất luồng (tự nhiên hoặc cố ý ngắt) FPS camera chính phải về lại.
    cam_chinh = "xuong_han" if "xuong_han" in cameras else (cameras[0] if cameras else None)
    lan_ngat = [i for i, r in enumerate(rows) if so(r, "thu_ngat") or so(r, "mat_luong")]
    khong_hoi = []
    for i in lan_ngat:
        sau = rows[i + 1:i + 6]  # 5 lần đo kế tiếp
        if cam_chinh and sau and not any(so(r, f"fps_{cam_chinh}") >= 1 for r in sau):
            khong_hoi.append(rows[i]["luc"])
    print()
    print(f"Lần cố ý ngắt camera : {sum(int(so(r, 'thu_ngat')) for r in rows)}")
    print(f"Lần mất luồng / ngắt : {len(lan_ngat)}, không tự nối lại trong 5 lần đo: {len(khong_hoi)}"
          + (f" ({', '.join(khong_hoi)})" if khong_hoi else ""))
    if co_module:
        tat = [r["luc"] for r in rows if not so(r, "module_song")]
        print(f"Module có lúc không chạy: {len(tat)} lần đo" + (f" (đầu tiên {tat[0]})" if tat else ""))
    else:
        tat = []

    dat = gio >= 8 and song == len(rows) and not khong_hoi and not tat
    print()
    print("KẾT LUẬN:", "ĐẠT tiêu chí chạy liên tục ≥ 8 giờ, tự nối lại khi mất luồng" if dat
          else "CHƯA đạt — xem lại thời lượng, lần Frigate không trả lời, lần không tự nối lại")
    if tang_nhieu:
        print("Lưu ý: có thành phần RAM cuối buổi cao hơn đầu buổi trên 20% — nghi rò bộ nhớ.")


def main() -> int:
    p = argparse.ArgumentParser(description="Theo dõi hệ thống khi chạy dài")
    p.add_argument("--gio", type=float, default=8.0)
    p.add_argument("--moi", type=int, default=60, help="bao nhiêu giây đo một lần")
    p.add_argument("--db", default="data/incidents.db")
    p.add_argument("--ra", default=None)
    p.add_argument("--tong-ket", default=None, help="chỉ đọc lại tệp CSV cũ rồi in tổng kết")
    p.add_argument("--thu-ngat-moi", type=float, default=0, metavar="GIO",
                   help="cứ bấy nhiêu giờ thì khởi động lại rtsp-sim một lần để thử tự nối lại (0 = không)")
    args = p.parse_args()

    if args.tong_ket:
        tong_ket(Path(args.tong_ket))
        return 0

    ra = Path(args.ra or f"data/on_dinh_{datetime.now():%Y-%m-%d_%H%M}.csv")
    ra.parent.mkdir(parents=True, exist_ok=True)
    db = Path(args.db)

    dau = mot_dong(db, args.moi)
    if not dau["frigate_song"]:
        print("Frigate chưa chạy — bật 'docker compose up -d' trước đã.")
        return 1

    het = time.time() + args.gio * 3600
    lan_ngat_toi = time.time() + args.thu_ngat_moi * 3600 if args.thu_ngat_moi > 0 else None
    print(f"Đo mỗi {args.moi}s trong {args.gio} giờ, ghi vào {ra}. Ctrl+C để dừng sớm.")
    if lan_ngat_toi:
        print(f"Cứ {args.thu_ngat_moi} giờ sẽ khởi động lại rtsp-sim một lần để thử tự nối lại.")
    if not dau["module_song"]:
        print("Chú ý: không thấy tiến trình safety.cli — đang đo Frigate thuần.")
    with ra.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(dau))
        w.writeheader()
        w.writerow(dau)
        f.flush()
        try:
            while time.time() < het:
                time.sleep(args.moi)
                vua_ngat = False
                if lan_ngat_toi and time.time() >= lan_ngat_toi:
                    print(f"  {datetime.now():%H:%M:%S}  thử ngắt: khởi động lại rtsp-sim")
                    chay(["docker", "restart", "rtsp-sim"], timeout=120)
                    lan_ngat_toi += args.thu_ngat_moi * 3600
                    vua_ngat = True
                dong = mot_dong(db, args.moi)
                dong["thu_ngat"] = int(vua_ngat)
                if not dong["module_song"] and dau["module_song"]:
                    print(f"  {dong['luc']}  module KHÔNG còn chạy")
                # Camera mới xuất hiện giữa chừng thì bỏ qua cột lạ, không làm hỏng tệp.
                w.writerow({k: v for k, v in dong.items() if k in w.fieldnames})
                f.flush()
                if not dong["frigate_song"]:
                    print(f"  {dong['luc']}  Frigate KHÔNG trả lời API")
                elif dong["mat_luong"]:
                    print(f"  {dong['luc']}  mất luồng {dong['mat_luong']} lần")
        except KeyboardInterrupt:
            print("Dừng sớm theo yêu cầu.")

    print()
    tong_ket(ra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
