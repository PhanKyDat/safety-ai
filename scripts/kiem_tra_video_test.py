"""Soi bộ video kiểm thử trước khi đem đi đo (docs/bo-video-kiem-thu.md).

    python scripts/kiem_tra_video_test.py
    python scripts/kiem_tra_video_test.py --thu-muc data/videos_test --nhan

Kiểm ba thứ hay làm hỏng phép đo:

* **Tên tệp** sai quy ước thì không biết đoạn đó quay cảnh gì.
* **Độ phân giải / độ dài** không đạt thì đo xong cũng không dùng được.
* **Máy quay rung** là Frigate hiệu chỉnh chuyển động mãi và không phát hiện gì —
  đã gặp thật ở tuần 1, nên đo luôn độ lệch giữa khung đầu và các khung sau.

Dùng OpenCV, không cần ffmpeg ngoài.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MA_CANH = (
    "binh_thuong", "khong_mu", "khong_ao", "nga", "vung_nguy_hiem",
    "bay_ao_cam", "bay_mu_luoi_trai", "bay_den_vang", "lua", "khoi",
)
TEN_HOP_LE = re.compile(rf"^({'|'.join(MA_CANH)})_\d{{2}}(_(ngay|dem))?\.(mp4|mov|mkv)$")

# Nhan su co theo docs/data-contract.md, de goi y khi dien nhan.csv
NHAN_GOI_Y = {
    "khong_mu": "no_helmet",
    "khong_ao": "no_vest",
    "nga": "fall",
    "vung_nguy_hiem": "danger_zone",
    "lua": "fire",
    "khoi": "smoke",
}

RONG_TOI_THIEU = 1280
CAO_TOI_THIEU = 720
DAI_TOI_THIEU = 20.0
DAI_TOI_DA = 90.0


RONG_DO = 320  # thu nhỏ về chiều rộng này rồi mới đo, cho nhanh và đỡ nhiễu


def do_rung(cap: cv2.VideoCapture, buoc_giay: float = 0.5) -> float:
    """Máy quay trôi bao nhiêu điểm ảnh mỗi nửa giây (0 = đứng yên hẳn).

    Đo bằng phase correlation giữa **hai khung liền kề**, lấy trung vị:

    * So theo từng điểm ảnh thì người đi qua cũng làm số đo vọt lên — không
      phân biệt được camera rung với camera đứng yên mà đông người.
    * So khung đầu với khung cuối thì sau mươi giây nội dung đã khác hẳn, phép
      đo mất tin cậy (thử thực tế: độ tin cậy rớt còn 0,25 và cho số vô nghĩa).

    Đo trên video thật của nhóm: quay tay 1,55 px — đã chống rung 0,08 px.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tong_khung = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    buoc = max(1, int(fps * buoc_giay))

    truoc, cua_so, dich = None, None, []
    for i in range(0, tong_khung, buoc):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        xam = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        cao = max(1, int(xam.shape[0] * RONG_DO / xam.shape[1]))
        xam = cv2.resize(xam, (RONG_DO, cao)).astype(np.float32)
        if cua_so is None:
            cua_so = cv2.createHanningWindow((xam.shape[1], xam.shape[0]), cv2.CV_32F)
        if truoc is not None and truoc.shape == xam.shape:
            (dx, dy), tin_cay = cv2.phaseCorrelate(truoc, xam, cua_so)
            # Bỏ mẫu kém tin cậy (cảnh đổi đột ngột, video lặp vòng).
            if tin_cay > 0.3:
                dich.append(float(np.hypot(dx, dy)))
        truoc = xam
    return float(np.median(dich)) if dich else 0.0


def soi(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"tep": path.name, "loi": ["không mở được tệp"]}
    try:
        rong = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cao = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        so_khung = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dai = so_khung / fps if fps else 0.0
        rung = do_rung(cap)
    finally:
        cap.release()

    loi, nhac = [], []
    if not TEN_HOP_LE.match(path.name):
        loi.append("tên không đúng quy ước <ma_canh>_<so>_<ngay|dem>.mp4")
    if rong < RONG_TOI_THIEU or cao < CAO_TOI_THIEU:
        loi.append(f"{rong}x{cao} nhỏ hơn {RONG_TOI_THIEU}x{CAO_TOI_THIEU}")
    if dai < DAI_TOI_THIEU:
        loi.append(f"chỉ {dai:.0f}s, cần ≥ {DAI_TOI_THIEU:.0f}s")
    elif dai > DAI_TOI_DA:
        nhac.append(f"{dai:.0f}s hơi dài, cắt ngắn cho dễ gán nhãn")
    if fps < 20:
        nhac.append(f"{fps:.0f} fps hơi thấp")
    # Ngưỡng đặt giữa hai mốc đo thật: 0,08 px (chống rung) và 1,55 px (quay tay).
    if rung > 0.8:
        loi.append(f"máy quay trôi {rung:.2f} px/0,5s — Frigate sẽ không detect")
    elif rung > 0.3:
        nhac.append(f"hơi rung ({rung:.2f} px/0,5s)")

    return {
        "tep": path.name, "rong": rong, "cao": cao, "fps": fps,
        "dai": dai, "rung": rung, "loi": loi, "nhac": nhac,
    }


def tao_mau_nhan(thu_muc: Path, ket_qua: list[dict]) -> Path:
    """Tạo khung mẫu nhan.csv, mỗi video một dòng, gợi ý sẵn nhãn theo tên tệp."""
    dich = thu_muc / "nhan.csv"
    if dich.exists():
        return dich
    with dich.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tep", "nhan", "bat_dau_giay", "ket_thuc_giay", "ghi_chu"])
        for r in ket_qua:
            ma = r["tep"].split("_0")[0]
            w.writerow([r["tep"], NHAN_GOI_Y.get(ma, ""), "", "", ""])
    return dich


def main() -> int:
    p = argparse.ArgumentParser(description="Kiểm tra bộ video kiểm thử")
    p.add_argument("--thu-muc", default="data/videos_test")
    p.add_argument("--nhan", action="store_true", help="tạo khung mẫu nhan.csv")
    args = p.parse_args()

    thu_muc = Path(args.thu_muc)
    if not thu_muc.is_dir():
        print(f"Chưa có thư mục {thu_muc}. Tạo rồi chép video vào — xem docs/bo-video-kiem-thu.md")
        return 1

    video = sorted(f for f in thu_muc.iterdir() if f.suffix.lower() in {".mp4", ".mov", ".mkv"})
    if not video:
        print(f"Không thấy video nào trong {thu_muc}")
        return 1

    ket_qua = [soi(f) for f in video]
    hong = 0
    print(f"{'Tệp':<34}{'Khung hình':<13}{'Dài':<8}{'Rung':<7}Kết quả")
    for r in ket_qua:
        if r.get("loi") and "rong" not in r:
            print(f"{r['tep']:<34}{'':<28}LỖI: {r['loi'][0]}")
            hong += 1
            continue
        trang_thai = "ĐẠT" if not r["loi"] else "LỖI"
        hong += bool(r["loi"])
        print(
            f"{r['tep']:<34}{r['rong']}x{r['cao']:<8}{r['dai']:.0f}s{'':<4}"
            f"{r['rung']:.1f}{'':<3}{trang_thai}"
        )
        for m in r["loi"]:
            print(f"    - {m}")
        for m in r["nhac"]:
            print(f"    ~ {m}")

    print()
    print(f"{len(ket_qua) - hong}/{len(ket_qua)} đoạn đạt.")

    if args.nhan:
        dich = tao_mau_nhan(thu_muc, ket_qua)
        print(f"Khung mẫu nhãn: {dich} (mở bằng Excel, điền thời điểm sự cố)")

    thieu = [m for m in MA_CANH if not any(r["tep"].startswith(m) for r in ket_qua)]
    if thieu:
        print("Chưa có cảnh: " + ", ".join(thieu))
    return 0 if hong == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
