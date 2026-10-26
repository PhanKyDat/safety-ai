"""Bài B1: đối chiếu "không đội mũ" của Frigate với sự cố của module.

    python scripts/b1_doi_chung.py --gio 2
    python scripts/b1_doi_chung.py --gio 2 --db data/incidents.db --api http://127.0.0.1:5050/api

Frigate gắn thuộc tính `mu_bao_ho` lên từng người nó bám vết (xem docs/doi-chung-b1.md);
module ghi sự cố `no_helmet` vào bảng incidents sau khi qua luật thời gian. Script lấy
cả hai trong cùng một khoảng, rồi ghép theo camera + thời gian để biết hai bên đồng ý
bao nhiêu phần.

Script chỉ đếm và ghép, không chấm điểm: precision/recall phải so với nhãn thời điểm
sự cố của bộ video kiểm thử, làm ở tuần 9.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Console Windows mac dinh cp1252, in tieng Viet co dau se nem UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from safety.stats import gio_dia_phuong  # noqa: E402

MODEL = "mu_bao_ho"
LOP_KHONG_MU = "khong_mu"


def lay_su_kien_frigate(api: str, tu_ts: float, limit: int = 500) -> list[dict]:
    """Sự kiện person của Frigate trong khoảng, kèm thuộc tính phân loại."""
    url = f"{api.rstrip('/')}/events?after={tu_ts:.0f}&labels=person&limit={limit}&include_attributes=1"
    try:
        with urllib.request.urlopen(url, timeout=10) as f:
            return json.load(f)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Không gọi được {url}: {e}\n"
            "Cổng 5000 không mở ra host — xem docs/canh-bao-telegram.md mục 4 "
            "(docker-compose.override.yml), hoặc đổi --api."
        )


def nhan_phan_loai(event: dict) -> str | None:
    """Giá trị thuộc tính mu_bao_ho của một sự kiện, nếu Frigate đã gán."""
    data = event.get("data") or {}
    # Frigate để thuộc tính ở data.attributes (dạng {ten: gia_tri}) hoặc sub_label.
    thuoc_tinh = data.get("attributes") or {}
    if isinstance(thuoc_tinh, dict) and MODEL in thuoc_tinh:
        return str(thuoc_tinh[MODEL])
    for t in event.get("attributes") or []:
        if isinstance(t, dict) and t.get("name") == MODEL:
            return str(t.get("value"))
    return None


def lay_su_co_module(db: Path, tu_ts: float) -> list[dict]:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT camera, ts, sub_label, score, false_alarm FROM incidents"
            " WHERE label = 'no_helmet' AND ts >= ? ORDER BY ts",
            (tu_ts,),
        )
        return [dict(r) for r in rows]
    finally:
        con.close()


def ghep(frigate: list[dict], module: list[dict], dung_sai: float) -> int:
    """Số sự cố của module có một sự kiện 'không mũ' của Frigate cùng camera, lệch < dung_sai giây."""
    trung = 0
    for su_co in module:
        for ev in frigate:
            if ev.get("camera") != su_co["camera"]:
                continue
            bat_dau = ev.get("start_time") or 0
            ket_thuc = ev.get("end_time") or bat_dau
            if bat_dau - dung_sai <= su_co["ts"] <= ket_thuc + dung_sai:
                trung += 1
                break
    return trung


def main() -> int:
    p = argparse.ArgumentParser(description="Đối chiếu B1: Frigate vs module")
    p.add_argument("--gio", type=float, default=2.0, help="xét bao nhiêu giờ gần đây")
    p.add_argument("--db", default="data/incidents.db")
    p.add_argument("--api", default="http://127.0.0.1:5050/api")
    p.add_argument("--dung-sai", type=float, default=10.0, help="lệch bao nhiêu giây vẫn coi là trùng")
    args = p.parse_args()

    tu_ts = time.time() - args.gio * 3600
    db = Path(args.db)
    if not db.is_file():
        raise SystemExit(f"Không thấy {db}")

    su_kien = lay_su_kien_frigate(args.api, tu_ts)
    da_gan = [(e, nhan_phan_loai(e)) for e in su_kien]
    khong_mu = [e for e, n in da_gan if n == LOP_KHONG_MU]
    co_nhan = [e for e, n in da_gan if n is not None]
    su_co = lay_su_co_module(db, tu_ts)
    that = [s for s in su_co if not s["false_alarm"]]

    print(f"Khoảng xét: từ {gio_dia_phuong(tu_ts)} đến nay ({args.gio} giờ)")
    print()
    print("FRIGATE (Object Classification)")
    print(f"  Người được bám vết          : {len(su_kien)}")
    print(f"  Đã gán nhãn {MODEL:<16}: {len(co_nhan)}")
    print(f"  Trong đó '{LOP_KHONG_MU}'          : {len(khong_mu)}")
    if su_kien and not co_nhan:
        print("  (chưa gán nhãn lần nào — model đã huấn luyện chưa? xem docs/doi-chung-b1.md)")
    print()
    print("MODULE (YOLO + luật thời gian)")
    print(f"  Sự cố no_helmet             : {len(su_co)}")
    print(f"  Sau khi trừ báo nhầm        : {len(that)}")
    print()
    trung = ghep(khong_mu, that, args.dung_sai)
    print(f"GHÉP (lệch < {args.dung_sai:.0f}s, cùng camera)")
    print(f"  Sự cố module khớp với Frigate: {trung}/{len(that)}")
    print(f"  Module bắt, Frigate không    : {len(that) - trung}")
    print()
    print("Precision/recall phải so với nhãn thời điểm của bộ video kiểm thử (tuần 9).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
