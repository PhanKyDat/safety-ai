"""Sinh sự cố giả vào một DB riêng, để dựng và thử trang thống kê.

    python scripts/seed_dashboard_demo.py --db data/incidents_demo.db --days 14

Khác `demo_incident_flow.py` của Đạt: script đó chạy cả đường thật (model, luật,
Frigate) nên cần torch + ultralytics; script này chỉ ghi thẳng vào bảng incidents
bằng đúng contract chung, nên máy chỉ làm giao diện không phải cài cả bộ AI.

Dữ liệu là giả, chỉ dùng để dựng giao diện — đừng trỏ trang thống kê vào DB thật
rồi seed đè lên.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from safety.contracts import DEFAULT_SEVERITY, Incident, IncidentLabel
from safety.db import init_db
from safety.mqtt_bridge import ghi_db

CAMERAS = ("xuong_han", "kho_1", "xuong_son")

# Tần suất thật lệch nhau nhiều: thiếu đồ bảo hộ là thường ngày, cháy và ngã hiếm.
TRONG_SO = {
    IncidentLabel.NO_HELMET: 40,
    IncidentLabel.NO_VEST: 25,
    IncidentLabel.DANGER_ZONE: 20,
    IncidentLabel.SMOKE: 8,
    IncidentLabel.FIRE: 4,
    IncidentLabel.FALL: 3,
}

MO_TA = {
    IncidentLabel.NO_HELMET: "ID {id} - khu han",
    IncidentLabel.NO_VEST: "ID {id} - loi di",
    IncidentLabel.DANGER_ZONE: "Frigate: vung_nguy_hiem",
    IncidentLabel.SMOKE: "2 hop khoi",
    IncidentLabel.FIRE: "1 hop lua",
    IncidentLabel.FALL: "ID {id} - nam {giay}s",
}


def sinh(so_luong: int, so_ngay: int, seed: int) -> list[Incident]:
    rng = random.Random(seed)
    nhan = list(TRONG_SO)
    trong_so = [TRONG_SO[n] for n in nhan]
    bay_gio = time.time()
    incidents = []

    for _ in range(so_luong):
        label = rng.choices(nhan, weights=trong_so, k=1)[0]
        # Giờ làm việc 7–17h cho giống thật, nhờ vậy biểu đồ theo giờ có hình dạng.
        lui = rng.uniform(0, so_ngay * 86400)
        ts = bay_gio - lui
        gio = time.localtime(ts).tm_hour
        if not 7 <= gio <= 17:
            ts -= (gio - 12) * 3600
        track_id = rng.randint(1, 60)
        incidents.append(
            Incident(
                camera=rng.choice(CAMERAS),
                label=label,
                severity=DEFAULT_SEVERITY[label],
                score=round(rng.uniform(0.62, 0.97), 2),
                ts=ts,
                sub_label=MO_TA[label].format(id=track_id, giay=rng.randint(10, 40)),
                snapshot=None,
                track_id=None if label in (IncidentLabel.FIRE, IncidentLabel.SMOKE) else track_id,
                frigate_event_id=f"{ts:.6f}-demo{rng.randrange(16**6):06x}",
            )
        )
    return incidents


def main() -> int:
    p = argparse.ArgumentParser(description="Sinh sự cố giả cho trang thống kê")
    p.add_argument("--db", default="data/incidents_demo.db")
    p.add_argument("--so-luong", type=int, default=180)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--seed", type=int, default=20261026)
    args = p.parse_args()

    db = init_db(args.db)
    incidents = sinh(args.so_luong, args.days, args.seed)
    for inc in incidents:
        ghi_db(db, inc)

    # Console Windows mac dinh cp1252, in tieng Viet co dau se loi -> giu ASCII.
    print(f"Da ghi {len(incidents)} su co gia vao {Path(db)}")
    print("Chay trang thong ke:")
    print(f"  .venv\\Scripts\\streamlit.exe run dashboard/app.py -- --db {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
