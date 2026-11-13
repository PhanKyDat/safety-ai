"""Bài B3: đo độ trễ từ lúc luật báo tới lúc sự cố hiện trong Frigate và tới Telegram.

    python scripts/do_b3.py --lan 10
    python scripts/do_b3.py --lan 10 --khong-telegram      # chỉ đo phần Frigate

Hai đoạn còn thiếu trong docs/doi-chieu.md mục 4 (phần đầu "vi phạm → luật báo" Đạt đã
đo bằng video). Mỗi lần đo:

1. Tạo một sự cố qua API y như module vẫn làm, bấm giờ lúc Frigate trả về `event_id`.
2. Hỏi lại `/events/<id>` cho tới khi Frigate thực sự có sự cố đó (đây mới là lúc
   người trực nhìn thấy được trong giao diện).
3. Gửi tin Telegram đúng hàm module dùng, bấm giờ lúc Telegram nhận.

Sự cố thử ghi rõ "ĐO B3" trong mô tả và dùng nhãn thật để giống hệt đường chạy thường
ngày. Chạy xong nhớ xoá chúng khỏi Frigate nếu không muốn lẫn vào thống kê.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

from safety.contracts import DEFAULT_SEVERITY, Incident, IncidentLabel  # noqa: E402
from safety.frigate import FrigateClient  # noqa: E402
from safety.notify import TelegramNotifier  # noqa: E402

GOC = Path(__file__).resolve().parent.parent


def cho_frigate_co(client: FrigateClient, event_id: str, han: float = 15.0) -> float | None:
    """Hỏi lại tới khi Frigate có sự cố. Trả về số giây chờ, None nếu quá hạn."""
    bat_dau = time.monotonic()
    while time.monotonic() - bat_dau < han:
        try:
            r = client.session.get(f"{client.api}/events/{event_id}", timeout=client.timeout)
            if r.status_code == 200:
                return time.monotonic() - bat_dau
        except Exception:  # noqa: BLE001 - lỗi mạng tạm thời thì thử lại
            pass
        time.sleep(0.2)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Đo độ trễ B3")
    p.add_argument("--lan", type=int, default=10)
    p.add_argument("--camera", default="xuong_han")
    p.add_argument("--nhan", default="no_helmet")
    p.add_argument("--api", default=None, help="mặc định lấy FRIGATE_API trong .env")
    p.add_argument("--khong-telegram", action="store_true")
    p.add_argument("--nghi", type=float, default=3.0, help="nghỉ bao nhiêu giây giữa hai lần")
    args = p.parse_args()

    load_dotenv(GOC / ".env")
    api = args.api or os.environ.get("FRIGATE_API", "http://127.0.0.1:5050/api")
    client = FrigateClient(api=api)

    if client.cameras() is None:
        print(f"Không gọi được Frigate ở {api} — cổng 5000 có mở ra host chưa?")
        return 1

    notifier = None
    if not args.khong_telegram:
        notifier = TelegramNotifier(
            token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            frigate_ui=os.environ.get("FRIGATE_UI"),
        )

    tao, hien, telegram = [], [], []
    print(f"Đo {args.lan} lần trên camera {args.camera}, nhãn {args.nhan}")
    print(f"{'Lần':<5}{'Gọi API (s)':>13}{'Hiện trong Frigate (s)':>24}{'Telegram (s)':>15}")

    for i in range(1, args.lan + 1):
        incident = Incident(
            camera=args.camera,
            label=IncidentLabel(args.nhan),
            severity=DEFAULT_SEVERITY[IncidentLabel(args.nhan)],
            score=0.9,
            ts=time.time(),
            sub_label=f"ĐO B3 lần {i}",
        )

        t0 = time.monotonic()
        event_id = client.create_event(incident, duration=5)
        t_tao = time.monotonic() - t0
        if not event_id:
            print(f"{i:<5}{'lỗi tạo sự cố':>13}")
            continue
        incident.frigate_event_id = event_id

        t_hien = cho_frigate_co(client, event_id)
        t_tele = None
        if notifier:
            t1 = time.monotonic()
            if notifier.send(incident):
                t_tele = time.monotonic() - t1

        tao.append(t_tao)
        if t_hien is not None:
            hien.append(t_tao + t_hien)
        if t_tele is not None:
            telegram.append(t_tele)

        print(f"{i:<5}{t_tao:>13.2f}{(t_tao + t_hien) if t_hien else float('nan'):>24.2f}"
              f"{t_tele if t_tele else float('nan'):>15.2f}")
        time.sleep(args.nghi)

    def tom_tat(ten: str, so: list[float]) -> None:
        if not so:
            print(f"{ten:<34} không có số đo")
            return
        sx = sorted(so)
        p95 = sx[min(len(sx) - 1, int(len(sx) * 0.95))]
        print(f"{ten:<34} trung vị {statistics.median(so):.2f} s | "
              f"nhỏ nhất {min(so):.2f} | lớn nhất {max(so):.2f} | p95 {p95:.2f} | n={len(so)}")

    print()
    tom_tat("Luật báo → API trả event_id", tao)
    tom_tat("Luật báo → sự cố có trong Frigate", hien)
    tom_tat("Luật báo → tin tới Telegram", telegram)
    print()
    print("Mục tiêu mục 9: toàn bộ chuỗi ≤ 5 giây (đã gồm 3 giây luật cố ý chờ).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
