"""Nối các đầu ra của sự cố theo đúng thứ tự: Frigate trước, rồi DB và Telegram.

IncidentProcessor chỉ gọi `submit`. Có Frigate thì sự cố tới Frigate trước, lấy mã
sự cố về rồi mới chuyển sang DB và Telegram, để dòng trong DB và tin nhắn đều có
đường dẫn mở sự cố. Không có Frigate thì sự cố đi thẳng sang DB và Telegram.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Mapping

from safety.alerts import TelegramPublisher, notifier_from_env
from safety.contracts import Incident
from safety.db import DBPublisher
from safety.frigate import FrigateClient, FrigatePublisher

log = logging.getLogger(__name__)


class Outputs:
    def __init__(
        self,
        telegram: TelegramPublisher | None = None,
        telegram_when: Callable[[Incident], bool] | None = None,
        db: DBPublisher | None = None,
    ):
        self.frigate: FrigatePublisher | None = None
        self.db = db
        self.telegram = telegram
        # Chọn sự cố nào được gửi Telegram, ví dụ chỉ mức cao trở lên. None = gửi hết.
        self.telegram_when = telegram_when

    def submit(self, incident: Incident) -> bool:
        if self.frigate is not None:
            return self.frigate.submit(incident)
        return self._fan_out(incident)

    def after_frigate(self, incident: Incident) -> None:
        """Gọi ở luồng Frigate sau mỗi lần gửi, kể cả khi Frigate lỗi."""
        if incident.frigate_event_id:
            log.info("Đã tạo sự cố trong Frigate: /explore?event_id=%s", incident.frigate_event_id)
        self._fan_out(incident)

    def _fan_out(self, incident: Incident) -> bool:
        # DB không lọc: thống kê cần đủ mọi sự cố, kể cả loại không gửi Telegram.
        stored = self.db.submit(incident) if self.db is not None else False
        return self._to_telegram(incident) or stored

    def _to_telegram(self, incident: Incident) -> bool:
        if self.telegram is None:
            return False
        if self.telegram_when is not None and not self.telegram_when(incident):
            return False
        return self.telegram.submit(incident)

    def close(self, timeout: float = 10.0) -> None:
        # Dừng Frigate trước: nó còn đẩy nốt sự cố sang DB và Telegram trong lúc dừng.
        for worker in (self.frigate, self.db, self.telegram):
            if worker is not None:
                worker.close(timeout)

    def summary(self) -> str:
        parts = []
        for worker in (self.frigate, self.db, self.telegram):
            if worker is not None:
                parts.append(f"{worker.what}: {worker.sent} thành công, {worker.failed} lỗi, "
                             f"{worker.dropped} bỏ do hàng đợi đầy, {worker.unsent} chưa gửi kịp lúc dừng")
        return "; ".join(parts) if parts else "không gửi đi đâu"


def build_outputs(
    frigate_api: str | None,
    camera: str,
    telegram: bool = False,
    telegram_when: Callable[[Incident], bool] | None = None,
    env: Mapping[str, str] | None = None,
    db_path: str | None = None,
) -> Outputs:
    """Dựng các đầu ra theo tuỳ chọn. Thiếu cấu hình thì báo rõ và bỏ qua đầu ra đó."""
    tele = None
    if telegram:
        if env is None:
            from dotenv import load_dotenv

            load_dotenv()
        notifier = notifier_from_env(env)
        if notifier is not None:
            name = notifier.check()
            if name is None:
                log.error("Mã bot Telegram không dùng được, không gửi Telegram")
            else:
                log.info("Gửi Telegram qua bot @%s", name)
                tele = TelegramPublisher(notifier)

    db = None
    if db_path:
        try:
            db = DBPublisher(db_path)
            log.info("Ghi sự cố vào %s", db.db_path)
        except (OSError, sqlite3.Error) as e:
            log.error("Không mở được DB %s, không ghi sự cố: %s", db_path, e)

    out = Outputs(telegram=tele, telegram_when=telegram_when, db=db)

    if frigate_api:
        client = FrigateClient(frigate_api)
        cameras = client.cameras()
        if cameras is None:
            log.warning("Chưa gọi được Frigate; vẫn chạy, sự cố sẽ được thử gửi khi có")
        elif camera not in cameras:
            # Sai tên camera thì Frigate từ chối mọi sự cố, nên báo ngay từ đầu.
            log.error("Frigate không có camera '%s' (đang có: %s)", camera, ", ".join(sorted(cameras)))
        out.frigate = FrigatePublisher(client, on_done=out.after_frigate)
    else:
        log.info("Không có --frigate: sự cố không gửi sang Frigate")
    return out
