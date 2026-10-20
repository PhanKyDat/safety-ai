"""Ghi sự cố của module vào bảng incidents, cùng bảng với cảnh báo vùng nguy hiểm.

Bảng thống kê Streamlit đọc thẳng bảng này (docs/data-contract.md mục 3b), nên sự cố
đồ bảo hộ, cháy/khói của module và sự cố vùng nguy hiểm do cầu nối MQTT ghi phải
nằm chung một chỗ, cùng tên cột. Câu INSERT dùng lại `ghi_db` của Uy để hai bên
không lệch nhau.

Ghi ở luồng nền như Frigate và Telegram: SQLite bị trang thống kê giữ khoá thì chỉ
luồng này chờ, vòng lặp camera vẫn chạy.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from safety.contracts import Incident
from safety.mqtt_bridge import ghi_db
from safety.worker import QueueWorker

log = logging.getLogger(__name__)

SCHEMA = Path(__file__).resolve().parent.parent / "schemas" / "incidents.sql"
DEFAULT_DB = "data/incidents.db"


def init_db(db_path: str | Path) -> Path:
    """Tạo bảng và chỉ mục nếu chưa có. Gọi lại nhiều lần không sao."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA.read_text(encoding="utf-8"))
        # WAL: trang thống kê đang đọc thì module và cầu nối vẫn ghi được, không phải chờ.
        con.execute("PRAGMA journal_mode=WAL")
    finally:
        con.close()
    return path


class DBPublisher(QueueWorker):
    what = "DB"

    def __init__(self, db_path: str | Path = DEFAULT_DB, maxsize: int = 100, on_done=None):
        self.db_path = init_db(db_path)
        super().__init__(maxsize=maxsize, on_done=on_done)

    def _handle(self, incident: Incident) -> bool:
        try:
            ghi_db(self.db_path, incident)
        except sqlite3.Error as e:
            log.error("Không ghi được sự cố %s/%s vào %s: %s",
                      incident.camera, incident.label.value, self.db_path, e)
            return False
        return True
