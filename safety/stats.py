"""Đọc và cập nhật bảng incidents cho trang thống kê.

Tách khỏi giao diện Streamlit để kiểm thử được bằng SQLite tạm, không cần mở trình
duyệt. Giao diện chỉ gọi mấy hàm ở đây.

Mở DB ở chế độ chỉ đọc khi truy vấn: module và cầu nối MQTT đang ghi liên tục, trang
thống kê không được giữ khoá ghi. Đánh dấu báo nhầm thì mới mở ghi, và ghi rất ngắn.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# Nhãn sự cố -> tên tiếng Việt. Giữ cùng cách gọi với tin Telegram (safety/notify.py).
LABEL_VI = {
    "no_helmet": "Không đội mũ bảo hộ",
    "no_vest": "Không mặc áo phản quang",
    "fire": "Phát hiện lửa",
    "smoke": "Phát hiện khói",
    "fall": "Người ngã",
    "danger_zone": "Người vào vùng nguy hiểm",
}

SEVERITY_VI = {"critical": "Nghiêm trọng", "high": "Cao", "medium": "Trung bình"}

COLUMNS = (
    "id", "camera", "label", "sub_label", "severity", "score", "ts",
    "snapshot", "track_id", "frigate_event_id", "false_alarm", "created_at",
)


@dataclass(frozen=True)
class Filter:
    """Bộ lọc của trang thống kê. None nghĩa là không lọc theo tiêu chí đó."""

    tu_ts: float | None = None
    den_ts: float | None = None
    cameras: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    an_bao_nham: bool = False

    def where(self) -> tuple[str, list]:
        dieu_kien, tham_so = [], []
        if self.tu_ts is not None:
            dieu_kien.append("ts >= ?")
            tham_so.append(self.tu_ts)
        if self.den_ts is not None:
            dieu_kien.append("ts < ?")
            tham_so.append(self.den_ts)
        for cot, gia_tri in (("camera", self.cameras), ("label", self.labels)):
            if gia_tri:
                dieu_kien.append(f"{cot} IN ({','.join('?' * len(gia_tri))})")
                tham_so.extend(gia_tri)
        if self.an_bao_nham:
            dieu_kien.append("false_alarm = 0")
        return (" WHERE " + " AND ".join(dieu_kien) if dieu_kien else ""), tham_so


def _doc(db_path: str | Path) -> sqlite3.Connection:
    """Kết nối chỉ đọc. uri=True để SQLite hiểu mode=ro."""
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def ton_tai(db_path: str | Path) -> bool:
    return Path(db_path).is_file()


def danh_sach(db_path: str | Path, loc: Filter | None = None, gioi_han: int = 500) -> list[dict]:
    """Các sự cố khớp bộ lọc, mới nhất trước."""
    loc = loc or Filter()
    where, tham_so = loc.where()
    sql = f"SELECT {', '.join(COLUMNS)} FROM incidents{where} ORDER BY ts DESC LIMIT ?"
    with _doc(db_path) as con:
        return [dict(r) for r in con.execute(sql, [*tham_so, gioi_han])]


def dem_theo(db_path: str | Path, cot: str, loc: Filter | None = None) -> dict[str, int]:
    """Đếm sự cố theo một cột (label / camera / severity)."""
    if cot not in {"label", "camera", "severity"}:
        raise ValueError(f"Không đếm theo cột {cot}")
    loc = loc or Filter()
    where, tham_so = loc.where()
    with _doc(db_path) as con:
        rows = con.execute(
            f"SELECT {cot} AS k, COUNT(*) AS n FROM incidents{where} GROUP BY {cot} ORDER BY n DESC",
            tham_so,
        )
        return {r["k"]: r["n"] for r in rows}


def dem_theo_ngay(db_path: str | Path, loc: Filter | None = None) -> dict[str, int]:
    """Đếm theo ngày địa phương, để vẽ biểu đồ cột."""
    loc = loc or Filter()
    where, tham_so = loc.where()
    with _doc(db_path) as con:
        rows = con.execute(
            f"SELECT date(ts, 'unixepoch', 'localtime') AS ngay, COUNT(*) AS n"
            f" FROM incidents{where} GROUP BY ngay ORDER BY ngay",
            tham_so,
        )
        return {r["ngay"]: r["n"] for r in rows}


def tong_quan(db_path: str | Path, loc: Filter | None = None) -> dict:
    """Mấy số liệu trên đầu trang."""
    loc = loc or Filter()
    where, tham_so = loc.where()
    with _doc(db_path) as con:
        r = con.execute(
            f"""SELECT COUNT(*) AS tong,
                       COALESCE(SUM(false_alarm), 0) AS bao_nham,
                       COALESCE(SUM(severity = 'critical'), 0) AS nghiem_trong,
                       MAX(ts) AS moi_nhat
                FROM incidents{where}""",
            tham_so,
        ).fetchone()
    tong = r["tong"] or 0
    return {
        "tong": tong,
        "bao_nham": r["bao_nham"] or 0,
        "nghiem_trong": r["nghiem_trong"] or 0,
        "moi_nhat": r["moi_nhat"],
        # Tỉ lệ báo nhầm là con số tiêu chí đánh giá cần, nên tính sẵn ở đây.
        "ti_le_bao_nham": (r["bao_nham"] or 0) / tong if tong else 0.0,
    }


def gia_tri_loc(db_path: str | Path) -> dict[str, list[str]]:
    """Danh sách camera và nhãn đang có trong DB, để dựng ô chọn."""
    with _doc(db_path) as con:
        return {
            "cameras": [r[0] for r in con.execute("SELECT DISTINCT camera FROM incidents ORDER BY 1")],
            "labels": [r[0] for r in con.execute("SELECT DISTINCT label FROM incidents ORDER BY 1")],
        }


def danh_dau_bao_nham(db_path: str | Path, incident_id: int, bao_nham: bool = True) -> bool:
    """Bật/tắt cờ báo nhầm cho một sự cố. Trả về True nếu có dòng được sửa."""
    with sqlite3.connect(db_path, timeout=5) as con:
        cur = con.execute(
            "UPDATE incidents SET false_alarm = ? WHERE id = ?",
            (1 if bao_nham else 0, incident_id),
        )
        return cur.rowcount > 0


def link_frigate(frigate_ui: str | None, frigate_event_id: str | None) -> str | None:
    """Link mở sự cố trong Frigate. 0.17 không có /events, trang xem là /explore."""
    if not frigate_ui or not frigate_event_id:
        return None
    return f"{frigate_ui.rstrip('/')}/explore?event_id={frigate_event_id}"


def gio_dia_phuong(ts: float) -> str:
    return time.strftime("%H:%M:%S %d/%m/%Y", time.localtime(ts))
