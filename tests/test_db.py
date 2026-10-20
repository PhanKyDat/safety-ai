"""Kiểm thử ghi sự cố vào bảng incidents: đúng thứ tự, chung bảng với cầu nối MQTT.

Dùng SQLite trong thư mục tạm, không đụng data/incidents.db thật.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import sqlite3

import dotenv

import safety.db as db_mod
from safety.alerts import TelegramPublisher
from safety.cli import db_path
from safety.contracts import Incident, IncidentLabel, Severity
from safety.db import DBPublisher, init_db
from safety.frigate import FrigatePublisher
from safety.mqtt_bridge import ghi_db
from safety.outputs import Outputs, build_outputs


def incident(label=IncidentLabel.NO_HELMET, severity=Severity.HIGH, track_id=7) -> Incident:
    return Incident(camera="xuong_han", label=label, severity=severity, score=0.87,
                    ts=1789012348.1, sub_label=f"ID {track_id}", track_id=track_id,
                    snapshot=f"snapshots/xuong_han_{label.value}_1789012348_id{track_id}.jpg")


class FakeClient:
    def __init__(self, ok=True):
        self.ok = ok

    def create_event(self, inc):
        return f"evt-{inc.track_id}" if self.ok else None


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, inc):
        self.sent.append(inc.track_id)
        return True


def outputs(db_file, client=None, notifier=None, when=None) -> Outputs:
    tele = TelegramPublisher(notifier) if notifier is not None else None
    out = Outputs(telegram=tele, telegram_when=when, db=DBPublisher(db_file))
    if client is not None:
        out.frigate = FrigatePublisher(client, on_done=out.after_frigate)
    return out


def rows(db_file, sql="SELECT label, track_id, frigate_event_id, snapshot, false_alarm FROM incidents ORDER BY id"):
    con = sqlite3.connect(db_file)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_ghi_sau_frigate_nen_dong_trong_db_co_ma_su_co(tmp_path):
    """Trang thống kê cần mã Frigate ở mỗi dòng để làm nút mở sự cố."""
    f = tmp_path / "incidents.db"
    out = outputs(f, FakeClient())
    out.submit(incident(track_id=1))
    out.close()
    assert rows(f) == [("no_helmet", 1, "evt-1",
                        "snapshots/xuong_han_no_helmet_1789012348_id1.jpg", 0)]
    assert out.db.sent == 1


def test_frigate_sap_van_ghi_db(tmp_path):
    f = tmp_path / "incidents.db"
    out = outputs(f, FakeClient(ok=False))
    out.submit(incident(track_id=1))
    out.close()
    assert rows(f) == [("no_helmet", 1, None, "snapshots/xuong_han_no_helmet_1789012348_id1.jpg", 0)]


def test_khong_co_frigate_thi_ghi_thang_db(tmp_path):
    f = tmp_path / "incidents.db"
    out = outputs(f)
    assert out.submit(incident(track_id=2)) is True
    out.close()
    assert [r[1] for r in rows(f)] == [2]


def test_loc_telegram_khong_lam_mat_dong_trong_db(tmp_path):
    """Thống kê phải đủ mọi sự cố, kể cả loại không gửi Telegram."""
    f = tmp_path / "incidents.db"
    n = FakeNotifier()
    out = outputs(f, FakeClient(), n, when=lambda inc: inc.severity is Severity.CRITICAL)
    out.submit(incident(IncidentLabel.NO_HELMET, Severity.HIGH, track_id=1))
    out.submit(incident(IncidentLabel.FIRE, Severity.CRITICAL, track_id=2))
    out.close()
    assert n.sent == [2]
    assert sorted(r[1] for r in rows(f)) == [1, 2]


def test_chung_bang_voi_cau_noi_mqtt(tmp_path):
    """Cầu nối của Uy tạo bảng trước (không có chỉ mục), module ghi sau: vẫn một bảng, đủ chỉ mục."""
    f = tmp_path / "incidents.db"
    ghi_db(f, Incident(camera="xuong_han", label=IncidentLabel.DANGER_ZONE, severity=Severity.HIGH,
                       score=1.0, ts=1789012300.0, sub_label="Frigate: vung_nguy_hiem",
                       frigate_event_id="ev1"))
    out = outputs(f, FakeClient())
    out.submit(incident(IncidentLabel.FIRE, Severity.CRITICAL, track_id=None))
    out.close()
    assert [(r[0], r[2]) for r in rows(f)] == [("danger_zone", "ev1"), ("fire", "evt-None")]
    indexes = {r[0] for r in rows(f, "SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_incidents_ts", "idx_incidents_camera_label"} <= indexes


def test_db_loi_thi_dem_vao_failed_va_telegram_van_gui(tmp_path, monkeypatch):
    def hong(*_):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_mod, "ghi_db", hong)
    n = FakeNotifier()
    out = outputs(tmp_path / "incidents.db", FakeClient(), n)
    out.submit(incident(track_id=1))
    out.close()
    assert out.db.failed == 1 and n.sent == [1]
    assert "DB: 0 thành công, 1 lỗi" in out.summary()


def test_khong_mo_duoc_db_thi_van_chay_khong_ghi(tmp_path):
    chan = tmp_path / "la_tep"
    chan.write_text("x")
    out = build_outputs(None, "xuong_han", db_path=str(chan / "incidents.db"))
    assert out.db is None
    out.close()


def test_init_db_goi_lai_nhieu_lan_khong_sao(tmp_path):
    f = tmp_path / "con" / "incidents.db"
    init_db(f)
    init_db(f)
    assert rows(f, "PRAGMA journal_mode") == [("wal",)]


def test_tuy_chon_db_tren_dong_lenh(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    assert db_path(None) is None                                  # không có --db: không ghi
    assert db_path("x.db") == "x.db"                              # --db x.db
    monkeypatch.setenv("INCIDENT_DB", "data/khac.db")
    assert db_path("") == "data/khac.db"                          # --db: theo .env, chung tệp với cầu nối
    monkeypatch.delenv("INCIDENT_DB")
    assert db_path("") == "data/incidents.db"
