"""Kiểm thử lớp đọc/ghi của trang thống kê bằng SQLite tạm.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import sqlite3
import time

import pytest

from safety.stats import (
    Filter,
    danh_dau_bao_nham,
    danh_sach,
    dem_theo,
    dem_theo_ngay,
    gia_tri_loc,
    link_frigate,
    tong_quan,
)

NGAY = 86400


@pytest.fixture
def db(tmp_path):
    """DB tạm: 3 sự cố hôm nay, 1 sự cố 5 ngày trước, 1 cái đã đánh dấu báo nhầm."""
    path = tmp_path / "incidents.db"
    bay_gio = time.time()
    rows = [
        ("xuong_han", "no_helmet", "ID 7", "high", 0.9, bay_gio - 60, "ev1", 0),
        ("xuong_han", "fire", "1 hop lua", "critical", 0.8, bay_gio - 120, "ev2", 0),
        ("kho_1", "danger_zone", "Frigate", "high", 1.0, bay_gio - 180, "ev3", 1),
        ("kho_1", "no_vest", "ID 9", "medium", 0.7, bay_gio - 5 * NGAY, "ev4", 0),
    ]
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE incidents (
               id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT NOT NULL, label TEXT NOT NULL,
               sub_label TEXT, severity TEXT NOT NULL, score REAL NOT NULL, ts REAL NOT NULL,
               snapshot TEXT, track_id INTEGER, frigate_event_id TEXT,
               false_alarm INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL DEFAULT 0)"""
    )
    con.executemany(
        """INSERT INTO incidents (camera, label, sub_label, severity, score, ts, frigate_event_id, false_alarm)
           VALUES (?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()
    con.close()
    return path


def test_danh_sach_moi_nhat_truoc(db):
    rows = danh_sach(db)
    assert len(rows) == 4
    assert [r["label"] for r in rows] == ["no_helmet", "fire", "danger_zone", "no_vest"]


def test_loc_theo_camera_va_nhan(db):
    assert len(danh_sach(db, Filter(cameras=("kho_1",)))) == 2
    assert len(danh_sach(db, Filter(labels=("fire", "no_helmet")))) == 2
    assert len(danh_sach(db, Filter(cameras=("kho_1",), labels=("no_vest",)))) == 1


def test_loc_theo_khoang_thoi_gian(db):
    gan_day = Filter(tu_ts=time.time() - NGAY)
    assert len(danh_sach(db, gan_day)) == 3


def test_an_bao_nham(db):
    assert len(danh_sach(db, Filter(an_bao_nham=True))) == 3


def test_tong_quan_tinh_ti_le_bao_nham(db):
    tq = tong_quan(db)
    assert tq["tong"] == 4
    assert tq["bao_nham"] == 1
    assert tq["nghiem_trong"] == 1
    assert tq["ti_le_bao_nham"] == pytest.approx(0.25)


def test_tong_quan_db_rong_khong_chia_cho_khong(tmp_path):
    path = tmp_path / "rong.db"
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE incidents (id INTEGER PRIMARY KEY, camera TEXT, label TEXT,
           sub_label TEXT, severity TEXT, score REAL, ts REAL, snapshot TEXT,
           track_id INTEGER, frigate_event_id TEXT, false_alarm INTEGER DEFAULT 0,
           created_at REAL DEFAULT 0)"""
    )
    con.commit()
    con.close()
    assert tong_quan(path)["ti_le_bao_nham"] == 0.0


def test_dem_theo_nhan_va_camera(db):
    assert dem_theo(db, "label")["no_helmet"] == 1
    assert dem_theo(db, "camera") == {"xuong_han": 2, "kho_1": 2}
    assert dem_theo(db, "camera", Filter(labels=("fire",))) == {"xuong_han": 1}


def test_dem_theo_cot_la_bi_tu_choi(db):
    with pytest.raises(ValueError):
        dem_theo(db, "snapshot; DROP TABLE incidents")


def test_dem_theo_ngay(db):
    theo_ngay = dem_theo_ngay(db)
    assert sum(theo_ngay.values()) == 4
    assert len(theo_ngay) == 2  # hom nay va 5 ngay truoc


def test_danh_dau_bao_nham_va_bo_danh_dau(db):
    row = danh_sach(db)[0]
    assert row["false_alarm"] == 0
    assert danh_dau_bao_nham(db, row["id"]) is True
    assert danh_sach(db)[0]["false_alarm"] == 1
    assert danh_dau_bao_nham(db, row["id"], False) is True
    assert danh_sach(db)[0]["false_alarm"] == 0


def test_danh_dau_id_khong_co_thi_bao_false(db):
    assert danh_dau_bao_nham(db, 9999) is False


def test_gia_tri_loc(db):
    gt = gia_tri_loc(db)
    assert gt["cameras"] == ["kho_1", "xuong_han"]
    assert "danger_zone" in gt["labels"]


def test_khong_ghi_duoc_khi_chi_doc(db):
    # danh_sach mo che do ro; neu lo mo ghi thi test nay khong bat duoc loi gi,
    # nen kiem truc tiep: ket noi ro phai tu choi UPDATE.
    import sqlite3 as s

    con = s.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    with pytest.raises(s.OperationalError):
        con.execute("UPDATE incidents SET false_alarm = 1")
    con.close()


def test_link_frigate_dung_trang_explore():
    assert link_frigate("https://localhost:8971", "ev1") == "https://localhost:8971/explore?event_id=ev1"
    # Gốc URL có dấu / thừa thì không được sinh ra //explore
    assert link_frigate("https://localhost:8971/", "ev1") == "https://localhost:8971/explore?event_id=ev1"
    assert link_frigate(None, "ev1") is None
    assert link_frigate("https://localhost:8971", None) is None
