"""Kiểm thử phần chấm điểm trên video có đáp án (F1 theo sự cố, AUROC/AUPRC theo đoạn).

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import pytest

from safety.contracts import BBox, Detection, DetectClass, FrameResult
from safety.evaluation import (
    GtEvent,
    auroc,
    average_precision,
    frame_scores,
    load_gt,
    match_events,
    parse_time,
    window_truth,
)
from safety.rules import PPERule


def test_doc_thoi_gian_giay_va_phut_giay():
    assert parse_time("12.5") == 12.5
    assert parse_time("1:05") == 65.0


def test_doc_dap_an(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("nhan,bat_dau,ket_thuc,ghi_chu\nno_helmet,5,20,ao xanh\nfall,0:32,0:40,\n", encoding="utf-8")
    assert load_gt(p) == [GtEvent("no_helmet", 5, 20, "ao xanh"), GtEvent("fall", 32, 40, "")]


def test_video_khong_vi_pham_chi_co_tieu_de(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("nhan,bat_dau,ket_thuc,ghi_chu\n", encoding="utf-8")
    assert load_gt(p) == []


@pytest.mark.parametrize("row", ["khong_mu,5,20,", "no_helmet,20,5,"])
def test_dap_an_sai_nhan_hoac_sai_thoi_gian_thi_bao_loi(tmp_path, row):
    p = tmp_path / "a.csv"
    p.write_text("nhan,bat_dau,ket_thuc,ghi_chu\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_gt(p)


GT = [GtEvent("no_helmet", 5, 20), GtEvent("no_helmet", 40, 50), GtEvent("fall", 60, 70)]


def test_bao_dung_trong_khoang_tinh_do_tre():
    s = match_events(GT, [("no_helmet", 8.0)])["no_helmet"]
    assert (s.tp, s.fp, s.fn, s.latencies) == (1, 0, 1, [3.0])


def test_bao_sau_khi_het_khoang_trong_dung_sai_van_dung():
    """Luật cần 3 giây mới báo, vi phạm ngắn thì sự cố tới sau khi khoảng đã hết."""
    assert match_events(GT, [("no_helmet", 23.0)], tolerance=5)["no_helmet"].tp == 1
    assert match_events(GT, [("no_helmet", 26.0)], tolerance=5)["no_helmet"].fp == 1


def test_bao_lai_trong_cung_khoang_la_bao_lap_khong_phai_bao_nham():
    s = match_events(GT, [("no_helmet", 8.0), ("no_helmet", 15.0)])["no_helmet"]
    assert (s.tp, s.repeats, s.fp) == (1, 1, 0)


def test_sai_nhan_la_bao_nham_va_bo_sot():
    scores = match_events(GT, [("no_vest", 62.0)])
    assert scores["no_vest"].fp == 1
    assert scores["fall"].fn == 1


def test_precision_recall_f1():
    s = match_events(GT, [("no_helmet", 8.0), ("no_helmet", 30.0)])["no_helmet"]
    assert (s.precision, s.recall, s.f1) == (0.5, 0.5, 0.5)


def test_doan_video_trung_khoang_vi_pham_la_doan_co():
    gt = [GtEvent("fire", 3.0, 5.0)]
    assert window_truth(gt, "fire", duration=10, window=2) == [0, 1, 1, 0, 0]


def test_auroc_tinh_tay():
    # Cặp (có, không): 0,9>0,8; 0,9>0,3; 0,4<0,8; 0,4>0,3 -> 3/4.
    assert auroc([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.3]) == pytest.approx(0.75)
    assert auroc([1, 0], [0.9, 0.1]) == 1.0
    assert auroc([1, 0], [0.5, 0.5]) == 0.5
    assert auroc([1, 1], [0.5, 0.6]) is None


def test_auprc_tinh_tay():
    # Xếp giảm dần: 0,9 (có) -> precision 1; 0,4 (có) -> precision 2/3. Trung bình = 5/6.
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.3]) == pytest.approx(5 / 6)
    assert average_precision([0, 0], [0.1, 0.2]) is None


def test_diem_khung_khong_mu_lay_trong_vung_dau():
    person = Detection(DetectClass.PERSON, BBox(100, 100, 200, 400), 0.9, track_id=1)
    head = Detection(DetectClass.NO_HELMET, BBox(130, 105, 170, 150), 0.6)
    feet = Detection(DetectClass.NO_HELMET, BBox(130, 350, 170, 395), 0.9)  # ngoài vùng đầu
    fire = Detection(DetectClass.FIRE, BBox(400, 300, 450, 350), 0.7)
    r = FrameResult("cam", 1.0, 640, 480, (person, head, feet, fire))
    s = frame_scores(r, PPERule("cam"))
    assert s["no_helmet"] == 0.6 and s["fire"] == 0.7 and s["no_vest"] == 0.0
