"""Đọc nhãn mu_bao_ho từ sự kiện Frigate (scripts/b1_doi_chung.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import b1_doi_chung as b1  # noqa: E402


def test_frigate_017_ghi_thang_vao_data():
    # Dạng thật lấy từ API Frigate 0.17.2 ngày 11/11: attributes là danh sách rỗng.
    ev = {"data": {"attributes": [], "mu_bao_ho": "khong_mu", "mu_bao_ho_score": 0.96}}
    assert b1.nhan_phan_loai(ev) == "khong_mu"


def test_du_phong_attributes_dang_dict():
    assert b1.nhan_phan_loai({"data": {"attributes": {"mu_bao_ho": "co_mu"}}}) == "co_mu"


def test_chua_phan_loai():
    assert b1.nhan_phan_loai({"data": {"attributes": []}}) is None
    assert b1.nhan_phan_loai({}) is None


def test_ghep_theo_cua_so_thoi_gian():
    frigate = [{"camera": "xuong_han", "start_time": 100.0, "end_time": 120.0}]
    module = [
        {"camera": "xuong_han", "ts": 125.0},  # lệch 5 s sau khi kết thúc -> khớp
        {"camera": "xuong_han", "ts": 200.0},  # quá xa
        {"camera": "kho_1", "ts": 110.0},  # khác camera
    ]
    assert b1.ghep(frigate, module, dung_sai=10.0) == 1
