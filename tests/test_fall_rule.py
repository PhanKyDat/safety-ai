"""Kiểm thử luật ngã / nằm bằng điểm khớp giả (17 điểm COCO), không cần mô hình.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import numpy as np

from safety.contracts import BBox, Detection, DetectClass, FrameResult, IncidentLabel, Keypoint, Severity
from safety.incidents import IncidentProcessor
from safety.rules import ExcludeZone, FallConfig, FallRule, body_tilt

T0 = 1789012340.0


def pose(shoulder, hip, foot=None, conf=0.9):
    """17 điểm: chỉ đặt vai (5, 6), hông (11, 12), cổ chân (15, 16); còn lại độ tin cậy 0."""
    pts = [Keypoint(0, 0, 0.0)] * 17
    for i, (x, y) in ((5, shoulder), (6, shoulder), (11, hip), (12, hip)):
        pts[i] = Keypoint(x, y, conf)
    if foot is not None:
        pts[15] = pts[16] = Keypoint(foot[0], foot[1], conf)
    return tuple(pts)


STANDING = pose((150, 120), (150, 200), (150, 290))
LYING = pose((100, 280), (180, 285), (260, 290))       # thân và chân nằm ngang trên sàn
BENDING = pose((100, 200), (170, 205), (170, 290))      # thân ngang nhưng chân đứng: cúi nhặt đồ
LYING_NO_FEET = pose((100, 280), (180, 285))            # chân bị kệ che
HIDDEN = pose((100, 280), (180, 285), conf=0.1)         # không thấy rõ vai, hông


def person(kps, box=BBox(90, 100, 270, 300), track_id=1):
    return Detection(DetectClass.PERSON, box, 0.88, track_id=track_id, keypoints=kps)


def frames(kps_seq, start=T0, fps=5.0, **kw):
    return [FrameResult("xuong_han", start + i / fps, 640, 360, (person(k, **kw),))
            for i, k in enumerate(kps_seq)]


def run(rule, seq):
    out = []
    for f in seq:
        out.extend(rule.update(f))
    return out


def test_goc_than_va_chan():
    torso, legs = body_tilt(person(STANDING))
    assert torso < 1 and legs < 1
    torso, legs = body_tilt(person(LYING))
    assert torso > 85 and legs > 85
    assert body_tilt(person(HIDDEN)) == (None, None)


def test_nam_du_3_giay_thi_bao_nga():
    incidents = run(FallRule("xuong_han"), frames([LYING] * 16))      # 0 -> 3,0 s
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.label is IncidentLabel.FALL and inc.severity is Severity.CRITICAL
    assert inc.track_id == 1 and inc.sub_label == "ID 1 - nằm 3s"
    assert inc.ts == T0 + 3.0


def test_nam_chua_du_lau_thi_khong_bao():
    assert run(FallRule("xuong_han"), frames([LYING] * 14 + [STANDING] * 20)) == []


def test_dung_va_cui_nhat_do_khong_bao():
    assert run(FallRule("xuong_han"), frames([STANDING] * 50)) == []
    assert run(FallRule("xuong_han"), frames([BENDING] * 50)) == []


def test_khong_thay_chan_thi_dua_vao_than():
    assert len(run(FallRule("xuong_han"), frames([LYING_NO_FEET] * 16))) == 1


def test_bi_che_mot_luc_khong_lam_mat_dong_ho():
    """Không thấy rõ vai/hông thì chưa kết luận, không phải bằng chứng đã đứng dậy."""
    seq = [LYING] * 8 + [HIDDEN] * 4 + [LYING] * 4                     # vẫn tới 3,0 s
    assert len(run(FallRule("xuong_han"), frames(seq))) == 1


def test_mat_dau_qua_lau_thi_dem_lai():
    rule = FallRule("xuong_han")
    first = frames([LYING] * 10)                                        # 0 -> 1,8 s
    later = frames([LYING] * 10, start=T0 + 4.0)                         # mất 2,2 s > dung sai 1 s
    assert run(rule, first + later) == []


def test_khong_bao_don_khi_van_nam():
    assert len(run(FallRule("xuong_han"), frames([LYING] * 200))) == 1   # 40 s, chờ 60 s mới báo lại


def test_dung_day_roi_nam_lai_ngay_khong_bao_lai():
    seq = [LYING] * 16 + [STANDING] * 5 + [LYING] * 20
    assert len(run(FallRule("xuong_han"), frames(seq))) == 1


def test_vung_loai_tru():
    """Ví dụ chỗ thợ nằm sửa dưới gầm xe nâng."""
    cfg = FallConfig(exclude=(ExcludeZone(0.0, 0.2, 0.5, 1.0, "gam xe nang"),))
    assert run(FallRule("xuong_han", cfg), frames([LYING] * 30)) == []


def test_mo_hinh_khong_co_diem_khop_thi_khong_bao():
    assert run(FallRule("xuong_han"), frames([None] * 30)) == []


def test_su_co_nga_di_qua_bo_xu_ly_va_frigate_khoanh_nguoi(tmp_path):
    """Người đã nằm thì đứng yên, nên Frigate chụp muộn ~0,3 s vẫn khoanh đúng."""
    proc = IncidentProcessor("xuong_han", snapshot_dir=tmp_path)
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    out = []
    for f in frames([LYING] * 16):
        out.extend(proc.handle(image, f))
    assert [i.label for i in out] == [IncidentLabel.FALL]
    assert out[0].snapshot.endswith("_fall_1789012343_id1.jpg")
    assert out[0].to_frigate_payload()["draw"]["boxes"][0]["box"] == [0.1406, 0.2778, 0.2812, 0.5556]
