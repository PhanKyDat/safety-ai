"""Kiểm thử luật PPE bằng dữ liệu giả, chưa cần model của Nam Trường.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from safety.contracts import BBox, Detection, DetectClass, FrameResult, IncidentLabel, Keypoint
from safety.rules import PPEConfig, PPERule

PERSON = BBox(100, 100, 200, 400)
HEAD = BBox(110, 110, 190, 160)      # nằm trong 25% trên của khung người
BODY = BBox(110, 250, 190, 350)      # dưới vùng đầu


def frame(ts: float, *dets: Detection) -> FrameResult:
    return FrameResult(camera="xuong_han", ts=ts, width=640, height=360, detections=dets)


def person(track_id: int = 1) -> Detection:
    return Detection(DetectClass.PERSON, PERSON, 0.9, track_id=track_id)


def at(cls: DetectClass, box: BBox, conf: float = 0.8) -> Detection:
    return Detection(cls, box, conf)


def run(rule: PPERule, frames):
    out = []
    for f in frames:
        out.extend(rule.update(f))
    return out


def test_bao_sau_du_3_giay():
    rule = PPERule("xuong_han")
    frames = [frame(t / 5, person(), at(DetectClass.NO_HELMET, HEAD)) for t in range(21)]
    incidents = run(rule, frames)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.label is IncidentLabel.NO_HELMET
    assert inc.track_id == 1
    assert inc.sub_label == "ID 1"
    assert inc.ts >= 3.0


def test_chua_du_3_giay_thi_khong_bao():
    rule = PPERule("xuong_han")
    frames = [frame(t / 5, person(), at(DetectClass.NO_HELMET, HEAD)) for t in range(14)]
    assert run(rule, frames) == []


def test_co_mu_thi_khong_bao():
    rule = PPERule("xuong_han")
    frames = [frame(t / 5, person(), at(DetectClass.HELMET, HEAD)) for t in range(30)]
    assert run(rule, frames) == []


def test_mat_dau_ngan_van_tinh_lien_tuc():
    """Người khuất sau kệ 0,6 giây rồi hiện lại: đồng hồ không được đặt lại."""
    rule = PPERule("xuong_han")
    frames = []
    for t in range(21):
        if t in (7, 8, 9):
            frames.append(frame(t / 5, person()))          # chỉ thấy người, không kết luận được
        else:
            frames.append(frame(t / 5, person(), at(DetectClass.NO_HELMET, HEAD)))
    assert len(run(rule, frames)) == 1


def test_deo_mu_lai_thi_dat_lai_dong_ho():
    rule = PPERule("xuong_han")
    frames = []
    for t in range(21):
        acc = at(DetectClass.HELMET, HEAD) if t == 10 else at(DetectClass.NO_HELMET, HEAD)
        frames.append(frame(t / 5, person(), acc))
    assert run(rule, frames) == []


def test_cooldown_chan_bao_lap():
    rule = PPERule("xuong_han", PPEConfig(cooldown=60.0))
    frames = [frame(t / 5, person(), at(DetectClass.NO_HELMET, HEAD)) for t in range(150)]
    assert len(run(rule, frames)) == 1


def test_het_cooldown_thi_bao_lai():
    rule = PPERule("xuong_han", PPEConfig(cooldown=10.0))
    frames = [frame(t / 5, person(), at(DetectClass.NO_HELMET, HEAD)) for t in range(150)]
    assert len(run(rule, frames)) == 3


def test_mu_cua_nguoi_khac_khong_tinh():
    """Mũ nằm ở thân người (hoặc của người bên cạnh) không được coi là đội mũ."""
    rule = PPERule("xuong_han")
    frames = [
        frame(t / 5, person(), at(DetectClass.HELMET, BODY), at(DetectClass.NO_HELMET, HEAD))
        for t in range(21)
    ]
    assert len(run(rule, frames)) == 1


def test_hai_nguoi_dem_rieng():
    rule = PPERule("xuong_han")
    other = BBox(300, 100, 400, 400)
    other_head = BBox(310, 110, 390, 160)
    frames = []
    for t in range(21):
        dets = [person(1), at(DetectClass.NO_HELMET, HEAD),
                Detection(DetectClass.PERSON, other, 0.9, track_id=2)]
        if t >= 10:
            dets.append(at(DetectClass.NO_HELMET, other_head))
        frames.append(frame(t / 5, *dets))
    incidents = run(rule, frames)
    assert [i.track_id for i in incidents] == [1]


def test_thieu_ao_phan_quang():
    rule = PPERule("xuong_han")
    frames = [frame(t / 5, person(), at(DetectClass.NO_VEST, BODY)) for t in range(21)]
    incidents = run(rule, frames)
    assert len(incidents) == 1
    assert incidents[0].label is IncidentLabel.NO_VEST


def test_khong_co_track_id_thi_bo_qua():
    rule = PPERule("xuong_han")
    no_id = Detection(DetectClass.PERSON, PERSON, 0.9)
    frames = [frame(t / 5, no_id, at(DetectClass.NO_HELMET, HEAD)) for t in range(21)]
    assert run(rule, frames) == []


# --- Cách "vắng mặt": suy ra thiếu mũ/áo khi thấy rõ người mà không thấy đồ bảo hộ ---

def kp_person(head_conf: float = 0.9, body_conf: float = 0.9, track_id: int = 1) -> Detection:
    """17 điểm khớp COCO: 0–4 mặt/tai, 5–6 vai, 11–12 hông."""
    kps = [Keypoint(150, 120, head_conf)] * 5 + [Keypoint(150, 200, body_conf)] * 12
    return Detection(DetectClass.PERSON, PERSON, 0.9, track_id=track_id, keypoints=tuple(kps))


VANG_MAT = PPEConfig(infer_missing=True)


def test_mac_dinh_van_cach_cu_khong_thay_lop_vi_pham_thi_khong_bao():
    rule = PPERule("xuong_han")
    assert run(rule, [frame(t / 5, kp_person()) for t in range(30)]) == []


def test_vang_mat_thay_ro_dau_ma_khong_co_mu_thi_bao_thieu_mu():
    rule = PPERule("xuong_han", VANG_MAT)
    labels = {i.label for i in run(rule, [frame(t / 5, kp_person(), at(DetectClass.VEST, BODY)) for t in range(21)])}
    assert labels == {IncidentLabel.NO_HELMET}


def test_vang_mat_co_mu_va_ao_thi_im_lang():
    rule = PPERule("xuong_han", VANG_MAT)
    frames = [frame(t / 5, kp_person(), at(DetectClass.HELMET, HEAD), at(DetectClass.VEST, BODY)) for t in range(30)]
    assert run(rule, frames) == []


def test_vang_mat_khong_thay_ro_nguoi_thi_khong_ket_luan():
    """Quay lưng, bị che: điểm mặt/vai mờ thì không được coi là vi phạm."""
    rule = PPERule("xuong_han", VANG_MAT)
    assert run(rule, [frame(t / 5, kp_person(head_conf=0.2, body_conf=0.2)) for t in range(30)]) == []


def test_vang_mat_thay_ro_than_ma_khong_co_ao_thi_bao_thieu_ao():
    rule = PPERule("xuong_han", VANG_MAT)
    labels = {i.label for i in run(rule, [frame(t / 5, kp_person(), at(DetectClass.HELMET, HEAD)) for t in range(21)])}
    assert labels == {IncidentLabel.NO_VEST}


def test_vang_mat_can_mo_hinh_tu_the():
    """Mô hình không có điểm khớp thì không biết người có được thấy rõ không, nên không suy ra."""
    rule = PPERule("xuong_han", VANG_MAT)
    assert run(rule, [frame(t / 5, person()) for t in range(30)]) == []


# --- Chống báo trùng khi đổi mã người ---

def person_at(track_id: int, dx: float = 0.0) -> Detection:
    return Detection(DetectClass.PERSON, BBox(100 + dx, 100, 200 + dx, 400), 0.9, track_id=track_id)


def violating_frames(track_id: int, t0: float, seconds: float, dx: float = 0.0):
    return [frame(t0 + t / 5, person_at(track_id, dx), at(DetectClass.NO_HELMET, BBox(110 + dx, 110, 190 + dx, 160)))
            for t in range(int(seconds * 5))]


def test_doi_ma_ngay_cho_cu_thi_khong_bao_lai():
    """Khuất sau kệ 1 giây rồi hiện lại cùng chỗ với mã mới: vẫn là một người, một sự cố."""
    rule = PPERule("xuong_han")
    frames = violating_frames(1, 0.0, 5) + violating_frames(2, 6.0, 6)
    assert [i.track_id for i in run(rule, frames)] == [1]


def test_nguoi_khac_o_cho_khac_van_bi_bao():
    """Không nhận diện lại được người, nên chỉ gộp khi hiện lại gần chỗ cũ."""
    rule = PPERule("xuong_han")
    frames = violating_frames(1, 0.0, 5) + violating_frames(2, 6.0, 6, dx=400)
    assert [i.track_id for i in run(rule, frames)] == [1, 2]


def test_mat_ma_lau_hon_cua_so_thi_tinh_nguoi_moi():
    rule = PPERule("xuong_han", PPEConfig(reentry_window=10.0))
    frames = violating_frames(1, 0.0, 5) + violating_frames(2, 20.0, 6)
    assert [i.track_id for i in run(rule, frames)] == [1, 2]


def test_tat_chong_trung_thi_bao_theo_tung_ma():
    rule = PPERule("xuong_han", PPEConfig(reentry_window=0))
    frames = violating_frames(1, 0.0, 5) + violating_frames(2, 6.0, 6)
    assert [i.track_id for i in run(rule, frames)] == [1, 2]
