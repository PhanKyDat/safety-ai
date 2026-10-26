"""Kiểm thử ghép hai mô hình (người/PPE có bám vết + cháy/khói không bám vết),
và chạy nhiều camera chung một bộ mô hình nhưng mỗi camera một bộ bám vết.

Dùng mô hình và tracker giả nên không cần GPU hay tệp .pt thật.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import types

import numpy as np
import pytest

import safety.pipeline as pipeline_mod
from safety.contracts import DetectClass
from safety.pipeline import SafetyPipeline, _map_classes

FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


class T:
    """Đứng thay tensor của ultralytics: .cpu().numpy() trả về mảng."""

    def __init__(self, a):
        self.a = np.array(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, rows, ids=None):
        self.rows = rows
        self.xyxy = T([r[1] for r in rows])
        self.conf = T([r[2] for r in rows])
        self.cls = T([r[0] for r in rows])
        self.id = T(ids) if ids is not None else None
        self.hint_ids = ids
        self.data = types.SimpleNamespace(device="cpu")
        self._n = len(rows)

    def __len__(self):
        return self._n

    def cpu(self):
        return self

    def numpy(self):
        return self


class Result:
    def __init__(self, rows, ids=None):
        self.boxes = Boxes(rows, ids) if rows else None
        self.keypoints = None
        self.orig_img = None

    def __getitem__(self, idx):
        return Result([self.boxes.rows[i] for i in idx])

    def update(self, boxes):
        """Như Results.update của ultralytics: nhận hàng [x1, y1, x2, y2, id, score, cls]."""
        rows = [(int(r[6]), list(r[:4]), float(r[5])) for r in boxes.tolist()]
        self.boxes = Boxes(rows, [int(r[4]) for r in boxes.tolist()])


class FakeTracker:
    """Gắn mã theo ids của mô hình giả (hoặc 1, 2, 3...); nhớ đã thấy những khung nào."""

    def __init__(self):
        self.seen = []

    def update(self, det, img):
        self.seen.append(img)
        ids = det.hint_ids or list(range(1, len(det) + 1))
        return np.array([[*r[1], i, r[2], r[0], k] for k, (r, i) in enumerate(zip(det.rows, ids))],
                        dtype=float)


class FakeModel:
    def __init__(self, names, rows, ids=None, imgsz=None):
        self.names = names
        self.rows, self.ids = rows, ids
        self.overrides = {"imgsz": imgsz} if imgsz else {}
        self.predict_calls = 0
        self.batches = []
        self.last_kw = {}

    def predict(self, frames, **kw):
        self.predict_calls += 1
        self.batches.append(len(frames))
        self.last_kw = kw
        out = []
        for f in frames:
            r = Result(self.rows, self.ids)
            r.orig_img = f
            out.append(r)
        return out


PERSON_BOX = [100, 100, 200, 300]
FIRE_BOX = [300, 200, 360, 260]


@pytest.fixture
def models(monkeypatch):
    zoo = {}
    monkeypatch.setattr(pipeline_mod, "YOLO", lambda path: zoo[path])
    monkeypatch.setattr(pipeline_mod, "_make_tracker", lambda config, device: FakeTracker())
    return zoo


def test_gop_nguoi_co_ma_dinh_danh_va_lua_khong_ma(models):
    models["coco.pt"] = FakeModel({0: "person", 1: "car"}, [(0, PERSON_BOX, 0.9)], ids=[7])
    models["fire.pt"] = FakeModel({0: "fire", 1: "smoke"}, [(0, FIRE_BOX, 0.8)])
    p = SafetyPipeline("coco.pt", "xuong_han", extra_models=["fire.pt"])
    r = p.process(1.0, FRAME)
    got = {(d.cls, d.track_id) for d in r.detections}
    assert got == {(DetectClass.PERSON, 7), (DetectClass.FIRE, None)}
    assert len(p._trackers["xuong_han"].seen) == 1  # chỉ mô hình chính qua bộ bám vết
    assert models["fire.pt"].predict_calls == 1


def test_lop_trung_o_mo_hinh_phu_bi_bo_de_khong_dem_hai_lan(models):
    models["ppe.pt"] = FakeModel({0: "person", 1: "helmet"}, [(0, PERSON_BOX, 0.9)], ids=[1])
    models["fire.pt"] = FakeModel({0: "person", 1: "fire"}, [(0, PERSON_BOX, 0.7), (1, FIRE_BOX, 0.8)])
    p = SafetyPipeline("ppe.pt", "xuong_han", extra_models=["fire.pt"])
    classes = [d.cls for d in p.process(1.0, FRAME).detections]
    assert classes.count(DetectClass.PERSON) == 1
    assert DetectClass.FIRE in classes


def test_mo_hinh_phu_khong_them_lop_moi_thi_khong_chay(models):
    """Không chạy thì không tốn thêm một lần suy luận mỗi khung."""
    models["coco.pt"] = FakeModel({0: "person"}, [(0, PERSON_BOX, 0.9)], ids=[1])
    models["coco2.pt"] = FakeModel({0: "person"}, [(0, PERSON_BOX, 0.9)])
    p = SafetyPipeline("coco.pt", "xuong_han", extra_models=["coco2.pt"])
    p.process(1.0, FRAME)
    assert p.extras == [] and models["coco2.pt"].predict_calls == 0


def test_detect_classes_gop_ca_hai_mo_hinh(models):
    models["coco.pt"] = FakeModel({0: "person"}, [])
    models["fire.pt"] = FakeModel({0: "fire", 1: "smoke"}, [])
    p = SafetyPipeline("coco.pt", "xuong_han", extra_models=["fire.pt"])
    assert p.detect_classes == {DetectClass.PERSON, DetectClass.FIRE, DetectClass.SMOKE}


def test_khong_co_mo_hinh_phu_thi_chay_nhu_cu(models):
    models["coco.pt"] = FakeModel({0: "person"}, [(0, PERSON_BOX, 0.9)], ids=[3])
    p = SafetyPipeline("coco.pt", "xuong_han")
    assert [d.track_id for d in p.process(1.0, FRAME).detections] == [3]


def test_moi_mo_hinh_chay_dung_co_anh_luc_train(models):
    """Mô hình cháy train ở 512 mà ép chạy 640 thì báo nhầm lửa tăng từ 4,7% lên 15,1%."""
    models["coco.pt"] = FakeModel({0: "person"}, [], imgsz=640)
    models["fire.pt"] = FakeModel({0: "fire"}, [], imgsz=512)
    p = SafetyPipeline("coco.pt", "xuong_han", extra_models=["fire.pt"])
    p.process(1.0, FRAME)
    assert models["coco.pt"].last_kw["imgsz"] == 640
    assert models["fire.pt"].last_kw["imgsz"] == 512


def test_ten_lop_lech_kieu_viet_van_khop():
    """Lệch chữ hoa hay gạch nối thì vẫn khớp, không mất lớp một cách lặng lẽ."""
    m = _map_classes({0: "Fire", 1: "SMOKE", 2: "No-Helmet", 3: "no vest", 4: "forklift"})
    assert m == {0: DetectClass.FIRE, 1: DetectClass.SMOKE,
                 2: DetectClass.NO_HELMET, 3: DetectClass.NO_VEST}


def test_nhieu_camera_chung_mo_hinh_moi_camera_mot_bo_bam_vet(models):
    """Ba camera trong một lô: mỗi mô hình chạy một lần cho cả lô, mỗi camera một tracker."""
    models["coco.pt"] = FakeModel({0: "person"}, [(0, PERSON_BOX, 0.9)])
    models["fire.pt"] = FakeModel({0: "fire"}, [(0, FIRE_BOX, 0.8)])
    p = SafetyPipeline("coco.pt", "xuong_han", extra_models=["fire.pt"])
    frames = {cam: np.full((360, 640, 3), i, dtype=np.uint8) for i, cam in enumerate(["a", "b", "c"])}
    out = p.process_batch([(cam, 2.0, f) for cam, f in frames.items()])

    assert [r.camera for r in out] == ["a", "b", "c"]
    assert models["coco.pt"].batches == [3] and models["fire.pt"].batches == [3]
    assert set(p._trackers) == {"a", "b", "c"}
    for cam, tracker in p._trackers.items():
        assert len(tracker.seen) == 1 and tracker.seen[0] is frames[cam]
    for r in out:
        assert {d.cls for d in r.detections} == {DetectClass.PERSON, DetectClass.FIRE}


def test_tracker_cua_mot_camera_khong_bi_khung_camera_khac_cap_nhat(models):
    """Lô sau thiếu camera b (mất luồng) thì tracker của b đứng yên, không nhận khung của a."""
    models["coco.pt"] = FakeModel({0: "person"}, [(0, PERSON_BOX, 0.9)])
    p = SafetyPipeline("coco.pt", "a")
    p.process_batch([("a", 1.0, FRAME), ("b", 1.0, FRAME)])
    p.process_batch([("a", 1.2, FRAME)])
    assert len(p._trackers["a"].seen) == 2
    assert len(p._trackers["b"].seen) == 1


def test_lo_co_hai_khung_cung_camera_thi_bao_loi(models):
    models["coco.pt"] = FakeModel({0: "person"}, [])
    p = SafetyPipeline("coco.pt", "a")
    with pytest.raises(ValueError):
        p.process_batch([("a", 1.0, FRAME), ("a", 1.2, FRAME)])


def test_tracker_chua_tra_ma_thi_nguoi_chua_co_track_id(models):
    """Giống model.track(): tracker chưa mở mã thì giữ hộp nhưng không có mã, luật sẽ bỏ qua."""

    class Empty(FakeTracker):
        def update(self, det, img):
            return np.empty((0, 8))

    models["coco.pt"] = FakeModel({0: "person"}, [(0, PERSON_BOX, 0.9)])
    p = SafetyPipeline("coco.pt", "a", tracker_factory=Empty)
    (r,) = p.process_batch([("a", 1.0, FRAME)])
    assert [(d.cls, d.track_id) for d in r.detections] == [(DetectClass.PERSON, None)]
