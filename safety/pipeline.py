"""Chạy YOLO + tracking trên từng khung hình, trả về FrameResult theo contract.

Một SafetyPipeline phục vụ nhiều camera: mỗi mô hình chỉ nạp một lần và chạy
theo lô (một lô = khung mới nhất của mọi camera), còn bộ bám vết thì mỗi camera
một cái. GPU 4 GB không chứa nổi 4–8 bản sao của ba mô hình (tư thế, đồ bảo hộ,
cháy), nên không thể chạy mỗi camera một tiến trình. Bộ bám vết thì không được
dùng chung: trộn hai luồng vào một tracker sẽ làm track_id nhảy loạn. Vì vậy ở
đây không gọi model.track() (Ultralytics giữ một tracker bên trong mô hình) mà
tự giữ BYTETracker theo tên camera, làm đúng các bước callback bám vết của
Ultralytics làm.

Mô hình chính (người, đồ bảo hộ) chạy kèm bám vết vì luật PPE đếm giờ theo
từng người. Mô hình phụ (cháy/khói) chỉ nhận diện, không bám vết: lửa không có
mã định danh, luật cháy xét mật độ chứ không đếm theo người. Mô hình cháy và mô
hình đồ bảo hộ là hai tệp riêng, nên mỗi khung chạy hai lần suy luận.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import torch
from ultralytics import YOLO

from safety.contracts import BBox, Detection, DetectClass, FrameResult, Keypoint

log = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Chuẩn hoá tên lớp: "No-Helmet", "no helmet" đều thành no_helmet, "Fire" thành fire."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _map_classes(names: dict[int, str]) -> dict[int, DetectClass]:
    """Lớp model không có trong DetectClass thì bỏ qua.

    Model COCO mặc định có 80 lớp nhưng ta chỉ quan tâm person; model PPE và
    model cháy của Nam Trường khớp thêm helmet/vest/fire/smoke. Tên được chuẩn
    hoá chữ hoa/thường và gạch nối trước khi khớp, để lệch kiểu viết không làm
    mất lớp một cách lặng lẽ.
    """
    mapping = {}
    for idx, name in names.items():
        try:
            mapping[idx] = DetectClass(_normalize(name))
        except ValueError:
            continue
    return mapping


class SafetyPipeline:
    def __init__(
        self,
        model: str,
        camera: str,
        device: str | int = 0,
        conf: float = 0.25,
        imgsz: int | None = None,
        tracker: str = "bytetrack.yaml",
        extra_models: Sequence[str] = (),
        tracker_factory: Callable[[], object] | None = None,
    ):
        self.camera = camera
        self.device = device
        self.conf = conf
        self.tracker = tracker
        self._new_tracker = tracker_factory or (lambda: _make_tracker(tracker, device))
        self._trackers: dict[str, object] = {}
        self.model = YOLO(model)
        # Mặc định chạy mỗi mô hình ở đúng cỡ ảnh nó được train. Đo 16/10: mô hình cháy
        # train ở 512 mà ép chạy ở 640 thì báo nhầm lửa tăng từ 4,7% lên 15,1% số khung.
        self.imgsz = imgsz or _trained_imgsz(self.model)
        self.classes = _map_classes(self.model.names)
        log.info(
            "Model %s: dùng %d/%d lớp -> %s, cỡ ảnh %s",
            model,
            len(self.classes),
            len(self.model.names),
            sorted(c.value for c in self.classes.values()),
            self.imgsz,
        )

        # Mô hình phụ chỉ góp những lớp mô hình chính chưa có, để một vật không
        # bị nhận hai lần (ví dụ cả hai mô hình cùng có lớp person).
        self.extras: list[tuple[YOLO, dict[int, DetectClass], int]] = []
        for path in extra_models:
            extra = YOLO(path)
            have = set(self.detect_classes)
            mapping = {i: c for i, c in _map_classes(extra.names).items() if c not in have}
            if not mapping:
                log.warning(
                    "Model phụ %s không có lớp nào mới so với các model trước (lớp của nó: %s), bỏ qua",
                    path, sorted(extra.names.values()),
                )
                continue
            size = _trained_imgsz(extra)
            log.info("Model phụ %s: dùng %d/%d lớp -> %s, cỡ ảnh %s, không bám vết",
                     path, len(mapping), len(extra.names), sorted(c.value for c in mapping.values()), size)
            self.extras.append((extra, mapping, size))

    @property
    def detect_classes(self) -> set[DetectClass]:
        """Mọi lớp mà pipeline nhận diện được, gộp cả mô hình chính lẫn mô hình phụ."""
        found = set(self.classes.values())
        for _, mapping, _ in self.extras:
            found.update(mapping.values())
        return found

    def process(self, ts: float, frame: np.ndarray) -> FrameResult:
        """Một khung của camera mặc định (tên camera truyền lúc tạo pipeline)."""
        return self.process_batch([(self.camera, ts, frame)])[0]

    def process_batch(self, items: Sequence[tuple[str, float, np.ndarray]]) -> list[FrameResult]:
        """Mỗi phần tử là (camera, ts, frame); trả về FrameResult cùng thứ tự.

        Một lô không được có hai khung cùng camera: bộ bám vết của camera đó sẽ
        nhận hai khung trong một bước và coi như hai thời điểm liên tiếp.
        """
        if not items:
            return []
        cameras = [cam for cam, _, _ in items]
        if len(set(cameras)) != len(cameras):
            raise ValueError(f"Một lô có hai khung cùng camera: {cameras}")
        frames = [frame for _, _, frame in items]

        results = self.model.predict(frames, device=self.device, conf=self.conf,
                                     imgsz=self.imgsz, verbose=False)
        per_frame = []
        for cam, result in zip(cameras, results):
            result = self._track(cam, result)
            per_frame.append(list(_detections(result, self.classes)))

        for extra, mapping, size in self.extras:
            extra_results = extra.predict(frames, device=self.device, conf=self.conf,
                                          imgsz=size, verbose=False)
            for dets, res in zip(per_frame, extra_results):
                dets.extend(_detections(res, mapping))

        out = []
        for (cam, ts, frame), dets in zip(items, per_frame):
            height, width = frame.shape[:2]
            out.append(FrameResult(camera=cam, ts=ts, width=width, height=height,
                                   detections=tuple(dets)))
        return out

    def _track(self, camera: str, result):
        """Gắn track_id cho kết quả của một camera, giống callback bám vết của Ultralytics.

        Tracker không trả về gì (vd khung đầu, hay chưa đủ điểm tin cậy để mở mã mới)
        thì giữ nguyên kết quả không có mã, như model.track() vẫn làm: luật bỏ qua
        người chưa có mã chứ không đếm sai.
        """
        tracker = self._trackers.get(camera)
        if tracker is None:
            tracker = self._trackers[camera] = self._new_tracker()
        boxes = result.boxes
        if boxes is None:
            return result
        tracks = tracker.update(boxes.cpu().numpy(), result.orig_img)
        if len(tracks) == 0:
            return result
        result = result[tracks[:, -1].astype(int)]
        result.update(boxes=torch.as_tensor(tracks[:, :-1], device=boxes.data.device))
        return result


def _make_tracker(config: str, device):
    """BYTETracker dựng từ tệp cấu hình y như Ultralytics dựng khi gọi model.track()."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(config)))
    cfg.device = device
    return TRACKER_MAP[cfg.tracker_type](args=cfg)


def _trained_imgsz(model, default: int = 640) -> int:
    """Cỡ ảnh lúc train, Ultralytics lưu trong tệp .pt. Không có thì dùng 640."""
    size = (getattr(model, "overrides", None) or {}).get("imgsz")
    return int(size) if isinstance(size, (int, float)) and size > 0 else default


def _detections(result, classes: dict[int, DetectClass]) -> tuple[Detection, ...]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_idx = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    kps = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None

    out = []
    for i, idx in enumerate(cls_idx):
        cls = classes.get(idx)
        if cls is None:
            continue
        out.append(
            Detection(
                cls=cls,
                bbox=BBox.from_list(xyxy[i]),
                conf=float(confs[i]),
                track_id=int(ids[i]) if ids is not None else None,
                keypoints=_keypoints(kps[i]) if kps is not None else None,
            )
        )
    return tuple(out)


def _keypoints(raw: np.ndarray) -> tuple[Keypoint, ...]:
    return tuple(Keypoint(float(x), float(y), float(c)) for x, y, c in raw)
