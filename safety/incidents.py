"""Nối luật với đầu ra: khung hình -> luật -> ảnh chụp -> Frigate.

Một IncidentProcessor cho mỗi camera, giống SafetyPipeline: luật giữ trạng thái
theo track_id của camera đó, trộn hai camera vào một bộ luật sẽ đếm lẫn người.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from safety.contracts import Detection, DetectClass, FrameResult, Incident, IncidentLabel
from safety.rules import FireConfig, FireRule, PPEConfig, PPERule

log = logging.getLogger(__name__)

RED = (60, 60, 220)
WHITE = (255, 255, 255)

# Sự cố cháy/khói không gắn với người nào, nên ảnh chụp khoanh mọi hộp lửa/khói trong khung.
_EVIDENCE = {
    IncidentLabel.FIRE: DetectClass.FIRE,
    IncidentLabel.SMOKE: DetectClass.SMOKE,
}


def snapshot_name(incident: Incident) -> str:
    """Tên tệp ảnh: camera_nhãn_giây[_idN]. Hai người cùng vi phạm trong một giây vẫn khác tên."""
    name = f"{incident.camera}_{incident.label.value}_{incident.ts:.0f}"
    if incident.track_id is not None:
        name += f"_id{incident.track_id}"
    return name + ".jpg"


def evidence(
    result: FrameResult, incident: Incident, skip: Callable[[Detection], bool] | None = None
) -> list[Detection]:
    """Các hộp cần khoanh: người vi phạm (theo track_id), hoặc mọi hộp lửa/khói trong khung.

    `skip` loại hộp nằm trong vùng loại trừ của luật cháy, để không khoanh đúng chỗ
    luật đã bỏ qua.
    """
    cls = _EVIDENCE.get(incident.label)
    found = []
    for d in result.detections:
        if cls is not None:
            if d.cls is cls and not (skip and skip(d)):
                found.append(d)
        elif (d.cls is DetectClass.PERSON and d.track_id is not None
              and d.track_id == incident.track_id):
            found.append(d)
    return found


def relative_box(d: Detection, width: int, height: int) -> tuple[float, float, float, float]:
    """Pixel x1,y1,x2,y2 -> tỉ lệ [x, y, rộng, cao] như Frigate cần, kẹp trong 0–1."""
    x1, y1 = max(0.0, d.bbox.x1 / width), max(0.0, d.bbox.y1 / height)
    x2, y2 = min(1.0, d.bbox.x2 / width), min(1.0, d.bbox.y2 / height)
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def save_snapshot(
    image: np.ndarray, result: FrameResult, incident: Incident, folder: str | Path,
    found: list[Detection] | None = None,
) -> str | None:
    """Lưu khung hình lúc sự cố kèm hộp khoanh bằng chứng. Lỗi thì trả về None, không ném."""
    import cv2

    img = image.copy()
    for d in found if found is not None else evidence(result, incident):
        x1, y1, x2, y2 = (int(v) for v in d.bbox.to_list())
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)

    text = f"{incident.label.value}  {incident.sub_label or ''}  {incident.score:.2f}".strip()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), RED, -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    path = Path(folder) / snapshot_name(incident)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), img):
            raise OSError("cv2.imwrite trả về False")
    except OSError as e:
        log.error("Không lưu được ảnh sự cố %s: %s", path, e)
        return None
    return path.as_posix()


class IncidentProcessor:
    """Chạy cả hai luật trên mỗi khung, chụp ảnh và gửi sự cố đi."""

    def __init__(
        self,
        camera: str,
        publisher=None,
        snapshot_dir: str | Path | None = "snapshots",
        ppe: PPEConfig | None = None,
        fire: FireConfig | None = None,
    ):
        self.camera = camera
        self.publisher = publisher
        self.snapshot_dir = snapshot_dir
        self.ppe = PPERule(camera, ppe or PPEConfig())
        self.fire = FireRule(camera, fire or FireConfig())

    def handle(self, image: np.ndarray, result: FrameResult) -> list[Incident]:
        incidents = self.ppe.update(result) + self.fire.update(result)
        for incident in incidents:
            found = evidence(result, incident, skip=lambda d: self.fire._excluded(d, result))
            if incident.label in _EVIDENCE:
                # Chỉ khoanh trong Frigate cho cháy/khói. Frigate không nhận ảnh của module mà
                # tự chụp khung đang có lúc nhận sự cố, khung đó đi trước khung của module
                # ~0,3 giây (độ trễ restream, đo 20/10) cộng thời gian xử lý: người đang đi đã
                # dịch chỗ nên hộp rơi vào khoảng trống. Lửa, khói đứng yên nên vẫn khoanh đúng.
                incident.boxes = tuple(relative_box(d, result.width, result.height) for d in found)
            if self.snapshot_dir is not None:
                incident.snapshot = save_snapshot(image, result, incident, self.snapshot_dir, found)
            if self.publisher is not None:
                self.publisher.submit(incident)
        return incidents
