"""Vẽ khung nhận diện và bảng thông số lên khung hình cho cửa sổ --show (để quay video demo).

Mỗi người được tô theo trạng thái luật: xanh = đủ đồ bảo hộ / chưa kết luận, cam = đang đếm giờ
vi phạm (vd "thiếu mũ 1,8/3 s"), đỏ = đã báo sự cố. Dải trên cùng là thông số chạy (camera, mô
hình, khung/giây, thời gian suy luận, số người, số sự cố theo loại); dải dưới là các sự cố gần nhất.

Chữ vẽ bằng Pillow vì OpenCV không có font tiếng Việt có dấu.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

from safety.contracts import DetectClass, FrameResult, Incident, IncidentLabel
from safety.notify import LABEL_VI

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)

GREEN, ORANGE, RED = (46, 160, 67), (237, 137, 54), (220, 38, 38)
OBJ_COLORS = {
    DetectClass.HELMET: (56, 189, 248), DetectClass.VEST: (250, 204, 21),
    DetectClass.NO_HELMET: RED, DetectClass.NO_VEST: RED,
    DetectClass.FIRE: (255, 69, 0), DetectClass.SMOKE: (160, 160, 160),
}
OBJ_VI = {
    DetectClass.HELMET: "mũ", DetectClass.VEST: "áo PQ", DetectClass.NO_HELMET: "không mũ",
    DetectClass.NO_VEST: "không áo", DetectClass.FIRE: "lửa", DetectClass.SMOKE: "khói",
}
SHORT = {IncidentLabel.NO_HELMET: "thiếu mũ", IncidentLabel.NO_VEST: "thiếu áo",
         IncidentLabel.FIRE: "lửa", IncidentLabel.SMOKE: "khói", IncidentLabel.FALL: "ngã"}


def _font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _num(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


class Overlay:
    """Trạng thái hiển thị của một camera: đếm sự cố và giữ vài sự cố gần nhất."""

    def __init__(self, camera: str, model_name: str = ""):
        self.camera = camera
        self.model_name = model_name
        self.recent: deque[tuple[float, Incident]] = deque(maxlen=3)
        self.counts: Counter = Counter()
        self._fonts: dict[int, object] = {}

    def add(self, incident: Incident) -> None:
        self.recent.appendleft((time.time(), incident))
        self.counts[incident.label] += 1

    def f(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = _font(size)
        return self._fonts[size]

    def person_status(self, person, result: FrameResult, processor) -> tuple[tuple[int, int, int], str]:
        """Màu và dòng trạng thái của một người theo các luật đồ bảo hộ và ngã."""
        tid = person.track_id
        if tid is None or processor is None:
            return GREEN, ""
        parts, color = [], GREEN
        cfg = processor.ppe.config
        for label in (IncidentLabel.NO_HELMET, IncidentLabel.NO_VEST):
            s = processor.ppe.streak_of(tid, label)
            if s is None or s.since is None or s.last_hit is None:
                continue
            if s.last_alert is not None and result.ts - s.last_alert < cfg.cooldown:
                parts.append(f"{SHORT[label]}: ĐÃ BÁO")
                color = RED
            elif result.ts - s.last_hit <= cfg.gap_tolerance:
                parts.append(f"{SHORT[label]} {_num(s.last_hit - s.since)}/{_num(cfg.duration, 0)} s")
                if color is GREEN:
                    color = ORANGE
        fs = processor.fall.streak_of(tid)
        if fs is not None and fs.since is not None and fs.last_hit is not None:
            if fs.last_alert is not None and result.ts - fs.last_alert < processor.fall.config.cooldown:
                parts.append("ngã: ĐÃ BÁO")
                color = RED
            elif result.ts - fs.last_hit <= processor.fall.config.gap_tolerance:
                parts.append(f"nằm {_num(fs.last_hit - fs.since)}/{_num(processor.fall.config.duration, 0)} s")
                if color is GREEN:
                    color = ORANGE
        return color, " · ".join(parts) if parts else "đủ đồ bảo hộ"

    def render(self, frame: np.ndarray, result: FrameResult, processor=None,
               fps: float | None = None, infer_ms: float | None = None) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame[:, :, ::-1]).convert("RGBA")
        W, H = img.size
        k = max(0.6, W / 1280)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        small, mid, big = self.f(int(15 * k)), self.f(int(18 * k)), self.f(int(21 * k))

        people = [x for x in result.detections if x.cls is DetectClass.PERSON]
        for o in result.detections:
            if o.cls is DetectClass.PERSON or o.cls not in OBJ_COLORS:
                continue
            x1, y1, x2, y2 = o.bbox.to_list()
            c = OBJ_COLORS[o.cls]
            d.rectangle([x1, y1, x2, y2], outline=c + (255,), width=max(2, int(2 * k)))
            d.text((x1 + 2, y2 + 1), f"{OBJ_VI[o.cls]} {_num(o.conf, 2)}", font=small, fill=c + (255,))
        for p in people:
            color, status = self.person_status(p, result, processor)
            x1, y1, x2, y2 = p.bbox.to_list()
            d.rectangle([x1, y1, x2, y2], outline=color + (255,), width=max(3, int(3 * k)))
            tag = f"#{p.track_id}  {status}" if p.track_id is not None else status
            tw = d.textlength(tag, font=mid)
            ty = max(0, y1 - mid.size - 8)
            d.rectangle([x1, ty, x1 + tw + 10, ty + mid.size + 6], fill=color + (220,))
            d.text((x1 + 5, ty + 2), tag, font=mid, fill=(255, 255, 255, 255))

        # dải thông số trên cùng
        bar = int(64 * k)
        d.rectangle([0, 0, W, bar], fill=(17, 24, 39, 200))
        line1 = f"Camera: {self.camera}"
        if self.model_name:
            line1 += f"   |   Mô hình: {self.model_name}"
        if fps is not None:
            line1 += f"   |   {_num(fps)} khung/giây"
        if infer_ms is not None:
            line1 += f"   |   suy luận {infer_ms:.0f} ms"
        line1 += f"   |   người trong khung: {len(people)}"
        total = sum(self.counts.values())
        by = ", ".join(f"{SHORT[l]} {n}" for l, n in self.counts.items()) or "chưa có"
        d.text((10, 6), line1, font=big, fill=(255, 255, 255, 255))
        d.text((10, 8 + big.size), f"Sự cố đã báo: {total}  ({by})", font=mid, fill=(253, 224, 71, 255))

        # dải sự cố gần nhất
        if self.recent:
            rows = len(self.recent) + 1
            top = H - int((rows * (mid.size + 8) + 10))
            d.rectangle([0, top, W, H], fill=(17, 24, 39, 200))
            d.text((10, top + 5), "Sự cố gần nhất:", font=mid, fill=(253, 224, 71, 255))
            for i, (t, inc) in enumerate(self.recent):
                txt = (f"{time.strftime('%H:%M:%S', time.localtime(t))}   [{inc.severity.vi}] "
                       f"{LABEL_VI.get(inc.label, inc.label.value)}   {inc.sub_label or ''}")
                d.text((10, top + 5 + (i + 1) * (mid.size + 8)), txt, font=mid,
                       fill=(255, 255, 255, 255) if i else (252, 165, 165, 255))

        out = Image.alpha_composite(img, layer).convert("RGB")
        return np.asarray(out)[:, :, ::-1].copy()
