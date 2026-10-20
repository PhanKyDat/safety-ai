"""Luật phát hiện thiếu đồ bảo hộ.

Đo thực tế trên luồng Frigate (scripts/measure_tracks.py, 180 giây): mã định danh
sống trung vị 5,1 giây và người được phát hiện ở 90% số khung trong vòng đời đó —
tracker giữ đúng người, chỉ có phát hiện ngắt quãng khi người khuất sau kệ hàng.
Vì vậy luật đếm theo mốc thời gian có dung sai mất dấu, chứ không đòi thấy liên tục:
đòi liên tục thì chỉ 21% số đoạn đạt nổi 3 giây và luật gần như không bao giờ kích hoạt.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from safety.contracts import (
    DEFAULT_SEVERITY,
    BBox,
    Detection,
    DetectClass,
    FrameResult,
    Incident,
    IncidentLabel,
)


@dataclass(frozen=True)
class PPEConfig:
    duration: float = 3.0
    gap_tolerance: float = 1.0
    cooldown: float = 60.0
    head_ratio: float = 0.25
    min_inside: float = 0.5


# Mũ xét trong vùng đầu; áo phản quang xét trên cả khung người.
_CHECKS = (
    (IncidentLabel.NO_HELMET, DetectClass.HELMET, DetectClass.NO_HELMET, True),
    (IncidentLabel.NO_VEST, DetectClass.VEST, DetectClass.NO_VEST, False),
)


def head_region(person: BBox, ratio: float) -> BBox:
    return BBox(person.x1, person.y1, person.x2, person.y1 + person.height * ratio)


def fraction_inside(inner: BBox, outer: BBox) -> float:
    w = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    h = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    area = inner.width * inner.height
    return (w * h / area) if area > 0 else 0.0


@dataclass
class _Streak:
    since: float | None = None
    last_hit: float | None = None
    last_alert: float | None = None
    score: float = 0.0


@dataclass
class PPERule:
    """Một đối tượng cho mỗi camera: trạng thái đếm giờ gắn với track_id của camera đó."""

    camera: str
    config: PPEConfig = field(default_factory=PPEConfig)
    _streaks: dict[tuple[int, IncidentLabel], _Streak] = field(default_factory=dict, init=False)

    def streak_of(self, track_id: int, label: IncidentLabel) -> _Streak | None:
        """Trạng thái đồng hồ đếm của một người, để hiển thị hoặc gỡ lỗi."""
        return self._streaks.get((track_id, label))

    def update(self, frame: FrameResult) -> list[Incident]:
        incidents: list[Incident] = []
        for person in frame.detections:
            if person.cls is not DetectClass.PERSON or person.track_id is None:
                continue
            for label, ok_cls, bad_cls, head_only in _CHECKS:
                violating = self._violating(person, frame, ok_cls, bad_cls, head_only)
                incident = self._advance(person.track_id, label, frame.ts, violating)
                if incident is not None:
                    incidents.append(incident)
        self._prune(frame.ts)
        return incidents

    def _violating(
        self,
        person: Detection,
        frame: FrameResult,
        ok_cls: DetectClass,
        bad_cls: DetectClass,
        head_only: bool,
    ) -> Detection | None | bool:
        """Trả về detection vi phạm, False nếu có đồ bảo hộ, None nếu không kết luận được.

        Không kết luận được thì giữ nguyên đồng hồ đếm: người bị che khuất một lúc
        không phải là bằng chứng có mũ, mà cũng không nên tính là vi phạm.
        """
        region = head_region(person.bbox, self.config.head_ratio) if head_only else person.bbox
        bad = None
        for d in frame.detections:
            if d.cls not in (ok_cls, bad_cls):
                continue
            if fraction_inside(d.bbox, region) < self.config.min_inside:
                continue
            if d.cls is ok_cls:
                return False
            if bad is None or d.conf > bad.conf:
                bad = d
        return bad

    def _advance(
        self, track_id: int, label: IncidentLabel, ts: float, violating
    ) -> Incident | None:
        key = (track_id, label)
        if violating is False:
            self._streaks.pop(key, None)
            return None
        if violating is None:
            return None

        streak = self._streaks.setdefault(key, _Streak())
        if streak.since is None or (ts - streak.last_hit) > self.config.gap_tolerance:
            streak.since = ts
        streak.last_hit = ts
        streak.score = violating.conf

        if ts - streak.since < self.config.duration:
            return None
        if streak.last_alert is not None and ts - streak.last_alert < self.config.cooldown:
            return None
        streak.last_alert = ts
        return Incident(
            camera=self.camera,
            label=label,
            severity=DEFAULT_SEVERITY[label],
            score=streak.score,
            ts=ts,
            sub_label=f"ID {track_id}",
            track_id=track_id,
        )

    def _prune(self, ts: float) -> None:
        limit = self.config.cooldown + self.config.duration
        for key, streak in list(self._streaks.items()):
            if streak.last_hit is not None and ts - streak.last_hit > limit:
                del self._streaks[key]


@dataclass(frozen=True)
class ExcludeZone:
    """Vùng bỏ qua, toạ độ tỉ lệ 0–1 giống cách Frigate khai vùng.

    Dùng tỉ lệ chứ không dùng pixel để đổi độ phân giải camera vẫn đúng chỗ.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    ten: str = ""

    def to_bbox(self, width: int, height: int) -> BBox:
        return BBox(self.x1 * width, self.y1 * height, self.x2 * width, self.y2 * height)


@dataclass(frozen=True)
class FireConfig:
    window: int = 10
    min_hits: int = 6
    cooldown: float = 60.0
    min_inside: float = 0.5
    exclude: tuple[ExcludeZone, ...] = ()


@dataclass
class _Window:
    hits: deque[float]
    last_alert: float | None = None


@dataclass
class FireRule:
    """Luật cháy và khói.

    Khác luật đồ bảo hộ ở chỗ lửa không gắn với một người nào nên không đếm theo
    mã định danh. Thay vào đó xét mật độ: phải thấy ở ít nhất 6 trong 10 khung gần
    nhất mới tính. Lửa thật cháy liên tục nên dễ đạt, còn ánh đèn vàng loé lên hay
    hơi nước bay qua chỉ xuất hiện vài khung rồi mất.

    Vùng loại trừ dành cho những chỗ biết trước hay gây báo nhầm: đèn báo, cửa lò,
    ống xả hơi nước.
    """

    camera: str
    config: FireConfig = field(default_factory=FireConfig)
    _windows: dict[IncidentLabel, _Window] = field(default_factory=dict, init=False)

    def window_of(self, label: IncidentLabel) -> _Window | None:
        """Trạng thái cửa sổ của một nhãn, để hiển thị hoặc gỡ lỗi."""
        return self._windows.get(label)

    def update(self, frame: FrameResult) -> list[Incident]:
        incidents: list[Incident] = []
        for label, cls in (
            (IncidentLabel.FIRE, DetectClass.FIRE),
            (IncidentLabel.SMOKE, DetectClass.SMOKE),
        ):
            best = 0.0
            for d in frame.detections:
                if d.cls is cls and not self._excluded(d, frame):
                    best = max(best, d.conf)

            window = self._windows.setdefault(
                label, _Window(deque(maxlen=self.config.window))
            )
            window.hits.append(best)

            seen = sum(1 for c in window.hits if c > 0)
            if seen < self.config.min_hits:
                continue
            if (
                window.last_alert is not None
                and frame.ts - window.last_alert < self.config.cooldown
            ):
                continue

            window.last_alert = frame.ts
            incidents.append(
                Incident(
                    camera=self.camera,
                    label=label,
                    severity=DEFAULT_SEVERITY[label],
                    score=max(window.hits),
                    ts=frame.ts,
                    sub_label=f"{seen}/{len(window.hits)} khung gần nhất",
                )
            )
        return incidents

    def _excluded(self, detection: Detection, frame: FrameResult) -> bool:
        for zone in self.config.exclude:
            box = zone.to_bbox(frame.width, frame.height)
            if fraction_inside(detection.bbox, box) >= self.config.min_inside:
                return True
        return False


@dataclass(frozen=True)
class FallConfig:
    lying_angle: float = 60.0
    leg_angle: float = 45.0
    kp_conf: float = 0.3
    duration: float = 3.0
    gap_tolerance: float = 1.0
    cooldown: float = 60.0
    min_inside: float = 0.5
    exclude: tuple[ExcludeZone, ...] = ()


def _tilt(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    """Góc lệch khỏi phương thẳng đứng, độ: 0 là đứng thẳng, 90 là nằm ngang."""
    dx, dy = abs(top[0] - bottom[0]), abs(bottom[1] - top[1])
    return math.degrees(math.atan2(dx, dy))


def _mid(points, idx: tuple[int, int], min_conf: float) -> tuple[float, float] | None:
    a, b = points[idx[0]], points[idx[1]]
    if a.conf < min_conf or b.conf < min_conf:
        return None
    return (a.x + b.x) / 2, (a.y + b.y) / 2


def body_tilt(person: Detection, min_conf: float = 0.3) -> tuple[float | None, float | None]:
    """(góc thân, góc chân) theo điểm khớp COCO. Không thấy rõ thì trả None cho phần đó.

    Thân: giữa hai vai -> giữa hai hông. Chân: giữa hai hông -> giữa hai cổ chân,
    không thấy cổ chân thì lấy đầu gối.
    """
    k = person.keypoints
    if not k or len(k) < 17:
        return None, None
    shoulders = _mid(k, (5, 6), min_conf)
    hips = _mid(k, (11, 12), min_conf)
    if shoulders is None or hips is None:
        return None, None
    feet = _mid(k, (15, 16), min_conf) or _mid(k, (13, 14), min_conf)
    return _tilt(shoulders, hips), (_tilt(hips, feet) if feet is not None else None)


@dataclass
class FallRule:
    """Luật người ngã / nằm: thân nằm ngang liên tục `duration` giây thì báo.

    Dùng mô hình tư thế có sẵn (yolo26n-pose), không train. Đo trên video nhà kho
    (20/10, 98 lần thấy người đi đứng): góc thân trung vị 13°, lớn nhất 41°, nên
    ngưỡng nằm là 60°. Không dựa vào tỉ lệ khung người: người đi bị kệ che một phần
    đã có khung rộng gấp 1,78 lần cao.

    Người cúi nhặt đồ cũng có thân gần nằm ngang, nhưng chân vẫn đứng. Vì vậy thấy
    được chân thì chân cũng phải nghiêng quá `leg_angle` mới tính là nằm.

    Không thấy rõ vai và hông (30% số lần trên video thử) thì chưa kết luận: giữ
    nguyên đồng hồ, giống luật đồ bảo hộ.

    Giới hạn: người nằm dọc theo hướng nhìn của camera trông như đứng (bị co ngắn),
    luật không bắt được; cần video thật để đo.
    """

    camera: str
    config: FallConfig = field(default_factory=FallConfig)
    _streaks: dict[int, _Streak] = field(default_factory=dict, init=False)

    def streak_of(self, track_id: int) -> _Streak | None:
        return self._streaks.get(track_id)

    def lying(self, person: Detection, frame: FrameResult) -> bool | None:
        """True nếu đang nằm, False nếu đứng/cúi hoặc trong vùng loại trừ, None nếu chưa rõ."""
        for zone in self.config.exclude:
            if fraction_inside(person.bbox, zone.to_bbox(frame.width, frame.height)) >= self.config.min_inside:
                return False
        torso, legs = body_tilt(person, self.config.kp_conf)
        if torso is None:
            return None
        if torso < self.config.lying_angle:
            return False
        return legs is None or legs >= self.config.leg_angle

    def update(self, frame: FrameResult) -> list[Incident]:
        incidents: list[Incident] = []
        for person in frame.detections:
            if person.cls is not DetectClass.PERSON or person.track_id is None:
                continue
            incident = self._advance(person, frame.ts, self.lying(person, frame))
            if incident is not None:
                incidents.append(incident)
        self._prune(frame.ts)
        return incidents

    def _advance(self, person: Detection, ts: float, lying: bool | None) -> Incident | None:
        track_id = person.track_id
        if lying is False:
            streak = self._streaks.get(track_id)
            if streak is not None:
                streak.since = None      # giữ last_alert để người vừa đứng dậy rồi nằm lại không báo dồn
            return None
        if lying is None:
            return None

        streak = self._streaks.setdefault(track_id, _Streak())
        if streak.since is None or (ts - streak.last_hit) > self.config.gap_tolerance:
            streak.since = ts
        streak.last_hit = ts
        streak.score = person.conf

        if ts - streak.since < self.config.duration:
            return None
        if streak.last_alert is not None and ts - streak.last_alert < self.config.cooldown:
            return None
        streak.last_alert = ts
        return Incident(
            camera=self.camera,
            label=IncidentLabel.FALL,
            severity=DEFAULT_SEVERITY[IncidentLabel.FALL],
            score=streak.score,
            ts=ts,
            sub_label=f"ID {track_id} - nằm {ts - streak.since:.0f}s",
            track_id=track_id,
        )

    def _prune(self, ts: float) -> None:
        limit = self.config.cooldown + self.config.duration
        for key, streak in list(self._streaks.items()):
            if streak.last_hit is not None and ts - streak.last_hit > limit:
                del self._streaks[key]
