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
    # Cách "vắng mặt": thấy rõ đầu (thân) mà không có mũ (áo) thì coi là vi phạm, không
    # chờ mô hình nói "không mũ". Đo 04/11 trên 7 đoạn tự quay góc cao: mô hình PPE nhận
    # được mũ và áo thật ở ~3/4 số khung nhưng chưa ra no_helmet/no_vest lần nào, nên
    # luật kiểu cũ bắt 0/6 vi phạm; cách vắng mặt bắt 6/6, 0 báo nhầm (mẫu nhỏ).
    # Giới hạn: mũ thường bị nhận là mũ bảo hộ (mũ lưỡi trai trên video kho) thì vẫn sót.
    # Bật mặc định từ 10/11: thêm 3 đoạn ban đêm có 2 người, cả bộ 12 video bắt 8/8 thiếu mũ
    # và 7/7 thiếu áo (luật kiểu cũ 0/15), 1 báo nhầm (áo vàng dưới đèn đỏ, lỗi mô hình).
    # Không có mô hình tư thế thì không có điểm khớp, cách này tự không làm gì.
    infer_missing: bool = True
    kp_conf: float = 0.5
    head_points: int = 2    # số điểm mặt/tai (mũi, 2 mắt, 2 tai) phải thấy rõ
    # Suy ra thiếu mũ còn đòi thấy rõ 2 vai và khung người đủ chắc. Chạy thử demo 06/11, mô
    # hình tư thế khoanh cái xô nước thành người (điểm 0,25–0,44) có điểm mặt ở 22/31 lần, đủ
    # 2 vai 3/31 lần, và luật báo "không mũ" hai lần (một lần lọt qua khi chỉ đòi vai). Người
    # thật ở đoạn không mũ: điểm ≥ 0,5 ở 47/50 lần, có mặt + 2 vai 36/50.
    # Thiếu áo không lọc theo điểm người: người đi sát cửa bị che chỉ còn 0,4–0,7 mà vẫn thấy
    # rõ vai, hông; còn cái xô không bao giờ có hông nên không dính trường hợp này.
    head_needs_shoulders: bool = True
    head_person_conf: float = 0.5
    # Chống báo trùng khi đổi mã người: mã định danh sống trung vị chỉ 5,1 s (đo tuần 3),
    # người khuất sau kệ rồi hiện lại thường mang mã mới và bị báo lần nữa. Sự cố mới được
    # coi là của người vừa báo nếu mã cũ đã mất không quá `reentry_window` giây và người mới
    # hiện ngay gần chỗ cũ (tâm hộp cách không quá `reentry_distance` lần cạnh dài hộp cũ,
    # thường là chiều cao người: đủ cho đi khuất sau kệ rồi ló ra phía bên kia).
    # Cố ý hẹp: không nhận diện lại được người, gộp rộng hơn thì người khác đi vào sẽ bị sót.
    reentry_window: float = 10.0
    reentry_distance: float = 1.0
    # Gộp thêm "cùng chỗ đã báo": trong thời gian chờ, người vi phạm cùng loại mà khung người
    # trùng ≥ `same_spot_iou` (IoU) với khung lúc báo của một sự cố trước thì coi là cùng sự
    # cố, dù mất mã lâu hơn `reentry_window`. Chạy 8 giờ 16–17/11 trên video giả lập lặp
    # 17 s: mỗi vòng lặp người có mã mới và xuất hiện lại ở chỗ cũ, chỉ có luật trên thì ra
    # 11.487 sự cố/5,7 giờ (~2.000/giờ) và đầy ổ. IoU 0,5 để hai người đứng cạnh nhau (khung
    # không chồng nhau) vẫn bị báo riêng. Chỉ gộp với sự cố mà mã của nó lúc này không còn
    # vi phạm cùng loại (đã mất, hoặc mã đã sang người khác đang tuân thủ khi hai người đi chéo).
    same_spot_iou: float = 0.5


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
    first_box: BBox | None = None   # khung người lúc bắt đầu đếm
    seen: int = 0                   # luật ngã: số khung thấy người từ lúc bắt đầu đếm
    hits: int = 0                   # luật ngã: trong đó số khung rõ ràng đang nằm
    by_angle: bool = False          # luật ngã: có khung nào nằm theo góc thân (không chỉ khung bè)
    boxes: list = field(default_factory=list)   # luật ngã: khung người các khung đang nằm


@dataclass
class _Alerted:
    """Một người vừa bị báo: để nhận ra khi người đó hiện lại dưới mã mới."""

    track_id: int
    label: IncidentLabel
    alert_ts: float
    last_seen: float    # lần cuối mã này còn vi phạm loại này
    box: BBox           # khung lúc đó
    alert_box: BBox     # khung lúc báo


def _iou(a: BBox, b: BBox) -> float:
    w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = w * h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _overlaps(a: BBox, b: BBox) -> bool:
    """Hai khung người cùng chỗ: chồng nhau theo chiều ngang ≥ 30% khung hẹp hơn và có chồng theo chiều dọc."""
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    return w > 0.3 * min(a.width, b.width) and min(a.y2, b.y2) > max(a.y1, b.y1)


def _near(a: BBox, b: BBox, factor: float) -> bool:
    ax, ay = (a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2
    bx, by = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    return math.hypot(ax - bx, ay - by) <= factor * max(a.width, a.height)


@dataclass
class PPERule:
    """Một đối tượng cho mỗi camera: trạng thái đếm giờ gắn với track_id của camera đó."""

    camera: str
    config: PPEConfig = field(default_factory=PPEConfig)
    _streaks: dict[tuple[int, IncidentLabel], _Streak] = field(default_factory=dict, init=False)
    _alerted: list[_Alerted] = field(default_factory=list, init=False)

    def streak_of(self, track_id: int, label: IncidentLabel) -> _Streak | None:
        """Trạng thái đồng hồ đếm của một người, để hiển thị hoặc gỡ lỗi."""
        return self._streaks.get((track_id, label))

    def update(self, frame: FrameResult) -> list[Incident]:
        incidents: list[Incident] = []
        people = [d for d in frame.detections if d.cls is DetectClass.PERSON and d.track_id is not None]
        checks = []
        busy: dict[IncidentLabel, set[int]] = {label: set() for label, *_ in _CHECKS}
        for person in people:
            for label, ok_cls, bad_cls, head_only in _CHECKS:
                violating = self._violating(person, frame, ok_cls, bad_cls, head_only)
                checks.append((person, label, violating))
                if violating is not None and violating is not False:
                    busy[label].add(person.track_id)
        for person, label, _ in checks:
            if person.track_id in busy[label]:
                for a in self._alerted:
                    if a.track_id == person.track_id and a.label is label:
                        a.last_seen, a.box = frame.ts, person.bbox
        for person, label, violating in checks:
            incident = self._advance(person, label, frame.ts, violating, busy[label])
            if incident is not None:
                incidents.append(incident)
        self._prune(frame.ts)
        return incidents

    def _same_as_alerted(self, person: Detection, label: IncidentLabel, ts: float,
                         busy: set[int], first_box: BBox | None = None) -> _Alerted | None:
        """Sự cố trước (cùng loại, còn trong thời gian chờ, mã của nó không còn vi phạm) mà
        người này rất có thể chính là người đó:
        - mất mã chưa lâu và hiện lại gần chỗ cuối cùng thấy (lúc bắt đầu đếm hoặc bây giờ), hoặc
        - đang đứng đúng chỗ đã bị báo (khung trùng ≥ `same_spot_iou`)."""
        cfg = self.config
        if cfg.reentry_window <= 0 and cfg.same_spot_iou <= 0:
            return None
        for a in self._alerted:
            if a.label is not label or a.track_id in busy or ts - a.alert_ts >= cfg.cooldown:
                continue
            if cfg.reentry_window > 0 and ts - a.last_seen <= cfg.reentry_window and (
                    _near(a.box, person.bbox, cfg.reentry_distance)
                    or (first_box is not None and _near(a.box, first_box, cfg.reentry_distance))):
                return a
            if cfg.same_spot_iou > 0 and _iou(a.alert_box, person.bbox) >= cfg.same_spot_iou:
                return a
        return None

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

        Bật `infer_missing` thì khi không thấy lớp nào, người được thấy rõ (theo điểm
        khớp của mô hình tư thế) cũng tính là vi phạm, trả về chính detection người.
        """
        region = head_region(person.bbox, self.config.head_ratio) if head_only else person.bbox
        bad = None
        for d in frame.detections:
            if d.cls not in (ok_cls, bad_cls):
                continue
            if fraction_inside(d.bbox, region) < self.config.min_inside and not (
                head_only and self._covers_face(d.bbox, person)
            ):
                continue
            if d.cls is ok_cls:
                return False
            if bad is None or d.conf > bad.conf:
                bad = d
        if bad is None and self.config.infer_missing and self._clearly_visible(person, head_only):
            return person
        return bad

    def _covers_face(self, box: BBox, person: Detection) -> bool:
        """Mũ nằm đè lên điểm mặt/tai đủ tin cậy của người này.

        Bổ sung cho cách xét "1/4 trên của khung người": người đứng sát mép dưới khung hình
        bị cắt mất nửa thân, khung người ngắn lại nên vùng đầu quá nhỏ và mũ thật bị bỏ ra
        ngoài (video ban đêm 10/11: báo "không mũ" người đang đội mũ vàng).
        """
        k = person.keypoints
        if not k:
            return False
        return any(p.conf >= self.config.kp_conf and box.x1 <= p.x <= box.x2 and box.y1 <= p.y <= box.y2
                   for p in k[:5])

    def _clearly_visible(self, person: Detection, head_only: bool) -> bool:
        """Thấy rõ đầu (đủ điểm mặt/tai) hoặc thân (2 vai + 2 hông). Không có điểm khớp thì không."""
        k = person.keypoints
        if not k or len(k) < 13:
            return False
        c = self.config.kp_conf
        if head_only:
            if person.conf < self.config.head_person_conf:
                return False
            if self.config.head_needs_shoulders and not (k[5].conf >= c and k[6].conf >= c):
                return False
            return sum(1 for p in k[:5] if p.conf >= c) >= self.config.head_points
        return all(k[i].conf >= c for i in (5, 6, 11, 12))

    def _advance(
        self, person: Detection, label: IncidentLabel, ts: float, violating,
        busy: set[int] = frozenset(),
    ) -> Incident | None:
        track_id = person.track_id
        key = (track_id, label)
        if violating is False:
            self._streaks.pop(key, None)
            return None
        if violating is None:
            return None

        streak = self._streaks.setdefault(key, _Streak())
        if streak.since is None or (ts - streak.last_hit) > self.config.gap_tolerance:
            streak.since = ts
            streak.first_box = person.bbox
        streak.last_hit = ts
        streak.score = violating.conf

        if ts - streak.since < self.config.duration:
            return None
        if streak.last_alert is not None and ts - streak.last_alert < self.config.cooldown:
            return None
        earlier = self._same_as_alerted(person, label, ts, busy, streak.first_box)
        if earlier is not None:
            # Cùng người, mã mới: nhận lại thời điểm báo cũ để cooldown chạy tiếp như một người.
            streak.last_alert = earlier.alert_ts
            earlier.track_id, earlier.last_seen, earlier.box = track_id, ts, person.bbox
            return None
        streak.last_alert = ts
        self._alerted.append(_Alerted(track_id, label, ts, ts, person.bbox, person.bbox))
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
        self._alerted = [a for a in self._alerted if ts - a.alert_ts < self.config.cooldown]


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
    # Điều kiện thứ hai, theo kiểu "hoặc": khung người bè ngang (rộng/cao ≥ wide_ratio) và
    # thấy rõ 2 vai + 2 hông (≥ wide_kp_conf). Camera góc cao nhìn chéo xuống làm người nằm bị
    # co ngắn: 3 video ngã của nhóm (Uy quay 16/11) có góc thân chỉ 31°/3°/40° nên luật góc sót
    # cả 3, trong khi khung người lúc nằm rộng/cao ≥ 0,57, lúc đứng ≤ 0,43. Đòi thấy rõ vai và
    # hông để loại người đi bị kệ che (khung bè ra nhưng mất nửa thân). 0 = tắt.
    wide_ratio: float = 0.7
    wide_kp_conf: float = 0.5
    # Trong `duration` giây phải có ít nhất tỉ lệ này số khung thấy người rõ ràng đang nằm;
    # khung "chưa rõ" không còn được tính như đang nằm. Người ngã nằm yên nên gần như mọi
    # khung đều đạt (3 video ngã: 36–46 khung liền); người cởi áo / dang tay chỉ lác đác 1–5
    # khung bè xen giữa các khung không rõ vai hông, trước đây vẫn cộng đủ 3 s (13 báo nhầm
    # trên 12 video của nhóm, 17/11). 0 = như cũ.
    min_lying_ratio: float = 0.8
    # Khung bè thôi chưa đủ: ở góc camera cao, người đứng sát mép hay ngay dưới camera cũng có
    # khung bè, thân và chân lệch 3–30° y như người ngã nhìn chéo (13 báo nhầm, 17/11). Nên khi
    # chỉ nhờ điều kiện khung bè, còn đòi: (1) nằm yên — tâm khung xê dịch trong lúc đếm không
    # quá `wide_max_motion` lần chiều cao khung (3 video ngã 0,03–0,04; báo nhầm 0,19–1,07);
    # (2) đã tụt chiều cao — cao khung bây giờ không quá `wide_max_height_ratio` lần khung người
    # cao nhất ở cùng chỗ trong `wide_history` giây trước lúc bắt đầu đếm (ngã 0,37–0,66; báo
    # nhầm 0,73–1,44 hoặc trước đó không ai đứng ở chỗ đó). Xét theo chỗ chứ không theo mã vì
    # người ngã hay bị đổi mã lúc chạm sàn.
    wide_max_motion: float = 0.10
    wide_max_height_ratio: float = 0.75
    wide_history: float = 6.0
    edge_margin: float = 0.01     # khung cách mép ảnh dưới 1% coi như bị cắt, không xét khung bè


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

    Người nằm dọc theo hướng nhìn của camera trông như đứng (bị co ngắn) nên góc thân
    không bắt được; từ 17/11 thêm điều kiện "khung bè ngang" (FallConfig.wide_ratio) cho
    trường hợp này và cho camera góc cao.
    """

    camera: str
    config: FallConfig = field(default_factory=FallConfig)
    _streaks: dict[int, _Streak] = field(default_factory=dict, init=False)
    _history: deque = field(default_factory=deque, init=False)   # (ts, khung) mọi người, để xét tụt chiều cao

    def streak_of(self, track_id: int) -> _Streak | None:
        return self._streaks.get(track_id)

    def lying(self, person: Detection, frame: FrameResult) -> bool | None:
        """True nếu đang nằm, False nếu đứng/cúi hoặc trong vùng loại trừ, None nếu chưa rõ."""
        return self._lying(person, frame)[0]

    def _lying(self, person: Detection, frame: FrameResult) -> tuple[bool | None, bool]:
        """(quyết định như `lying`, có phải nhờ góc thân không)."""
        for zone in self.config.exclude:
            if fraction_inside(person.bbox, zone.to_bbox(frame.width, frame.height)) >= self.config.min_inside:
                return False, False
        torso, legs = body_tilt(person, self.config.kp_conf)
        if torso is not None and torso >= self.config.lying_angle:
            # Thân ngang: chân cũng ngang (hoặc không thấy chân) là nằm; chân còn đứng là
            # đang cúi — kể cả khi khung bè ngang, nên không xét điều kiện khung bè nữa.
            return legs is None or legs >= self.config.leg_angle, True
        if self._wide(person, frame):
            return True, False
        return (None if torso is None else False), False

    def _wide_confirmed(self, streak: _Streak, since: float, box: BBox) -> bool:
        """Điều kiện thêm khi chỉ nhờ khung bè: nằm yên và đã tụt chiều cao (FallConfig)."""
        cfg = self.config
        if not streak.boxes or box.height <= 0:
            return False
        cx = [(b.x1 + b.x2) / 2 for b in streak.boxes]
        cy = [(b.y1 + b.y2) / 2 for b in streak.boxes]
        if (max(cx) - min(cx) + max(cy) - min(cy)) > cfg.wide_max_motion * box.height:
            return False
        before = [b.height for t, b in self._history
                  if since - cfg.wide_history <= t < since and _overlaps(b, box)]
        return bool(before) and box.height <= cfg.wide_max_height_ratio * max(before)

    def _wide(self, person: Detection, frame: FrameResult) -> bool:
        """Khung người bè ngang và thấy rõ cả 2 vai, 2 hông (xem FallConfig.wide_ratio).

        Khung chạm mép ảnh thì bỏ: người đứng sát mép bị cắt mất nửa dưới nên khung thấp
        và bè y như người vừa ngã (báo nhầm còn lại trên video của nhóm, 17/11)."""
        cfg = self.config
        box = person.bbox
        if cfg.wide_ratio <= 0 or box.height <= 0:
            return False
        mx, my = cfg.edge_margin * frame.width, cfg.edge_margin * frame.height
        if box.x1 <= mx or box.y1 <= my or box.x2 >= frame.width - mx or box.y2 >= frame.height - my:
            return False
        k = person.keypoints
        if not k or len(k) < 13 or any(k[i].conf < cfg.wide_kp_conf for i in (5, 6, 11, 12)):
            return False
        return person.bbox.width / person.bbox.height >= cfg.wide_ratio

    def update(self, frame: FrameResult) -> list[Incident]:
        incidents: list[Incident] = []
        people = [d for d in frame.detections if d.cls is DetectClass.PERSON and d.track_id is not None]
        for person in people:
            lying, by_angle = self._lying(person, frame)
            incident = self._advance(person, frame.ts, lying, by_angle)
            if incident is not None:
                incidents.append(incident)
        self._history.extend((frame.ts, p.bbox) for p in people)
        self._prune(frame.ts)
        return incidents

    def _advance(self, person: Detection, ts: float, lying: bool | None,
                 by_angle: bool = True) -> Incident | None:
        track_id = person.track_id
        if lying is False:
            streak = self._streaks.get(track_id)
            if streak is not None:
                streak.since = None      # giữ last_alert để người vừa đứng dậy rồi nằm lại không báo dồn
            return None
        if lying is None:
            streak = self._streaks.get(track_id)
            if streak is not None and streak.since is not None:
                streak.seen += 1
            return None

        streak = self._streaks.setdefault(track_id, _Streak())
        if streak.since is None or (ts - streak.last_hit) > self.config.gap_tolerance:
            streak.since = ts
            streak.seen = streak.hits = 0
            streak.by_angle = False
            streak.boxes = []
        streak.last_hit = ts
        streak.score = person.conf
        streak.seen += 1
        streak.hits += 1
        streak.by_angle = streak.by_angle or by_angle
        streak.boxes.append(person.bbox)
        if len(streak.boxes) > 600:                     # nằm rất lâu: giữ ~2 phút gần nhất
            del streak.boxes[:-600]

        if ts - streak.since < self.config.duration:
            return None
        if streak.hits < self.config.min_lying_ratio * streak.seen:
            return None
        if not streak.by_angle and not self._wide_confirmed(streak, streak.since, person.bbox):
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
        keep = self.config.wide_history + self.config.duration + 60
        while self._history and ts - self._history[0][0] > keep:
            self._history.popleft()
