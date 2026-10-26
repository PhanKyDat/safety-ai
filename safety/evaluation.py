"""Chấm module trên video có đáp án: F1 theo sự cố, AUROC/AUPRC theo đoạn video.

Đáp án là một tệp CSV đi kèm mỗi video (cùng tên, đuôi .csv), mỗi dòng một khoảng
có vi phạm:

    nhan,bat_dau,ket_thuc,ghi_chu
    no_helmet,5,20,nguoi ao xanh
    fall,0:32,0:40,nga tren nem

Thời gian ghi bằng giây hoặc phút:giây. Video không có vi phạm thì tệp chỉ có dòng tiêu đề.

Hai mức chấm, vì hai câu hỏi khác nhau:

- Mức sự cố (F1): luật có báo đúng lúc không. Sự cố báo trong khoảng vi phạm (cộng
  thêm `tolerance` giây sau khi khoảng kết thúc, vì luật cần 3 giây mới báo) là báo
  đúng; mỗi khoảng chỉ tính một lần đúng, báo thêm trong cùng khoảng là "báo lặp".
  Báo ngoài mọi khoảng là báo nhầm, khoảng không có sự cố nào là bỏ sót.
- Mức đoạn video (AUROC, AUPRC): mô hình phân biệt đoạn có vi phạm với đoạn không có
  tốt tới đâu, không phụ thuộc một ngưỡng nào. AUROC không dùng được cho phát hiện vật
  thể theo khung (không đếm được "đúng là không có vật"), nhưng dùng được ở đây vì
  đoạn video có cả hai loại.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from safety.contracts import DetectClass, FrameResult, IncidentLabel
from safety.rules import PPERule, _CHECKS, body_tilt


@dataclass(frozen=True)
class GtEvent:
    label: str
    start: float
    end: float
    note: str = ""


def parse_time(value: str) -> float:
    """'12.5' -> 12.5 giây; '1:05' -> 65 giây."""
    value = value.strip()
    if ":" in value:
        minutes, seconds = value.split(":", 1)
        return int(minutes) * 60 + float(seconds)
    return float(value)


def load_gt(path: str | Path) -> list[GtEvent]:
    known = {label.value for label in IncidentLabel}
    events = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            label = (row.get("nhan") or "").strip()
            if not label:
                continue
            if label not in known:
                raise ValueError(f"{path}:{line_no}: nhãn '{label}' không có trong {sorted(known)}")
            start, end = parse_time(row["bat_dau"]), parse_time(row["ket_thuc"])
            if end <= start:
                raise ValueError(f"{path}:{line_no}: ket_thuc phải sau bat_dau")
            events.append(GtEvent(label, start, end, (row.get("ghi_chu") or "").strip()))
    return events


@dataclass
class LabelScore:
    gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    repeats: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.gt if self.gt else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, other: LabelScore) -> None:
        self.gt += other.gt
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.repeats += other.repeats
        self.latencies += other.latencies


def match_events(gt: list[GtEvent], alerts: list[tuple[str, float]],
                 tolerance: float = 5.0) -> dict[str, LabelScore]:
    """Ghép sự cố module báo (nhãn, giây trong video) với đáp án, theo từng nhãn."""
    scores: dict[str, LabelScore] = {}
    matched: set[int] = set()
    for i, ev in enumerate(gt):
        scores.setdefault(ev.label, LabelScore()).gt += 1
    for label, ts in sorted(alerts, key=lambda a: a[1]):
        score = scores.setdefault(label, LabelScore())
        hit = None
        for i, ev in enumerate(gt):
            if ev.label == label and ev.start <= ts <= ev.end + tolerance:
                hit = i
                if i not in matched:
                    break
        if hit is None:
            score.fp += 1
        elif hit in matched:
            score.repeats += 1
        else:
            matched.add(hit)
            score.tp += 1
            score.latencies.append(ts - gt[hit].start)
    for i, ev in enumerate(gt):
        if i not in matched:
            scores[ev.label].fn += 1
    return scores


def window_truth(gt: list[GtEvent], label: str, duration: float, window: float) -> list[int]:
    """1 nếu đoạn [k*window, (k+1)*window) chồng lên một khoảng vi phạm của nhãn này."""
    n = max(1, math.ceil(duration / window))
    out = []
    for k in range(n):
        a, b = k * window, (k + 1) * window
        out.append(int(any(ev.label == label and ev.start < b and ev.end > a for ev in gt)))
    return out


def frame_scores(result: FrameResult, ppe: PPERule) -> dict[str, float]:
    """Điểm "có vi phạm" của một khung cho từng nhãn, để vẽ ROC/PR theo đoạn.

    Đồ bảo hộ: điểm tin cậy cao nhất của lớp vi phạm mà luật gán được cho một người
    (đúng vùng đầu / khung người như luật xét). Cháy, khói: điểm cao nhất của lớp đó.
    Ngã: góc nghiêng của người nằm nhất, chia 90 (thấy chân thì lấy góc nhỏ hơn giữa
    thân và chân, giống luật đòi cả hai cùng nghiêng).
    """
    out = {label.value: 0.0 for label in IncidentLabel if label is not IncidentLabel.DANGER_ZONE}
    for d in result.detections:
        if d.cls is DetectClass.FIRE:
            out["fire"] = max(out["fire"], d.conf)
        elif d.cls is DetectClass.SMOKE:
            out["smoke"] = max(out["smoke"], d.conf)
        elif d.cls is DetectClass.PERSON:
            for label, ok_cls, bad_cls, head_only in _CHECKS:
                bad = ppe._violating(d, result, ok_cls, bad_cls, head_only)
                if bad not in (None, False):
                    out[label.value] = max(out[label.value], bad.conf)
            torso, legs = body_tilt(d)
            if torso is not None:
                angle = min(torso, legs) if legs is not None else torso
                out["fall"] = max(out["fall"], angle / 90.0)
    return out


def auroc(truth: list[int], scores: list[float]) -> float | None:
    """Diện tích dưới đường ROC = xác suất một đoạn có vi phạm được chấm điểm cao hơn
    một đoạn không có (Mann–Whitney), điểm bằng nhau tính nửa."""
    pos = [s for t, s in zip(truth, scores) if t]
    neg = [s for t, s in zip(truth, scores) if not t]
    if not pos or not neg:
        return None
    ranked = sorted((s, t) for t, s in zip(truth, scores))
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(t for _, t in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def average_precision(truth: list[int], scores: list[float]) -> float | None:
    """AUPRC theo kiểu average precision: trung bình precision tại mỗi đoạn đúng, xếp
    theo điểm giảm dần (các đoạn cùng điểm xét một lượt)."""
    n_pos = sum(truth)
    if not n_pos:
        return None
    pairs = sorted(zip(scores, truth), key=lambda x: -x[0])
    ap, tp, seen, i = 0.0, 0, 0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        new_tp = sum(t for _, t in pairs[i:j])
        tp += new_tp
        seen = j
        ap += new_tp / n_pos * (tp / seen)
        i = j
    return ap


def roc_points(truth: list[int], scores: list[float]) -> list[tuple[float, float]]:
    """(tỉ lệ báo nhầm, tỉ lệ bắt được) khi hạ dần ngưỡng, để vẽ."""
    n_pos, n_neg = sum(truth), len(truth) - sum(truth)
    pts = [(0.0, 0.0)]
    if not n_pos or not n_neg:
        return pts
    for thr in sorted(set(scores), reverse=True):
        tp = sum(1 for t, s in zip(truth, scores) if t and s >= thr)
        fp = sum(1 for t, s in zip(truth, scores) if not t and s >= thr)
        pts.append((fp / n_neg, tp / n_pos))
    return pts


def pr_points(truth: list[int], scores: list[float]) -> list[tuple[float, float]]:
    """(recall, precision) khi hạ dần ngưỡng, để vẽ."""
    n_pos = sum(truth)
    pts = []
    if not n_pos:
        return pts
    for thr in sorted(set(scores), reverse=True):
        tp = sum(1 for t, s in zip(truth, scores) if t and s >= thr)
        fp = sum(1 for t, s in zip(truth, scores) if not t and s >= thr)
        pts.append((tp / n_pos, tp / (tp + fp) if tp + fp else 1.0))
    return pts
