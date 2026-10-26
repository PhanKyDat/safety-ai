"""Chạy module trên một hay nhiều luồng: python -m safety.cli --source 0

Mỗi khung đi qua nhận diện + tracking, rồi qua luật đồ bảo hộ và luật cháy/khói.
Sự cố được chụp ảnh vào thư mục snapshots/. Có --frigate thì đẩy sang Frigate để
xem lại kèm đoạn ghi hình; có --telegram thì gửi tiếp vào nhóm Telegram chung
(mã bot và chat id lấy từ .env); có --db thì ghi vào bảng incidents cho trang
thống kê đọc:

    python -m safety.cli --source rtsp://127.0.0.1:8554/xuong_han \\
        --camera xuong_han --frigate http://127.0.0.1:5000/api --telegram --db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from safety.contracts import FrameResult
from safety.db import DEFAULT_DB
from safety.incidents import IncidentProcessor
from safety.outputs import build_outputs
from safety.pipeline import SafetyPipeline
from safety.stream import DEFAULT_FPS, MultiSource

log = logging.getLogger("safety.cli")


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def camera_sources(args: argparse.Namespace) -> dict[str, str | int]:
    """--cam TEN=NGUON (lặp lại được); không có --cam thì là một camera --camera/--source."""
    if not args.cam:
        return {args.camera: parse_source(args.source)}
    cams: dict[str, str | int] = {}
    for item in args.cam:
        name, sep, source = item.partition("=")
        name, source = name.strip(), source.strip()
        if not sep or not name or not source:
            raise SystemExit(f"--cam phải có dạng TEN=NGUON, nhận được '{item}'")
        if name in cams:
            raise SystemExit(f"Tên camera '{name}' bị lặp trong --cam")
        cams[name] = parse_source(source)
    return cams


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Đọc luồng + tracking + luật, đẩy sự cố sang Frigate")
    p.add_argument("--source", default="0", help="RTSP URL, đường dẫn file, hoặc số hiệu webcam")
    p.add_argument("--camera", default="xuong_han", help="Tên camera, phải trùng tên trong Frigate")
    p.add_argument("--cam", action="append", default=[], metavar="TEN=NGUON",
                   help="Thêm một camera, vd xuong_han=rtsp://127.0.0.1:8554/xuong_han (lặp lại được; "
                        "có --cam thì bỏ qua --camera/--source)")
    p.add_argument("--model", default="yolo26n-pose.pt",
                   help="Mô hình chính có bám vết. Mặc định là mô hình tư thế để luật ngã chạy được")
    p.add_argument("--extra-model", action="append", default=[], metavar="PATH",
                   help="Mô hình phụ chạy song song, không bám vết, vd models/fire_smoke_v0.pt (lặp lại được)")
    p.add_argument("--device", default="0", help="0 = GPU đầu tiên, cpu = chạy CPU")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-frames", type=int, default=0, help="Số lượt (mỗi lượt một khung mỗi camera), 0 = chạy mãi")
    p.add_argument("--json", action="store_true", help="In FrameResult dạng JSON mỗi dòng")
    p.add_argument("--show", action="store_true", help="Mở cửa sổ xem trực tiếp, nhấn Q để dừng sớm")
    p.add_argument("--frigate", default=None,
                   help="Địa chỉ API Frigate, vd http://127.0.0.1:5000/api. Bỏ trống = không gửi")
    p.add_argument("--snapshots", default="snapshots", help="Thư mục lưu ảnh sự cố")
    p.add_argument("--telegram", action="store_true",
                   help="Gửi sự cố vào nhóm Telegram chung (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID trong .env)")
    p.add_argument("--db", nargs="?", const="", default=None, metavar="PATH",
                   help=f"Ghi sự cố vào bảng incidents. Bỏ trống PATH = INCIDENT_DB trong .env, "
                        f"không có thì {DEFAULT_DB} (cùng tệp cầu nối MQTT ghi)")
    return p


def db_path(arg: str | None) -> str | None:
    """--db không kèm đường dẫn thì dùng chung tệp với cầu nối MQTT và trang thống kê."""
    if arg is None:
        return None
    if arg:
        return arg
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get("INCIDENT_DB", "").strip() or DEFAULT_DB


def draw(frame, result, title: str = "safety-ai"):
    import cv2

    for d in result.detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox.to_list())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (76, 175, 80), 2)
        text = f"{d.cls.value}#{d.track_id} {d.conf:.2f}"
        cv2.putText(frame, text, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (76, 175, 80), 1, cv2.LINE_AA)
    cv2.imshow(title, frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cams = camera_sources(args)
    multi = len(cams) > 1

    pipeline = SafetyPipeline(
        extra_models=args.extra_model,
        model=args.model,
        camera=next(iter(cams)),
        device=args.device,
        conf=args.conf,
    )
    if getattr(pipeline.model, "task", None) != "pose":
        log.warning("Mô hình %s không có điểm khớp: luật ngã sẽ không báo gì "
                    "(dùng --model yolo26n-pose.pt)", args.model)
    source = MultiSource(cams, target_fps=args.fps)
    outputs = build_outputs(args.frigate, list(cams), telegram=args.telegram,
                            db_path=db_path(args.db))
    processors = {cam: IncidentProcessor(cam, publisher=outputs, snapshot_dir=args.snapshots)
                  for cam in cams}
    log.info("Chạy %d camera: %s", len(cams), ", ".join(f"{c}={u}" for c, u in cams.items()))

    seen_tracks: dict[str, set[int]] = {cam: set() for cam in cams}
    frames = {cam: 0 for cam in cams}
    rounds = 0
    incidents = 0
    started = 0.0
    infer_ms: list[float] = []
    frame_at: list[float] = []

    try:
        for batch in source:
            if not rounds:
                started = time.time()
            frame_at.append(time.time())
            t0 = time.perf_counter()
            results = pipeline.process_batch(batch)
            infer_ms.append(1000 * (time.perf_counter() - t0))
            rounds += 1

            stop = False
            for (cam, _, frame), result in zip(batch, results):
                frames[cam] += 1
                seen_tracks[cam].update(d.track_id for d in result.detections if d.track_id is not None)

                if args.json:
                    print(json.dumps(result.to_dict(), ensure_ascii=False))
                else:
                    print(_summarise(frames[cam], result, cam if multi else None))

                for incident in processors[cam].handle(frame, result):
                    incidents += 1
                    log.warning("SỰ CỐ %s/%s [%s] %s score %.2f, ảnh %s", cam, incident.label.value,
                                incident.severity.vi, incident.sub_label or "", incident.score,
                                incident.snapshot)

                if args.show and draw(frame, result, cam):
                    stop = True
            if stop or (args.max_frames and rounds >= args.max_frames):
                break
    except KeyboardInterrupt:
        print()
    finally:
        if args.show:
            import cv2

            cv2.destroyAllWindows()
        source.close()
        outputs.close()

    elapsed = time.time() - started if rounds else 0.0
    for cam in cams:
        print(f"[{cam}] {frames[cam]} khung, {len(seen_tracks[cam])} track_id khác nhau: "
              f"{sorted(seen_tracks[cam])}")
    print(f"Đã xử lý {rounds} lượt ({len(cams)} camera) trong {elapsed:.1f}s")
    print(_rate(frame_at))
    print(_speed(infer_ms, per="lô" if multi else "khung"))
    print(f"{incidents} sự cố; {outputs.summary()}")
    return 0


# Vài khung đầu chậm hơn hẳn vì GPU còn khởi động (khung đầu tiên ~2 giây).
# Tính chung vào trung bình thì một lần chạy 40 khung ra ~75 ms thay vì ~20 ms:
# chính lỗi này từng làm nhóm tin con số 71 ms/khung ở tuần 2.
WARMUP_FRAMES = 5


def _rate(frame_at: list[float]) -> str:
    """Nhịp xử lý ổn định, bỏ các khung khởi động GPU.

    Chia tổng số khung cho tổng thời gian thì khung đầu (~3 giây) kéo con số xuống:
    cùng một hệ thống, chạy 40 khung báo 3,6 FPS, chạy 150 khung báo 4,5 FPS, trong
    khi nhịp thật là 4,8. Đó là nguồn gốc con số "chỉ đạt 3,4–4,3 FPS khi chạy cùng
    Frigate" ở tuần 2–5.
    """
    steady = frame_at[WARMUP_FRAMES:]
    if len(steady) < 2:
        return f"Nhịp xử lý: mới có {len(frame_at)} khung, chưa đủ để đo nhịp ổn định"
    span = steady[-1] - steady[0]
    return (f"Nhịp xử lý: {(len(steady) - 1) / span:.2f} khung/giây mỗi camera "
            f"(bỏ {WARMUP_FRAMES} khung đầu, đo trên {span:.1f}s)")


def _speed(infer_ms: list[float], per: str = "khung") -> str:
    if not infer_ms:
        return "Chưa xử lý khung nào"
    steady = sorted(infer_ms[WARMUP_FRAMES:])
    if not steady:
        return f"Suy luận: mới có {len(infer_ms)} khung, toàn là khung khởi động GPU, chưa đo được tốc độ ổn định"
    return (f"Suy luận: khung đầu {infer_ms[0]:.0f} ms (GPU khởi động); "
            f"bỏ {WARMUP_FRAMES} {per} đầu thì trung vị {steady[len(steady) // 2]:.0f} ms/{per}, "
            f"95% số {per} dưới {steady[min(len(steady) - 1, int(0.95 * len(steady)))]:.0f} ms")


def _summarise(n: int, result: FrameResult, camera: str | None = None) -> str:
    head = f"[{camera}]" if camera else ""
    head += f"[{n:4d}] {time.strftime('%H:%M:%S', time.localtime(result.ts))}  "
    if not result.detections:
        return head + "(khong co gi)"
    parts = [
        f"{d.cls.value}#{d.track_id} {d.conf:.2f}" for d in result.detections
    ]
    return head + f"{len(result.detections)} vật thể: " + "  ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
