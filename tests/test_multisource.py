"""Kiểm thử gom nhiều camera thành lô (MultiSource) và cách đọc --cam của safety.cli.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import cv2
import numpy as np
import pytest

from safety.cli import build_parser, camera_sources
from safety.stream import MultiSource


def make_video(path, n_frames, fps=5.0, value=0):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n_frames):
        out.write(np.full((48, 64, 3), (value + i * 10) % 256, dtype=np.uint8))
    out.release()
    return str(path)


def test_tep_video_doc_song_song_tep_ngan_het_thi_vang_mat(tmp_path):
    a = make_video(tmp_path / "a.mp4", 10)
    b = make_video(tmp_path / "b.mp4", 4)
    batches = list(MultiSource({"a": a, "b": b}, target_fps=5.0))

    assert len(batches) == 10
    assert all([cam for cam, _, _ in bt] == ["a", "b"] for bt in batches[:4])
    assert all([cam for cam, _, _ in bt] == ["a"] for bt in batches[4:])
    # Mốc thời gian theo thời gian trong video: mỗi lô cách nhau đúng 0,2 giây.
    ts_a = [ts for bt in batches for cam, ts, _ in bt if cam == "a"]
    assert np.allclose(np.diff(ts_a), 0.2)


def test_chay_lai_tep_cho_ra_cung_ket_qua(tmp_path):
    a = make_video(tmp_path / "a.mp4", 6)
    b = make_video(tmp_path / "b.mp4", 6, value=100)

    def run():
        return [[(cam, int(f.mean())) for cam, _, f in bt] for bt in MultiSource({"a": a, "b": b})]

    assert run() == run()


def test_khong_tron_luong_truc_tiep_voi_tep():
    with pytest.raises(ValueError):
        MultiSource({"a": "rtsp://127.0.0.1:8554/a", "b": "data/samples/xuong_han.mp4"})


def test_khong_co_cam_thi_la_mot_camera_nhu_cu():
    args = build_parser().parse_args(["--source", "0", "--camera", "xuong_han"])
    assert camera_sources(args) == {"xuong_han": 0}


def test_doc_nhieu_cam():
    args = build_parser().parse_args(["--cam", "xuong_han=rtsp://h/x", "--cam", " kho = rtsp://h/k "])
    assert camera_sources(args) == {"xuong_han": "rtsp://h/x", "kho": "rtsp://h/k"}


@pytest.mark.parametrize("bad", [["--cam", "xuong_han"], ["--cam", "=rtsp://h/x"],
                                 ["--cam", "a=rtsp://h/1", "--cam", "a=rtsp://h/2"]])
def test_cam_sai_dang_hoac_trung_ten_thi_dung_ngay(bad):
    with pytest.raises(SystemExit):
        camera_sources(build_parser().parse_args(bad))
