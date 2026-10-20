"""Kiểm thử cách safety.cli báo nhịp xử lý.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""

from safety.cli import WARMUP_FRAMES, _rate


def test_khung_khoi_dong_gpu_khong_keo_nhip_xuong():
    """Khung đầu mất 3 giây vì GPU khởi động; sau đó đều 5 khung/giây."""
    frame_at = [0.0] + [3.0 + 0.2 * i for i in range(40)]
    assert "5.00 khung/giây" in _rate(frame_at)
    # Cách cũ: chia tổng số khung cho tổng thời gian.
    assert len(frame_at) / (frame_at[-1] - frame_at[0] + 0.2) < 4.0


def test_chay_qua_ngan_thi_noi_ro_chua_do_duoc():
    assert "chưa đủ" in _rate([0.0] * WARMUP_FRAMES)
