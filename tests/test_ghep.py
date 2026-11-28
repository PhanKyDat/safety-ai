"""--ghep: ghép khung của nhiều camera vào một cửa sổ lưới."""
import numpy as np

from safety.cli import mosaic


def test_ghep_4_camera_thanh_luoi_2x2():
    imgs = [np.full((720, 1280, 3), i * 60, np.uint8) for i in range(4)]
    out = mosaic(imgs, cell=(320, 180))
    assert out.shape == (360, 640, 3)
    assert out[0, 0, 0] == 0 and out[0, 639, 0] == 60 and out[359, 0, 0] == 120 and out[359, 639, 0] == 180


def test_ghep_3_camera_o_cuoi_de_den():
    out = mosaic([np.full((100, 100, 3), 255, np.uint8)] * 3, cell=(50, 50))
    assert out.shape == (100, 100, 3) and out[99, 99, 0] == 0


def test_cua_so_vua_man_hinh_giu_ti_le():
    from safety.cli import fit

    assert fit((1920, 1080), (1536, 864)) == (1382, 777)      # laptop để tỉ lệ 125%
    assert fit((1280, 720), (2560, 1440)) == (1280, 720)      # màn hình lớn: không phóng to
