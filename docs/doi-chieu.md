# Đối chiếu: Frigate thuần và Frigate + module của nhóm

Mục 6 của kế hoạch: module của nhóm thêm được gì so với Frigate dùng một mình, và tốn bao nhiêu
tài nguyên. Tài liệu này gom mọi số đo về một chỗ để cắt thẳng vào báo cáo cuối kỳ; ô nào ghi
**chưa đo** là còn chờ, kèm người phụ trách.

Cập nhật lần cuối: 17/11 (tuần 11): B1 đã đo, B4 có module, lần chạy dài thứ nhất.

## 1. So sánh tính năng (mục 6.1)

| Tiêu chí | Frigate thuần | Frigate + module nhóm | Bằng chứng |
|---|---|---|---|
| Quản lý camera, ghi hình, xem lại | Có | Có, giữ nguyên | Sự cố module mở được trong Frigate kèm đoạn ghi hình (M3, 04/11) |
| Vùng nguy hiểm | Có (zone) | Có, giữ nguyên; thêm Telegram + thống kê qua cầu nối MQTT | `safety/mqtt_bridge.py` |
| Không mũ / không áo phản quang | Phải tự gán nhãn, huấn luyện phân loại trên ảnh người (Object Classification) | Mô hình YOLO riêng + luật thời gian 3 s + cách vắng mặt | Mục 2 (B1) |
| Lửa / khói | Mô hình mặc định (COCO) không có lớp này | Có | Mục 3 (B2) |
| Ngã / nằm bất động | Không có | Có (mô hình tư thế + luật góc nghiêng) | Mục 3 (B2) |
| Luật mức độ, chống báo lặp | Cơ bản (theo vật thể) | Mức độ theo loại, thời gian chờ 60 s, gộp người bị đổi mã | `safety/rules.py` |
| Cảnh báo Telegram, thống kê an toàn | Không có sẵn | Có (Telegram kèm ảnh + link Frigate, trang Streamlit) | M3, 06/11 |

## 2. B1 — Không đội mũ: Object Classification của Frigate so với module

| | Frigate Object Classification (`mu_bao_ho`) | Module nhóm (cách vắng mặt, PPE v2) |
|---|---|---|
| Bắt được thiếu mũ (recall) | **2/8 khoảng vi phạm (25%)** | **6/8 (75%)**; bản PPE v1 tuần 9: 8/8 |
| Báo nhầm / precision | 0 báo nhầm, P 100% | 0 báo nhầm, P 100% |
| F1 | 40,0% | 85,7% |
| Bộ đo | 12 video: 10 đoạn tự quay góc cao (7 ban ngày 1 người, 3 ban đêm 2 người) + 2 video kho; cùng đáp án, cùng cách chấm (dung sai báo muộn 5 s) | |

Đo 17/11 trên laptop Đạt bằng `scripts/cham_b1.py` (cách dựng nằm ở đầu script): ghép 12 video
thành một, phát một lượt vào camera `b1` của Frigate (detect 640x360, 5 khung/s, bật
`mu_bao_ho`), rồi đổi giờ từng sự kiện về giây trong video và chấm bằng đúng hàm chấm của module.
Thời điểm báo của Frigate lấy theo lúc lượt bám vết chồng lên khoảng vi phạm, tức là có lợi cho
Frigate (không tính độ trễ). Số của module: `scripts/evaluate_videos.py`, `runs/eval/nga_chot`.

Vì sao Frigate sót 6/8 (15 lượt bám vết: 10 `co_mu`, 2 `khong_mu`, 3 chưa phân loại):

* **Gán một nhãn cho cả lượt bám vết.** Ở 3 đoạn ban ngày người đội mũ đi vào rồi mới tháo
  mũ; Frigate đã gán `co_mu` từ đầu và giữ nguyên, nên không bao giờ ra `khong_mu`. Module xét
  từng khung nên bắt được lúc tháo mũ. Đây là khác biệt về cách làm, không phải do model kém.
* **Không thấy người thì không phân loại.** Ở `test_khong_mu_03`, `xuong_han` và `goc`, bộ
  phát hiện mặc định của Frigate không ra lượt bám vết người nào (góc cao, người nhỏ), nên
  không có gì để phân loại.
* Hai đoạn bắt được là ban đêm, người bỏ mũ từ đầu lượt bám vết.

Cách chạy cũ (đếm, ghép với module, chưa chấm điểm): `docs/doi-chung-b1.md`.

**Model Frigate đã huấn luyện (11/11):** 149 ảnh `co_mu` + 107 ảnh `khong_mu`, lấy từ video
Pexels và video kho, **không lấy từ `test_*`**. Kiểm tra sơ bộ trên 3 đoạn chưa dùng để huấn
luyện: đúng 4/4 lượt bám vết — nhưng phần "có mũ" là cùng diễn viên với dữ liệu huấn luyện, và
một người đội mũ lưỡi trai Frigate không nhận ra là người nên không được phân loại. Đó chỉ là
kiểm tra model chạy được; số B1 thật là số đo trên `test_*`. Nguồn dữ liệu, cách gán nhãn và
các điểm cần nói trong báo cáo: `docs/doi-chung-b1.md` mục 3.1–3.2.

## 3. B2 — Lửa / khói, ngã: Frigate thuần so với có module

Frigate thuần (mô hình mặc định COCO) không có lớp lửa, khói hay ngã nên không phát hiện được
gì; cột đó là 0 theo thiết kế, không cần chạy. Số của module trên bộ công khai có đáp án của tác
giả (`docs/du-lieu-cong-khai.md`):

| | Frigate thuần | Module: bắt được | Module: báo nhầm | F1 | AUROC | AUPRC (ngẫu nhiên) |
|---|---|---|---|---|---|---|
| Lửa (FIRESENSE) | 0 | 10/11 video | 3/16 video (đều do đèn) | 80,0% | 0,89 | 0,88 (0,37) |
| Khói (FIRESENSE) | 0 | 12/13 video | 0/9 video | 96,0% | 0,88 | 0,91 (0,65) |
| Ngã (UR Fall, giữ khung cuối 4 s) | 0 | 8/30 video | 2/40 video | 40,0% | 0,81* | 0,76 (0,17)* |

\* Ngã: theo từng khung, chỉ trên khung mô hình thấy người. Ngã còn chờ video góc cao của nhóm
(Uy quay) để chốt ngưỡng; bộ UR Fall quay ngang ngực nên 40% khung người nằm không thấy người.

## 4. B3 — Độ trễ từ lúc có sự cố tới lúc hiện trong Frigate / Telegram

| Đoạn | Số đo | Ghi chú |
|---|---|---|
| Vi phạm bắt đầu → luật báo (trong module) | Trung vị 3,4 s (thiếu áo), 5,3 s (thiếu mũ) trên 12 video | Gồm 3 s luật cố ý chờ để chống báo nhầm thoáng qua |
| Luật báo → API trả mã sự cố | Trung vị 0,00 s (lớn nhất 0,02 s), n=10 | Gọi trong mạng Docker nên gần như tức thì |
| Luật báo → sự cố có trong Frigate | **Trung vị 0,22 s** (0,20–0,24), n=10 | Hỏi lại `/events/<id>` tới khi Frigate thực sự có |
| Luật báo → tin tới Telegram | **Trung vị 0,48 s** (0,44–2,17), n=10 | Lần đầu 2,17 s do bắt tay TLS, các lần sau ~0,45 s |
| Mục tiêu (mục 9) | ≤ 5 s | 3,4–5,3 s (luật) + 0,5 s ≈ 3,9–5,8 s: **thiếu áo đạt, thiếu mũ vượt 0,8 s**; phần chậm là 3 s luật cố ý chờ |

Đo bằng `scripts/do_b3.py` ngày 11/11 trên máy Uy (i5-13400F, Frigate 0.17.2 trong Docker,
4 camera đang chạy). Script tạo sự cố thật qua API y như module vẫn làm rồi bấm giờ từng
chặng; sự cố thử đã xoá khỏi Frigate sau khi đo.

Lưu ý khi viết báo cáo: chuỗi 5 giây gần như **chỉ nằm ở luật** (3 giây cố ý chờ để chống
báo nhầm thoáng qua). Phần hạ tầng — gọi API, Frigate ghi nhận, Telegram nhận tin — cộng
lại chưa tới 0,8 giây.

## 5. B4 — Tài nguyên với 1, 4, 8 camera

Phía module đã đo (26/10, RTX 3050 Ti 4 GB, 3 mô hình, luồng 720p 25 khung/giây):

| | 1 camera | 4 camera | 8 camera |
|---|---|---|---|
| Module, chỉ suy luận | 40 ms/lô | 67 ms/lô | 121 ms/lô, GPU ~370 MB |
| Module chạy thật qua RTSP | | 4,84 khung/giây/camera (4,91 khi chạy cùng Frigate) | 4,46 khung/giây/camera |
Phía Frigate đã đo (11/11, máy Uy: i5-13400F 16 luồng, 16 GB RAM, Frigate 0.17.2 trong Docker,
bộ phát hiện **CPU**, mỗi mức lấy 9 mẫu trong 3 phút sau khi chờ ổn định 60 giây):

| | 1 camera | 4 camera | 8 camera |
|---|---|---|---|
| Frigate thuần: CPU trung bình | 88,6% | 120,6% | 149,7% |
| Frigate thuần: CPU đỉnh | 112,8% | 146,2% | 190,9% |
| Frigate thuần: RAM | 652 MB | 984 MB | 1310 MB |
| Frigate thuần: GPU | 0 (bộ phát hiện chạy CPU) | 0 | 0 |
| Thời gian một lần phát hiện | 12,6 ms | 13,2 ms | 13,7 ms |
| FPS trung bình các camera | 5,04 | 4,44 | 4,26 |
| Frigate + module | đo trên laptop Đạt, xem bảng 5b | | |

Đọc số này ra sao:

* **Chỉ `xuong_han` bật phát hiện**, 3–7 camera còn lại chỉ ghi hình. Thêm camera chỉ ghi
  hình tốn thêm khoảng **8–10% CPU và 90 MB RAM mỗi camera** — phần giải mã và ghi đĩa,
  không đụng tới bộ phát hiện (thời gian một lần phát hiện gần như không đổi: 12,6 → 13,7 ms).
* **FPS tụt dần** (5,04 → 4,26) và có lúc chạm 1,2 ở mức 4 và 8 camera. Máy đang chạy cả
  máy giả lập lẫn Frigate, nên đây là giới hạn của một laptop chứ không phải của Frigate.
* Bật phát hiện cho cả 8 camera thì phải đổi sang OpenVINO hoặc GPU; con số trên chỉ đúng
  cho cấu hình "1 camera phát hiện + n camera ghi hình".

Đo bằng `scripts/do_b4.py` (tự sinh cấu hình từng mức, khởi động lại Frigate, chờ ổn định rồi
lấy mẫu, xong trả lại cấu hình cũ). Kết quả thô: `data/b4_2026-11-11_0500.csv`.

Chỗ nghẽn của module là CPU giải mã video (8 luồng 720p tốn 22% CPU), không phải GPU.

### 5b. Frigate thuần so với Frigate + module, cùng một máy (17/11)

Laptop Đạt: i5-11400H (12 luồng = 1200%), 16 GB RAM, RTX 3050 Ti 4 GB; Frigate 0.17.2 trong
Docker, bộ phát hiện CPU; module dùng mô hình tư thế + PPE v2 + cháy v2. Hai lượt chạy liền
nhau sáng 17/11 bằng `scripts/do_b4.py` (có và không `--module`), mỗi mức 11 mẫu trong 3 phút.

| | 1 camera | 4 camera | 8 camera |
|---|---|---|---|
| Frigate thuần: CPU Frigate (đỉnh) | 82% (96%) | 121% (152%) | 138% (174%) |
| Frigate thuần: RAM Frigate | 746 MB | 1146 MB | 1517 MB |
| Có module: CPU Frigate (đỉnh) | 87% (100%) | 130% (174%) | 204% (265%) |
| Có module: CPU module | 46% | 110% | 200% |
| Có module: RAM module | 1577 MB | 1846 MB | 2194 MB |
| Có module: GPU / VRAM cả card | 35% / 668 MB | 38% / 660 MB | 38% / 876 MB |
| Có module: nhịp module mỗi camera | 4,88 khung/s | 4,98 khung/s | 4,96 khung/s |
| Có module: thời gian một lô | 74 ms | 107 ms | 142 ms |
| CPU cả hệ (Frigate + máy giả lập [+ module]) | 116% → 169% | 156% → 278% | 170% → 449% |

Đọc số:

* Module giữ đủ ~5 khung/giây cho **cả 8 camera** (mục tiêu 5); card 4 GB dùng chưa tới 1 GB.
* Thêm module tốn khoảng **25 điểm % CPU mỗi camera** (≈ 2% của 12 luồng), phần lớn là giải
  mã video. 8 camera cả hệ dùng ~449% = khoảng 37% sức CPU của laptop.
* CPU Frigate ở mức 8 camera tăng 138% → 204% khi có module, vì module đọc 8 luồng qua bộ phát
  lại (go2rtc) của Frigate và Frigate phải nhận, lưu sự cố module đẩy vào.
* Lượt đo Frigate thuần tối 16/11 trên cùng laptop ra số gần gấp đôi (CPU 171/220/367%,
  phát hiện 21–33 ms thay vì 11 ms) vì máy đang bận việc khác (Visual Studio tự cập nhật);
  đã đo lại, giữ bản cũ ở `data/b4_laptop_frigate_thuan_may_ban.csv` để đối chiếu. Bài học:
  so có/không module phải đo liền nhau trong cùng điều kiện.

Kết quả thô: `data/b4_laptop_frigate_thuan.csv`, `data/b4_laptop_co_module.csv`.

## 6. Tiêu chí đánh giá (mục 9 của kế hoạch)

| Chỉ số | Mục tiêu | Hiện tại | Đạt? |
|---|---|---|---|
| mAP50 mũ, áo, người | ≥ 0,80 | mũ 0,825; áo 0,743 (tập ảnh test cố định, bản finetune_v1) | Mũ đạt, áo chưa |
| Recall lửa / ngã trên video kiểm thử | ≥ 0,90 | lửa 0,91 (10/11); ngã 0,27 (8/30, UR Fall) | Lửa đạt, ngã chưa |
| Báo nhầm mỗi giờ video mỗi camera | Ghi số đo thật | Đồ bảo hộ 10,8/giờ trên 12 video (1 lần trong 334 s); lửa 15,7/giờ trên FIRESENSE | Đã có số |
| Độ trễ tới Frigate / Telegram | ≤ 5 s | Trung vị 3,9 s (thiếu áo), 5,8 s (thiếu mũ) = luật + ~0,5 s hạ tầng | Thiếu áo đạt; thiếu mũ vượt 0,8 s, do 3 s luật cố ý chờ (mục 4) |
| Số camera chạy đồng thời | Ghi số đo thật | 8 camera chạy cùng Frigate: module 4,96 khung/giây mỗi camera (mục tiêu 5), CPU cả hệ ~37% laptop (mục 5b, 17/11) | Đạt |
| Chạy liên tục không lỗi | ≥ 8 giờ, tự nối lại khi mất luồng | **8,0 giờ** (17–18/11, laptop Đạt, 4 camera + module, mô hình v2): Frigate trả lời API 458/458 lần đo; module không lần nào ngừng; 4 lần cố ý ngắt camera + 4 lần tự rớt, **cả 8 lần tự nối lại**; FPS trung bình 4,1–5,5 mỗi camera. Xem mục 7 | **Đạt** |

## 7. Chạy liên tục 8 giờ (17–18/11)

Laptop Đạt (i5-11400H, RTX 3050 Ti 4 GB), Frigate 0.17.2 bộ phát hiện CPU, 4 camera giả lập
(`xuong_han` phát hiện + 3 camera kho chỉ ghi hình) phát video lặp; module (tư thế + PPE v2 +
cháy v2) đọc cả 4 camera và đẩy sự cố vào Frigate. Theo dõi bằng `scripts/do_on_dinh.py --gio 8
--moi 60 --thu-ngat-moi 2` (cứ 2 giờ cố ý khởi động lại máy giả lập để thử tự nối lại).
Chạy bằng `data/demo/chay_8_gio.ps1` (tự dừng nếu ổ C còn dưới 8 GB).

| | Kết quả |
|---|---|
| Thời gian | 8,0 giờ (23:28 → 07:28), 458 lần đo |
| Frigate trả lời API | 458/458 (100%) |
| Module ngừng | 0 lần |
| Mất luồng | 4 lần cố ý + 4 lần tự rớt, cả 8 lần tự nối lại trong vòng 5 phút đo |
| FPS trung bình (thấp nhất) | xuong_han 5,5 (4,8); kho_1 5,2 (3,8); kho_2 4,1 (1,2); kho_3 5,4 (4,2) |
| CPU trung bình | Frigate 124%, module 109%, máy giả lập 37% (trên 1200% của 12 luồng) |
| VRAM cả card | ~630 MB |
| RAM module | 1,67 GB → 1,01 GB (giảm, không rò) |
| RAM Frigate | 1,21 → 1,76 GB; tăng chậm dần theo giờ: +148, +197, +87, +34, +22, +38, +26 MB, đi ngang ~1,75 GB. Đi theo số sự cố tích luỹ (6.113 sự cố) nên nhiều khả năng là bộ nhớ đệm, không phải rò; chạy dài hơn mới khẳng định được |
| Sự cố module ghi | 6.113 trong 8 giờ (~190 mỗi giờ mỗi camera): video giả lập lặp 17 s, lúc nào cũng có người vi phạm, mỗi chỗ báo lại sau 60 s chờ — không phải tỉ lệ ngoài thực tế |
| Ổ đĩa | tốn ~17 GB trong 8 giờ, phần lớn là đoạn ghi hình Frigate giữ quanh 6.113 sự cố (giữ 14 ngày). Ngoài thực tế ít sự cố hơn nhiều, nhưng phải tính dung lượng theo số sự cố, không chỉ theo số camera |

Lần chạy thứ nhất (16–17/11) dừng ở 5,7 giờ vì đầy ổ: luật báo lặp cũ ra 11.487 sự cố trên video
lặp. Đã sửa (commit 4950f01) trước lần chạy này.
