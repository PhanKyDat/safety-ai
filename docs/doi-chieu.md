# Đối chiếu: Frigate thuần và Frigate + module của nhóm

Mục 6 của kế hoạch: module của nhóm thêm được gì so với Frigate dùng một mình, và tốn bao nhiêu
tài nguyên. Tài liệu này gom mọi số đo về một chỗ để cắt thẳng vào báo cáo cuối kỳ; ô nào ghi
**chưa đo** là còn chờ, kèm người phụ trách.

Cập nhật lần cuối: 10/11 (tuần 9).

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

| | Frigate Object Classification | Module nhóm (cách vắng mặt) |
|---|---|---|
| Recall thiếu mũ | **chưa đo** (Uy: model `mu_bao_ho` đã dựng, 101 ảnh mẫu, chưa huấn luyện) | 8/8 khoảng vi phạm (100%) |
| Precision | **chưa đo** | 100% (0 báo nhầm) |
| Bộ đo | 12 video: 10 đoạn tự quay góc cao (7 ban ngày 1 người, 3 ban đêm 2 người) + 2 video kho | |

Cách chạy: `docs/doi-chung-b1.md`. Số của module: `scripts/evaluate_videos.py` (mặc định đã dùng
cách vắng mặt), kết quả `runs/eval/tuan9_tat_ca`.

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
| Luật báo → sự cố có trong Frigate | **chưa đo** (Uy) | Module gửi API ngay, một luồng riêng |
| Luật báo → tin tới Telegram | **chưa đo** (Uy) | Gửi sau khi Frigate trả mã sự cố |
| Mục tiêu (mục 9) | ≤ 5 s | |

## 5. B4 — Tài nguyên với 1, 4, 8 camera

Phía module đã đo (26/10, RTX 3050 Ti 4 GB, 3 mô hình, luồng 720p 25 khung/giây):

| | 1 camera | 4 camera | 8 camera |
|---|---|---|---|
| Module, chỉ suy luận | 40 ms/lô | 67 ms/lô | 121 ms/lô, GPU ~370 MB |
| Module chạy thật qua RTSP | | 4,84 khung/giây/camera (4,91 khi chạy cùng Frigate) | 4,46 khung/giây/camera |
| Frigate thuần: CPU, GPU, RAM | **chưa đo** (Uy) | **chưa đo** | **chưa đo** |
| Frigate + module: CPU, GPU, RAM | **chưa đo** (Uy) | **chưa đo** | **chưa đo** |

Chỗ nghẽn của module là CPU giải mã video (8 luồng 720p tốn 22% CPU), không phải GPU.

## 6. Tiêu chí đánh giá (mục 9 của kế hoạch)

| Chỉ số | Mục tiêu | Hiện tại | Đạt? |
|---|---|---|---|
| mAP50 mũ, áo, người | ≥ 0,80 | mũ 0,825; áo 0,743 (tập ảnh test cố định, bản finetune_v1) | Mũ đạt, áo chưa |
| Recall lửa / ngã trên video kiểm thử | ≥ 0,90 | lửa 0,91 (10/11); ngã 0,27 (8/30, UR Fall) | Lửa đạt, ngã chưa |
| Báo nhầm mỗi giờ video mỗi camera | Ghi số đo thật | Đồ bảo hộ 10,8/giờ trên 12 video (1 lần trong 334 s); lửa 15,7/giờ trên FIRESENSE | Đã có số |
| Độ trễ tới Frigate / Telegram | ≤ 5 s | Luật báo sau 3,4–5,3 s; phần tới Frigate/Telegram chưa đo | Chờ B3 |
| Số camera chạy đồng thời | Ghi số đo thật | 8 camera ở 4,46 khung/giây (mục tiêu 5) trên laptop; 4 camera đạt | Đã có số |
| Chạy liên tục không lỗi | ≥ 8 giờ, tự nối lại khi mất luồng | **chưa chạy** (Uy + Đạt) | Chờ |
