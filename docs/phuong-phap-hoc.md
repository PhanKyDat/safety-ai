# Thành phần nào học có giám sát, thành phần nào không

Hệ thống ghép nhiều thành phần, không phải một mô hình duy nhất. Mỗi thành phần thuộc một
loại khác nhau, và cách đánh giá cũng khác nhau theo loại.

| Thành phần | Loại | Học từ đâu | Nhóm có huấn luyện không | Đánh giá bằng |
|---|---|---|---|---|
| Mô hình đồ bảo hộ (YOLO26s: người, mũ, không mũ, áo, không áo) | **Học có giám sát**, học chuyển giao (fine-tune từ trọng số đã huấn luyện sẵn) | Ảnh đã gán hộp và nhãn: SH17, construction-ppe, bổ sung Hard Hat Workers | Có (bạn phụ trách dữ liệu) | P, R, F1, mAP, đường cong PR / AUPRC từng lớp trên tập test cố định |
| Mô hình cháy/khói (YOLO26s) | **Học có giám sát**, học chuyển giao | D-Fire đã làm sạch (ảnh gán hộp lửa, khói) | Có | Như trên; thêm video lửa công khai (FIRESENSE) |
| Mô hình tư thế (yolo26n-pose) | **Học có giám sát** (17 điểm khớp đã gán trên bộ COCO-Pose) | COCO | **Không**: dùng nguyên trọng số có sẵn | Gián tiếp, qua luật ngã |
| Bám vết ByteTrack | **Không phải học máy**: bộ lọc Kalman dự đoán vị trí + ghép cặp theo độ trùng hộp | Không học, chỉ có tham số chỉnh tay | – | Độ ổn định mã định danh (tuần 3: trung vị 5,1 s) |
| Luật sự cố (thiếu đồ bảo hộ 3 s, cháy 6/10 khung, ngã: thân ≥ 60° trong 3 s) | **Luật viết tay** (rule-based), ngưỡng chọn theo số đo trên video | Không học | – | F1 theo sự cố, AUROC/AUPRC theo đoạn video (`scripts/evaluate_videos.py`) |
| Phát hiện chuyển động của Frigate | **Không cần nhãn**: so khung hiện tại với ảnh nền cập nhật dần (xử lý ảnh cổ điển); gần với tinh thần không giám sát | Tự thích nghi với cảnh | – | Không đánh giá riêng; chỉ để Frigate quyết định khi nào chạy nhận diện |
| Nhận diện người của Frigate (cho vùng nguy hiểm) | **Học có giám sát** (mô hình phát hiện vật thể huấn luyện trên COCO) | COCO | Không | Như sự cố vùng nguy hiểm |

## Vì sao chọn học có giám sát cho phần chính

- Bài toán cần **gọi đúng tên** vi phạm (thiếu mũ khác thiếu áo khác cháy) để gửi đúng cảnh báo;
  học không giám sát chỉ nói được "khác thường", không nói được khác thường kiểu gì.
- Đã có sẵn bộ dữ liệu gán nhãn công khai cho đồ bảo hộ và cháy, nên chi phí gán nhãn thấp.
- Học chuyển giao (fine-tune từ trọng số có sẵn) cho kết quả tốt với ít dữ liệu hơn huấn luyện từ đầu.

**Giới hạn đã gặp:** học có giám sát chỉ tốt trên dữ liệu giống lúc huấn luyện. Tuần 7, mô hình đồ bảo hộ
đạt số tốt trên tập ảnh nhưng trên camera nhìn từ trên xuống thì nhận mũ len, mũ lưỡi trai là mũ bảo hộ.
Cách chữa vẫn là có giám sát: thêm ảnh góc camera thật và gán nhãn đúng các loại mũ thường.

## Chỗ có thể dùng học không giám sát (hướng mở rộng, chưa làm)

- **Phát hiện bất thường trên cảnh xưởng**: học từ video bình thường (không cần nhãn), báo khi cảnh
  khác hẳn thường ngày. Bắt được tình huống chưa có nhãn (vật rơi, đám đông tụ tập), nhưng không biết
  đó là sự cố gì, nên chỉ hợp làm lớp cảnh báo phụ.
- **Gom nhóm báo nhầm**: gom các sự cố bị bấm "báo nhầm" theo đặc trưng ảnh để tìm ra kiểu cảnh hay gây
  nhầm (đèn vàng, sàn xám), từ đó bổ sung dữ liệu huấn luyện.
- **Gán nhãn giả (bán giám sát)**: dùng mô hình có sẵn tự gán hộp "người" cho bộ dữ liệu thiếu nhãn người
  (Hard Hat Workers gần như không gán người), rồi người kiểm lại. Cần bạn phụ trách dữ liệu xác nhận đã áp
  dụng tới đâu trước khi ghi vào báo cáo.
