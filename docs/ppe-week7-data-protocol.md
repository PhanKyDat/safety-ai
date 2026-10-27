# Quy trình dữ liệu PPE tuần 7

## Mục tiêu

Khắc phục các lỗi trên camera thật: mũ thường bị nhận thành `helmet`, áo thường bị nhận thành `vest`, và ảnh góc nhìn từ trên xuống còn thiếu. TensorRT tạm hoãn; ưu tiên dữ liệu PPE.

## Phân loại video

| Tên video | Được dùng cho | Không được dùng cho |
|---|---|---|
| `test_*.mp4` | Đo model và luật trên video thật | Cắt khung train, chọn checkpoint, dò ngưỡng |
| `train_*.mp4` | Cắt khung, gán nhãn và thêm vào train | — |

Nếu cần tăng số hộp test, có thể gộp **một phần khung riêng biệt** từ `train_*.mp4` vào test cùng các khung test cũ. Không đưa cùng một khung hoặc đoạn liền kề của cùng cảnh vào cả train và test. Giữ nguyên test cũ; ghi nguồn clip, timestamp, split và số instances đã thêm để kết quả đo có thể diễn giải được. Không dùng `test_*.mp4` để train dưới bất kỳ hình thức nào.

## Quy tắc nhãn

- Mũ bảo hộ đạt chuẩn đang đội: `helmet`.
- Mũ lưỡi trai, mũ len, mũ trùm không đạt chuẩn hoặc đầu trần: `no_helmet`.
- Áo phản quang đạt chuẩn đang mặc: `vest`.
- Áo thun/áo thường không phản quang: `no_vest`.
- Ưu tiên góc nhìn camera kho/xưởng từ trên xuống, vật thể nhỏ và che khuất như trong hệ thống thật.
- Hộp phải khớp vật thể PPE/đầu/thân theo quy ước dataset; không gán một chiếc mũ rời hoặc áo rời thành PPE đang được mặc.

### Nguồn nhãn `head`

Không được tự động đổi mọi lớp `head` thành `no_helmet` mà chưa xác minh bộ dữ liệu và ảnh gốc. Hard Hat Workers mô tả lớp `head` là ví dụ chỉ có người/head khi thiếu hard hat, phù hợp làm nguồn ứng viên cho `no_helmet` sau khi kiểm tra ảnh. Với SH17, mô tả lớp `Head` bao gồm các góc nhìn đầu khác nhau và lớp `Helmet` được tách riêng; nhãn `Head` tự nó không khẳng định người đó không đội mũ bảo hộ. Vì vậy cần kiểm tra ảnh gốc, nhất là khi có mũ thường, mũ trùm, che khuất hoặc nhãn không rõ, trước khi remap.

Trong checkout hiện tại, báo cáo `data/hardhat_head_report_20261017_002733.json` ghi 1.240 hộp `head` đã tạo thành nhãn staging cho 162 ảnh, chưa đưa ảnh vào validation. Báo cáo cũng ghi rõ cần kiểm tra các category khác trước khi sử dụng. Chỉ dùng các nhãn staging sau khi đã duyệt ảnh và xác nhận nguồn/class mapping.

## Tiêu chí nghiệm thu model

Với mỗi checkpoint mới:

1. Chạy `scripts/measure_ppe_video.py` trên các video `test_*.mp4` độc lập.
2. Ghi số khung/sự kiện `no_helmet` và `no_vest` mà luật phát ra, cùng lỗi nhận nhầm đáng chú ý.
3. Chỉ nghiệm thu khi luật phát hiện được vi phạm PPE thật trong video test. Chỉ số ảnh val/test tĩnh vẫn được báo riêng, không thay thế phép đo video.

Video, ảnh công nhân, dataset và file trọng số (`.pt`) không được commit hoặc đẩy lên repo public.
