# Bài B1 — Object Classification của Frigate so với module của nhóm

Mục 6.2 của kế hoạch: so "không đội mũ" do **Frigate tự phân loại** với "không đội mũ" do
**model YOLO + luật thời gian** của nhóm. Tài liệu này là phần dựng bản đối chứng (việc
tuần 7 của Uy); số liệu precision/recall chạy ở tuần 9.

## 1. Hai cách làm khác nhau ở đâu

| | Frigate Object Classification | Module của nhóm |
|---|---|---|
| Cách nhận | Cắt ảnh **người Frigate đã bám vết**, đưa qua MobileNetV2 phân loại | YOLO phát hiện thẳng `helmet` / `no_helmet` trong khung |
| Huấn luyện | Bấm trên giao diện Frigate, vài phút, chạy CPU | Train ngoài, cần GPU, nhiều giờ |
| Kết quả | Thuộc tính gắn lên người: `mu_bao_ho: co_mu / khong_mu` | `Incident` có nhãn `no_helmet` sau khi qua luật |
| Điều kiện ra kết quả | Điểm ≥ `threshold` **và** ≥ 60% trong ít nhất 3 lần phân loại đồng ý | Thiếu mũ kéo dài ≥ 3 giây (có dung sai mất dấu) |
| Chỗ xem | Explore → Tracked Object Details, MQTT `frigate/tracked_object_details`, API | Telegram, bảng thống kê, bảng `incidents` |

Điểm đáng nói trong báo cáo: Frigate phân loại **từng lần nhìn thấy**, còn module áp
**luật theo thời gian**. Nên Frigate nhạy hơn với ảnh mờ một khung, module thì bỏ qua
những lần thoáng qua — đó là lý do hai bên lệch nhau, không phải bên nào "sai".

## 2. Cấu hình (đã có sẵn trong `frigate/config.example.yml`)

```yaml
classification:
  custom:
    mu_bao_ho:
      enabled: true
      threshold: 0.8
      save_attempts: 200
      object_config:
        objects: [person]
        classification_type: attribute
```

Dùng `attribute`, **không** dùng `sub_label`: mỗi người chỉ mang được một `sub_label`,
mà tuần sau còn thêm áo phản quang nữa thì hai thứ đè lên nhau.

## 3. Gán nhãn và huấn luyện (làm trên giao diện Frigate)

Đã chuẩn bị sẵn trên máy: model `mu_bao_ho`, hai lớp `co_mu` / `khong_mu`, và **101 ảnh
mẫu** cắt từ người mà Frigate đã bám vết (lớp `none` Frigate tự thêm).

1. Mở <https://localhost:8971> → **Classification** → `mu_bao_ho`.
2. Frigate hiện lần lượt từng lớp. Ở lớp `co_mu` chọn hết ảnh người **có đội mũ**, sang
   lớp `khong_mu` chọn ảnh người **đầu trần**. Ảnh không chọn tự rơi vào `none`.
3. Chọn xong lớp cuối thì Frigate **tự huấn luyện** (1–3 phút trên CPU).
4. Thiếu ảnh cho lớp nào thì để trống, chạy tiếp vài hôm rồi vào tab **Recent
   Classifications** gán thêm và huấn luyện lại.

Muốn sinh thêm ảnh mẫu:

```powershell
docker exec frigate curl -fsS -X POST http://127.0.0.1:5000/api/classification/generate_examples/object `
  -H "Content-Type: application/json" -d '{\"model_name\":\"mu_bao_ho\",\"label\":\"person\"}'
```

Gán nhãn cho đúng là phần quyết định: ảnh cắt từ camera trên cao, mũ nhìn từ đỉnh đầu,
nên nhìn kỹ rồi hãy chọn. Nhãn ẩu thì số liệu B1 vô nghĩa.

## 4. Đo B1 (tuần 9)

Phát cùng bộ video kiểm thử cho Frigate, chạy module song song, rồi:

```powershell
.venv\Scripts\python.exe scripts\b1_doi_chung.py --gio 2 --db data\incidents.db
```

Script lấy thuộc tính `mu_bao_ho` từ API sự kiện của Frigate, lấy sự cố `no_helmet` từ
bảng `incidents`, rồi in ra số người Frigate cho là không mũ, số sự cố module bắt được,
và phần chồng nhau theo cửa sổ thời gian. Đối chiếu với nhãn thời điểm sự cố của bộ video
kiểm thử để ra precision/recall cho từng bên.

## 5. Lưu ý khi chạy thật

* Model phân loại chạy trên **CPU**, nhẹ, nhưng vẫn cộng thêm tải khi nhiều camera —
  ghi lại số đo cho bài B4.
* Huấn luyện ngốn CPU 1–3 phút, đừng bấm lúc đang quay video demo.
* Gỡ bản đối chứng: đặt `enabled: false` trong `classification.custom.mu_bao_ho`, không
  cần xoá dữ liệu đã gán nhãn.
* Bật log chi tiết khi cần xem vì sao không gán được nhãn:

```yaml
logger:
  logs:
    frigate.data_processing.real_time.custom_classification: debug
```
