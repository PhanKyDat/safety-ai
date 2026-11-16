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

**Trạng thái (11/11): đã huấn luyện xong** — 149 ảnh `co_mu` + 107 ảnh `khong_mu`, file
model ở `/config/model_cache/mu_bao_ho/` trong container (xem mục 3.1). Muốn huấn luyện
lại hoặc thêm ảnh thì làm theo các bước dưới.

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

> **Không lấy ảnh huấn luyện từ video dùng để đo.** Ảnh mẫu phải cắt từ video kho hoặc
> các đoạn `train_*`; nếu lấy từ chính `test_*` thì model đã nhìn thấy đúng những người
> và đúng khung cảnh sẽ đem ra chấm, số liệu B1 đẹp lên một cách giả tạo và so sánh với
> module thành khập khiễng (module không được huấn luyện trên bộ test).
>
> Frigate sinh ảnh mẫu từ **người nó đang bám vết trên camera đang chạy**, nên trước khi
> bấm *generate examples* thì phát video kho / `train_*` cho `rtsp-sim`, đừng phát `test_*`.

### 3.1. Đã huấn luyện thế nào (11/11)

Lần đầu (sinh ảnh từ video kho) bị tắc vì video kho không có ai đội mũ bảo hộ đi một mình —
hai người đội mũ len luôn đi sát nhau nên ảnh cắt dính cả hai. Đã gỡ bằng cách lấy thêm
video công khai trên Pexels (giấy phép dùng tự do), **không lấy gì từ `test_*`**:

| Nguồn | Lớp | Vì sao chọn |
|---|---|---|
| Pexels 10817415 (mũ xanh, kho hàng) | `co_mu` | Người đội mũ bảo hộ đi một mình |
| Pexels 7448386 (bản 1080p, cắt vùng 800×450) | `co_mu` | Mũ đỏ nhìn **từ trên cao**, giống góc camera xưởng |
| Pexels 5100046 + 5100047 (cùng một người) | cả hai | Cùng người, cùng cảnh: lúc đầu trần, lúc đội mũ |
| Pexels 5100055 (mũ cam) | `co_mu` | Thêm màu mũ |
| Video kho của nhóm | `khong_mu` | Mũ len, mũ lưỡi trai, người đầu trần |

Các đoạn được nối thành một video, chống rung, thêm 5 giây hình đứng yên ở đầu mỗi đoạn để
Frigate dò chuyển động không báo "cả khung đang động", rồi phát cho `rtsp-sim`. Ảnh Frigate tự
cắt được gán nhãn **theo từng lượt bám vết** (xem ảnh rồi gán cả lượt), bỏ những ảnh lúc người
đang đội/bỏ mũ dở dang. Gán và huấn luyện qua API (cùng tác dụng với bấm trên giao diện):

```text
POST /api/classification/mu_bao_ho/dataset/categorize   {"category": "co_mu", "training_file": "<ảnh>"}
POST /api/classification/mu_bao_ho/train
```

Log Frigate: `Wrote training metadata for mu_bao_ho: 256 images` → `Successfully loaded updated model`.

Ba quyết định gán nhãn cần ghi vào báo cáo:

* **Mũ len, mũ lưỡi trai = `khong_mu`.** Đó không phải mũ bảo hộ; bộ test còn có cảnh bẫy
  `bay_mu_luoi_trai` đúng chỗ này.
* **Tránh để góc quay thành "đáp án".** Nếu mọi ảnh `co_mu` quay ngang mắt còn mọi ảnh
  `khong_mu` quay từ trên cao thì model sẽ học góc quay chứ không học mũ. Vì vậy thêm mũ nhìn
  từ trên cao (7448386) và cặp cùng-người-cùng-cảnh (5100046/47).
* Người đầu hói trong video kho chỉ lấy những ảnh nhìn rõ đầu, ảnh mờ thì bỏ.

### 3.2. Kiểm tra sơ bộ trên đoạn chưa từng đưa vào huấn luyện

Dựng một video 67 giây từ ba đoạn Pexels chưa dùng để huấn luyện (5100045, 5100044: người đội
mũ cam; 6079431: xưởng có người đầu trần và người đội mũ lưỡi trai), chạy 3 vòng qua Frigate:

| Lượt bám vết | Thật | Frigate gán | Điểm |
|---|---|---|---|
| Người đội mũ cam (2 đoạn, Frigate nối thành một lượt) | có mũ | `co_mu` | 1,00 (16/16 lần phân loại) |
| Người đầu trần, vòng 1 | không mũ | `khong_mu` | 0,92 |
| Người đầu trần, vòng 2 | không mũ | `khong_mu` | 0,98 |
| Người đầu trần, vòng 3 | không mũ | `khong_mu` | 0,96 |

Đúng 4/4 lượt. Mọi lần phân loại có điểm ≥ ngưỡng 0,8 trên người thật đều đúng; 2 lần sai
đều điểm thấp (0,50 và 0,60) nên Frigate đã tự bỏ qua.

**Đây chỉ là kiểm tra model chạy được, chưa phải số B1**, vì:

* Người đội mũ trong 5100044/45 là **cùng diễn viên** với 5100046/47 đã dùng để huấn luyện,
  nên phần "có mũ" dễ hơn thực tế.
* **Người đội mũ lưỡi trai trong 6079431 Frigate không nhận ra là người** (quay sau lưng, lẫn
  giữa máy móc) nên không được phân loại. Phân loại chỉ chạy trên người Frigate đã bám vết —
  bỏ sót ở khâu phát hiện thì khâu phân loại không cứu được. Khi đo B1 phải tách hai loại
  bỏ sót này.
* Có 2 lần Frigate nhận nhầm kệ hàng là người và model gán `co_mu` (0,66 và 0,94). Lỗi ở
  khâu phát hiện, nhưng sẽ tính thành báo nhầm nếu chỉ đếm thuộc tính.
* Dữ liệu huấn luyện là video stock, bộ `test_*` là nhóm tự quay — khác ánh sáng, khác mũ,
  khác góc. Số trên `test_*` gần như chắc chắn thấp hơn bảng trên.

## 4. Đo B1 (tuần 9)

Model nằm ở `frigate/model_cache/mu_bao_ho/` (`model.tflite` + `labelmap.txt`, đã gitignore).
Đo trên máy khác thì chép nguyên thư mục này sang cùng chỗ, đảm bảo `config.yml` có khối
`classification.custom.mu_bao_ho` như mục 2, rồi khởi động lại Frigate — không cần huấn luyện lại.
Kiểm tra model đã nạp: log Frigate không báo lỗi `mu_bao_ho`, và sau khi có người đi qua thì
`/api/events` có trường `data.mu_bao_ho`.

Phát cùng bộ video kiểm thử cho Frigate, chạy module song song, rồi:

```powershell
.venv\Scripts\python.exe scripts\b1_doi_chung.py --gio 2 --db data\incidents.db
```

Script lấy thuộc tính `mu_bao_ho` từ API sự kiện của Frigate (Frigate 0.17 ghi ở
`data.mu_bao_ho`, đã sửa script đọc đúng chỗ này ngày 11/11), lấy sự cố `no_helmet` từ
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
