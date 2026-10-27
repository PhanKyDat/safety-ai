# Bảng thống kê sự cố (việc tuần 6)

Người phụ trách: Long Uy.

| Tệp | Việc |
|---|---|
| `dashboard/app.py` | Giao diện Streamlit |
| `safety/stats.py` | Mọi truy vấn và cập nhật DB (có kiểm thử riêng) |
| `scripts/seed_dashboard_demo.py` | Sinh sự cố giả để dựng/thử giao diện |

## Chạy

```powershell
.venv\Scripts\streamlit.exe run dashboard/app.py
```

Mặc định đọc `INCIDENT_DB` trong `.env` (`data/incidents.db`) — **cùng tệp** module của Đạt
và cầu nối MQTT ghi vào. Muốn xem thử bằng dữ liệu giả:

```powershell
.venv\Scripts\python.exe scripts/seed_dashboard_demo.py --db data/incidents_demo.db
.venv\Scripts\streamlit.exe run dashboard/app.py -- --db data/incidents_demo.db
```

Dữ liệu giả ghi ra tệp riêng, không lẫn vào thống kê thật.

## Có gì trên trang

* **Bốn ô số liệu**: tổng sự cố, số nghiêm trọng, số báo nhầm kèm tỉ lệ, sự cố gần nhất.
  Tỉ lệ báo nhầm là con số mục 9 của kế hoạch cần ("số báo nhầm trên mỗi giờ video").
* **Biểu đồ**: theo ngày, theo loại sự cố, theo camera — đều ăn theo bộ lọc.
* **Bộ lọc** (thanh bên): khoảng ngày, camera, loại sự cố, ẩn sự cố đã báo nhầm, số dòng.
* **Danh sách sự cố**: mỗi dòng mở ra có ảnh, mức độ, độ tin cậy, mã bám vết,
  nút **Xem trong Frigate** và nút **Báo nhầm**.

## Nút "Báo nhầm"

Ghi `false_alarm = 1` vào đúng dòng trong bảng `incidents`, và bấm lại thì bỏ đánh dấu
(bấm nhầm là chuyện thường). Sự cố đã đánh dấu hiện chữ **ĐÃ ĐÁNH DẤU BÁO NHẦM** và có
thể ẩn khỏi danh sách bằng ô chọn ở thanh bên.

## Hai điều phải giữ khi sửa về sau

* **Đọc thì mở DB ở chế độ chỉ đọc.** Module và cầu nối ghi liên tục; trang thống kê giữ
  khoá ghi là chặn cả hệ thống. `safety/stats.py` mở `mode=ro` cho mọi truy vấn, chỉ lúc
  bấm "Báo nhầm" mới mở ghi và ghi rất ngắn. Có kiểm thử khoá hành vi này.
* **Chỉ đếm theo cột cho phép.** Hàm `dem_theo` chỉ nhận `label`, `camera`, `severity`;
  tên cột không đi vào câu SQL bằng cách ghép chuỗi tuỳ tiện.

DB bật chế độ WAL (Đạt đặt trong `safety/db.py`). Chép DB sang máy khác thì chép kèm hai
tệp `.db-wal` và `.db-shm`, nếu không sẽ thiếu những sự cố ghi gần nhất.

## Kiểm thử

```powershell
.venv\Scripts\python.exe -m pytest tests/test_stats.py -q
```

14 ca: lọc theo ngày/camera/nhãn, ẩn báo nhầm, đếm theo cột, tỉ lệ báo nhầm khi DB rỗng,
đánh dấu và bỏ đánh dấu, DB chỉ đọc phải từ chối ghi, link Frigate dùng đúng `/explore`.
