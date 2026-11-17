# Kịch bản demo phần Frigate (buổi bảo vệ)

Người phụ trách: Long Uy. Phần này demo nền VMS và đường cảnh báo; phần module (thiếu đồ bảo hộ,
cháy, ngã) Đạt demo bằng cửa sổ `--show`. Khoảng **4 phút**, chạy trên một laptop.

## 1. Chuẩn bị (trước giờ bảo vệ 30 phút)

```powershell
docker compose up -d
.\scripts\frigate-check.ps1                       # phải ra hết OK
```

* Camera `xuong_han` phát **video kho** (có người đi vào vùng nguy hiểm): `data\samples\xuong_han.mp4`
  là bản sao của `xuong_han_kho.mp4`. Đổi video thì `docker compose stop rtsp-sim`, chép, `docker compose up -d`.
* Mở sẵn 4 cửa sổ:

| Cửa sổ | Lệnh / trang | Ghi chú |
|---|---|---|
| 1. Frigate | <https://localhost:8971> đăng nhập `admin` | Mở sẵn tab *Live*, *Review*, *Explore* |
| 2. Cầu nối | `.venv\Scripts\python.exe -m safety.mqtt_bridge` | Gửi Telegram cảnh báo vùng nguy hiểm |
| 3. Module | lệnh của Đạt, thêm `--db data\incidents_demo_bao_ve.db --telegram --telegram-toi-da 5` | DB riêng cho số không lẫn |
| 4. Thống kê | `.venv\Scripts\streamlit.exe run dashboard/app.py -- --db data\incidents_demo_bao_ve.db` | |

* Telegram mở trên điện thoại, chiếu lên màn hình nếu được (hoặc Telegram Web trên laptop).
* Tắt thông báo Windows, tắt chế độ ngủ, cắm sạc.

## 2. Trình tự (khoảng 4 phút)

| Bước | Nói gì | Làm gì |
|---|---|---|
| 1 | Frigate nhận 4 camera, 1 camera phát hiện người, 3 chỉ ghi hình | *Live*: lưới 4 camera |
| 2 | Vùng nguy hiểm vẽ theo mặt sàn, người ở trong ≥ 2 s mới thành Alert | *Settings → Debug* camera `xuong_han`, bật *Zones* và *Bounding boxes*, chờ người đi vào lối giữa |
| 3 | Alert vào mục Review, cầu nối gửi Telegram có ảnh và link | *Review → Alerts*; điện thoại hiện tin |
| 4 | Bấm link trong tin → mở đúng sự cố trong Frigate kèm đoạn ghi hình | Bấm link Telegram |
| 5 | Sự cố của module (thiếu mũ, ngã…) nằm ở *Explore*, không ở Review — phát hiện khi làm | *Explore*, lọc nhãn `no_helmet` |
| 6 | Trang thống kê: theo ngày/loại/camera, nút "Báo nhầm" ra số báo nhầm mỗi giờ | Cửa sổ 4, bấm "Báo nhầm" một dòng |
| 7 | Phân quyền: tài khoản `viewer` chỉ xem, không sửa cấu hình | Đăng nhập `viewer` ở cửa sổ ẩn danh |

## 3. Khi có sự cố

| Chuyện gì | Làm gì |
|---|---|
| Không có mạng | Telegram không gửi được → bỏ bước 3–4, chiếu tin Telegram đã chụp sẵn (`hinh_telegram.png`), vẫn demo được Review / Explore / thống kê |
| Docker Desktop tắt / Frigate không lên | Chiếu **video quay màn hình dự phòng** (quay trước theo đúng trình tự trên) |
| Chờ mãi không có Alert | Video kho lặp lại khoảng 2 phút một lần; trong lúc chờ nói bước 5–7 trước |
| Telegram ngập tin | Cầu nối chỉ gửi mỗi review một lần; module có `--telegram-toi-da 5` |

Quay video dự phòng: OBS hoặc Xbox Game Bar (Win + G), quay cả 7 bước một lần, để trên Drive nhóm.

## 4. Sau buổi demo

* Xoá DB demo nếu không cần: `data\incidents_demo_bao_ve.db` (cùng hai tệp `-wal`, `-shm`).
* Sự cố module tạo trong Frigate lúc demo: xoá ở *Explore* hoặc qua API nếu không muốn lẫn vào số đo.

## 5. Câu hỏi có thể gặp (phần Frigate)

* **Sao không tự viết VMS?** Ghi hình, xem lại, phân quyền, vùng nguy hiểm đều đã có và đã được
  kiểm chứng; nhóm dồn sức vào phần Frigate không có (đồ bảo hộ, cháy, ngã) và đường cảnh báo.
* **Chạy được mấy camera?** Trên laptop: 8 camera ghi hình + 1 camera phát hiện, Frigate ~150% CPU
  (1,5 luồng / 16), xem `docs/doi-chieu.md` mục 5. Bật phát hiện cho tất cả thì cần OpenVINO hoặc GPU.
* **Độ trễ?** Hạ tầng < 0,8 s (Frigate 0,22 s, Telegram 0,48 s); phần còn lại là 3 s luật cố ý chờ.
* **Dữ liệu cá nhân?** Ghi liên tục giữ 1 ngày, sự cố 14 ngày; tài khoản chỉ xem; video người không lên repo.
* **Mất camera thì sao?** Frigate tự nối lại; module đọc qua restream nên không bị đứt (đã thử ngắt
  camera giả lập 2 lần, module vẫn 4,94 khung/giây).
