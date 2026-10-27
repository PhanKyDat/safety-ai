# Bộ video kiểm thử

Người phụ trách: Long Uy. Mọi phép đo của nhóm (B1–B4, mục 9 tiêu chí đánh giá) chạy trên
bộ video này, nên quay xong sớm là cả nhóm đo được sớm.

> **Video không lên repo.** `data/` đã gitignore. Chia nhau qua Drive của nhóm.
> **Ai lên hình phải đồng ý trước**, và chỉ dùng cho đồ án — hình người là dữ liệu cá nhân
> theo Luật Bảo vệ dữ liệu cá nhân (hiệu lực 01/01/2026).

## 1. Cách quay

| Thứ | Chốt |
|---|---|
| Vị trí máy | Cao 2,5–3 m, chúi xuống khoảng 30–45°, **đặt cố định** (kê lên kệ/tủ, đừng cầm tay) |
| Độ phân giải | 1280×720 trở lên, 25–30 khung/giây |
| Độ dài mỗi đoạn | 20–60 giây |
| Ánh sáng | Quay 2 lần: ban ngày và lúc đèn vàng buổi tối |
| Khoảng cách | Người cao khoảng 1/4 đến 1/2 chiều cao khung hình |

Camera **phải đứng yên**: Frigate khởi động ở chế độ hiệu chỉnh chuyển động và chỉ thoát
ra khi chuyển động nhỏ hơn 5% khung hình. Cầm tay quay là Frigate không phát hiện gì cả
(đã gặp ở tuần 1, xem [frigate-setup.md](frigate-setup.md)).

Mỗi đoạn nên có **5 giây đầu không có người** để Frigate kịp hiệu chỉnh.

## 2. Danh sách cảnh cần quay

| Mã | Cảnh | Yêu cầu | Dùng để đo |
|---|---|---|---|
| `binh_thuong` | Có mũ + áo phản quang, đi lại bình thường | 2–3 đoạn, nhiều người | Báo nhầm (càng ít càng tốt) |
| `khong_mu` | Không đội mũ, đứng/đi trong khung **≥ 10 s** | 2–3 đoạn | B1, recall thiếu mũ |
| `khong_ao` | Không mặc áo phản quang **≥ 10 s** | 2 đoạn | Recall thiếu áo |
| `nga` | Ngã rồi **nằm yên ≥ 5 s** — ngã lên nệm | 2–3 đoạn | Recall ngã |
| `vung_nguy_hiem` | Đi vào và đứng trong vùng đã vẽ | 2 đoạn | Vùng nguy hiểm |
| `bay_ao_cam` | Mặc áo cam/vàng **thường** (không phải áo phản quang) | 1–2 đoạn | Báo nhầm |
| `bay_mu_luoi_trai` | Đội mũ lưỡi trai, mũ len | 1–2 đoạn | Báo nhầm |
| `bay_den_vang` | Đèn vàng, ánh sáng ấm, hơi nước nếu có | 1–2 đoạn | Báo nhầm cháy/khói |

**Lửa và khói: không đốt thật.** Dùng video công khai có giấy phép rõ ràng (Pexels,
Pixabay). Tải về `data/videos_test/`, đặt tên `lua_01_ngay.mp4`, `khoi_01_ngay.mp4`, và
ghi nguồn vào `data/videos_test/nguon.txt`.

Đã tải sẵn 3 đoạn từ Pexels: `lua_01_ngay`, `lua_02_ngay`, `khoi_01_ngay` (nguồn ghi
trong `nguon.txt`). Cả ba đều **quay cận cảnh và hơi rung**, nên trước khi dùng phải
chống rung và kéo dài cho đủ 20 giây:

```powershell
.\scripts\make-sim-video.ps1 -Source data\videos_test\lua_01_ngay.mp4 -Stabilize
```

Lưu ý khi viết báo cáo: đây là video cận cảnh ngọn lửa, không phải cảnh nhà xưởng nhìn
từ camera trên cao, nên chỉ đo được "model có nhận ra lửa/khói không", chưa đo được
tình huống thật.

## 3. Đặt tên tệp

```
<ma_canh>_<so_thu_tu>_<ngay|dem>.mp4      ví dụ: khong_mu_01_ngay.mp4
```

Để hết trong `data/videos_test/`. Tên sai thì script kiểm tra sẽ báo.

## 4. Kiểm tra sau khi quay

```powershell
.venv\Scripts\python.exe scripts\kiem_tra_video_test.py
```

Script soi từng tệp: độ phân giải, số khung/giây, độ dài, và **độ rung của máy quay**
(so khung đầu với các khung sau). Đoạn nào rung quá thì quay lại, hoặc chữa bằng:

```powershell
.\scripts\make-sim-video.ps1 -Source data\videos_test\<ten>.mp4 -Stabilize
```

## 5. Ghi nhãn thời điểm sự cố

Không có nhãn thì không tính được precision/recall. Script ở trên tạo sẵn khung mẫu
`data/videos_test/nhan.csv`; mở bằng Excel rồi điền mỗi sự cố một dòng:

```csv
tep,nhan,bat_dau_giay,ket_thuc_giay,ghi_chu
khong_mu_01_ngay.mp4,no_helmet,3.5,21.0,nguoi mac ao xanh
nga_01_ngay.mp4,fall,12.0,25.0,nga nguoi roi nam yen
binh_thuong_01_ngay.mp4,,,,khong co su co - dung do bao nham
```

* `nhan` dùng đúng tên trong hợp đồng dữ liệu: `no_helmet`, `no_vest`, `fire`, `smoke`,
  `fall`, `danger_zone`. Đoạn không có sự cố thì để trống (vẫn phải có dòng).
* Thời gian tính bằng giây **từ đầu đoạn video**.
* Một đoạn có nhiều sự cố thì ghi nhiều dòng.

## 6. Phát cho hệ thống chạy thử

```powershell
docker compose stop rtsp-sim
Copy-Item data\videos_test\khong_mu_01_ngay.mp4 data\samples\xuong_han.mp4
docker compose up -d
docker compose restart frigate
```

Hoặc thêm hẳn một camera giả lập riêng cho bộ kiểm thử trong `sim/mediamtx.yml` để không
phải chép đè.
