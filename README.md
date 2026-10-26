# Hệ thống phát hiện sự cố an toàn lao động bằng camera AI

Đồ án 12 tuần, nhóm 3 người. Frigate làm nền VMS, nhóm làm **Module AI an toàn** gắn vào Frigate.

| Khối | Ai làm | Việc |
|---|---|---|
| Frigate (nền VMS) | Cài & cấu hình | Camera, xem trực tiếp, ghi hình, xem lại, phát hiện người, vùng nguy hiểm, phân quyền |
| Module AI an toàn | Nhóm tự làm | Thiếu mũ / áo phản quang, cháy / khói, người ngã; áp luật; đẩy sự cố vào Frigate |
| Cảnh báo & thống kê | Nhóm tự làm | Telegram, bảng thống kê Streamlit |

## Đọc trước khi code

**[docs/data-contract.md](docs/data-contract.md)** — định dạng dữ liệu giữa các phần, đã chốt tuần 1.
Nguồn sự thật trong code là [`safety/contracts.py`](safety/contracts.py).

Tóm tắt:

- AI → luật: `FrameResult` chứa các `Detection{track_id, cls, bbox, conf, keypoints}`
- Luật → Frigate / Telegram / DB: `Incident{camera, label, sub_label, severity, score, ts, snapshot}`
- Nhãn sự cố module đẩy vào Frigate: `no_helmet`, `no_vest`, `fire`, `smoke`, `fall`; riêng `danger_zone` do Frigate tự sinh, module chỉ chuyển tiếp

## Cài môi trường

Cần Python 3.12 (3.13+ chưa có bánh xe PyTorch CUDA ổn định), Git, và card NVIDIA nếu muốn chạy GPU.

```powershell
git clone https://github.com/PhanKyDat/safety-ai.git
cd safety-ai
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Kiểm tra GPU đã được nhận:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Phải ra `True` kèm tên card.

Chạy thử webcam:

```powershell
yolo predict model=yolo26n.pt source=0 show=True device=0
```

Nếu báo `running scripts is disabled` khi activate venv:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Chạy module

Đọc một luồng và chạy tracking, in ra `FrameResult`:

```powershell
.venv\Scripts\python.exe -m safety.cli --source 0                      # webcam
.venv\Scripts\python.exe -m safety.cli --source rtsp://127.0.0.1:8554/xuong_han
.venv\Scripts\python.exe -m safety.cli --source video.mp4 --json       # file, in JSON
```

Khi chưa có Frigate, tự dựng một luồng RTSP để thử (cần `mediamtx` và `ffmpeg` trên PATH):

```powershell
.\scripts\test-stream.ps1 -Video data\samples\webcam.mp4
.\scripts\test-stream.ps1 -Stop
```

Khi Frigate của Uy chạy, chỉ đổi URL sang `rtsp://<frigate_host>:8554/<camera>`, không phải sửa code.
Đọc luồng **chính**, không đọc luồng phụ `<camera>_sub`: luồng phụ nén mạnh làm mô hình cháy
nhận sàn xám là khói ở 22% số khung (luồng chính 4%), xem [docs/data-contract.md](docs/data-contract.md).

### Đẩy sự cố sang Frigate

Mỗi khung đi qua luật đồ bảo hộ và luật cháy/khói; sự cố được chụp ảnh vào `snapshots/`
(không lên repo) và, nếu có `--frigate`, tạo thành sự cố trong Frigate kèm đoạn ghi hình 25 giây,
xem ở mục **Explore**. Sự cố tạo qua API **không** hiện ở mục Review.

Module chạy trên máy host cần gọi API cổng 5000 của Frigate. Cổng này không đăng nhập nên
`docker-compose.yml` cố ý không mở; khi phát triển thì mở tạm, chỉ cho localhost, bằng một
file riêng trên máy mình (đã có trong `.gitignore`, Docker Compose tự đọc kèm):

```yaml
# docker-compose.override.yml
services:
  frigate:
    ports:
      - "127.0.0.1:5000:5000"
```

```powershell
docker compose up -d
.venv\Scripts\python.exe -m safety.cli --source rtsp://127.0.0.1:8554/xuong_han --camera xuong_han --frigate http://127.0.0.1:5000/api
```

`--camera` phải trùng tên camera trong Frigate, sai thì module báo lỗi ngay lúc khởi động.

### Nhiều camera trong một tiến trình

Mỗi camera thêm một `--cam TEN=NGUON` (có `--cam` thì bỏ qua `--camera`/`--source`). Mỗi mô hình
chỉ nạp một lần và chạy theo lô (khung mới nhất của mọi camera), còn bộ bám vết thì mỗi camera
một cái, nên mã người của hai camera không lẫn nhau:

```powershell
.venv\Scripts\python.exe -m safety.cli --cam xuong_han=rtsp://127.0.0.1:8554/xuong_han `
    --cam kho=rtsp://127.0.0.1:8554/kho --extra-model models\ppe_v1_finetune_best.pt `
    --extra-model models\fire_v1_yolo26s.pt --frigate http://127.0.0.1:5000/api
```

Đo ngày 26/10 trên RTX 3050 Ti 4 GB, 3 mô hình (tư thế + đồ bảo hộ + cháy), luồng 720p 25 khung/giây:

| | 1 camera | 4 camera | 8 camera |
|---|---|---|---|
| Chỉ suy luận (khung có sẵn) | 40 ms/lô | 67 ms/lô | 121 ms/lô, GPU dùng ~370 MB |
| Chạy thật qua RTSP | | 4,84 khung/giây/camera | 4,46 khung/giây/camera (199 ms/lô) |

Chỗ nghẽn khi chạy thật là **CPU giải mã video**, không phải GPU: chỉ đọc 8 luồng đã tốn 22% CPU
của máy (bộ giả lập phát 8 luồng tốn thêm 26%, lúc chạy thật không có phần này). Giải mã bằng GPU
qua OpenCV không đỡ được (27%). Vì vậy TensorRT không giúp mấy cho 8 camera; muốn nhẹ hơn thì đặt
luồng chính của camera ở 10 khung/giây thay vì 25 (module chỉ cần 5).

Mở luồng RTSP thì các camera phải xếp hàng (OpenCV khoá khi mở), mỗi camera chờ tới khung khoá kế
tiếp. Camera đặt khung khoá mỗi 1–2 giây như Frigate khuyên thì 8 camera kết nối xong trong ~14 giây.

Thêm `--telegram` để gửi tiếp vào nhóm Telegram chung, cùng bot và cùng kiểu tin với
cảnh báo vùng nguy hiểm. Mã bot và chat id lấy từ `.env` (`TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, xem [docs/canh-bao-telegram.md](docs/canh-bao-telegram.md)).
Sự cố tới Frigate trước rồi mới sang Telegram, để tin nhắn kèm đường dẫn mở sự cố;
Frigate sập thì Telegram vẫn báo, chỉ là không có đường dẫn. Hai bên gửi ở hai luồng
riêng nên Telegram chậm không làm sự cố tới Frigate muộn.

Thêm `--db` để ghi sự cố vào bảng `incidents` ([schemas/incidents.sql](schemas/incidents.sql)),
cùng tệp với cảnh báo vùng nguy hiểm mà cầu nối MQTT ghi (`INCIDENT_DB` trong `.env`, mặc định
`data/incidents.db`), để trang thống kê đọc một chỗ là đủ. Dòng được ghi sau khi Frigate trả
mã sự cố, nên có sẵn `frigate_event_id` để làm nút mở sự cố. DB không bị lọc như Telegram:
thống kê cần đủ mọi sự cố. Chạy thử bằng nhãn giả lập thì ghi ra tệp riêng
(`--db data/incidents_demo.db`) để sự cố giả không lẫn vào thống kê thật.

Chưa có model PPE/cháy thì chạy thử cả đường đi bằng nhãn giả lập. Sự cố giả lập ghi rõ
"GIẢ LẬP" trong mô tả; với `--telegram` mỗi lần chạy chỉ gửi tối đa 2 tin để không làm ngập nhóm:

```powershell
.venv\Scripts\python.exe scripts\demo_incident_flow.py --seconds 40
.venv\Scripts\python.exe scripts\demo_incident_flow.py --seconds 25 --telegram
```

Chấm module trên video kiểm thử có đáp án (F1 theo sự cố, AUROC/AUPRC theo đoạn, đường cong ROC/PR);
cách ghi đáp án xem [docs/dap-an-video.md](docs/dap-an-video.md):

```powershell
.venv\Scripts\python.exe scripts\evaluate_videos.py data\test_videos\*.mp4 --extra-model models\ppe_v1_finetune_best.pt
```

Chạy kiểm thử luật sự cố (không cần camera hay model):

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Đo độ ổn định mã định danh trên một luồng, phục vụ chỉnh tham số luật:

```powershell
.venv\Scripts\python.exe scripts\measure_tracks.py --seconds 180
```

## Chạy Frigate

Cần Docker Desktop (WSL2) và một video mẫu ở `data/samples/xuong_han.mp4`. Chi tiết: [docs/frigate-setup.md](docs/frigate-setup.md).

```powershell
Copy-Item .env.example .env
Copy-Item frigate\config.example.yml frigate\config.yml
docker compose up -d
.\scripts\frigate-check.ps1
```

Giao diện: <https://localhost:8971>. Module đọc `rtsp://127.0.0.1:8554/xuong_han`.

## Cảnh báo Telegram

Nghe MQTT của Frigate, chuyển cảnh báo "người vào vùng nguy hiểm" sang Telegram và ghi DB.
Cách tạo bot và lấy chat id: [docs/canh-bao-telegram.md](docs/canh-bao-telegram.md).

```powershell
.\.venv\Scripts\python.exe -m safety.mqtt_bridge --check   # kiểm tra token + MQTT
.\.venv\Scripts\python.exe -m safety.mqtt_bridge           # chạy thật
```

## Cấu trúc

```
safety/          Module AI
  contracts.py     Định dạng dữ liệu dùng chung
  stream.py        Đọc luồng RTSP/webcam/file, tự kết nối lại, giới hạn FPS
  pipeline.py      YOLO + tracking -> FrameResult
  cli.py           Chạy thử
docs/            Tài liệu chốt
schemas/         DDL SQLite cho lịch sử sự cố
frigate/         Config mẫu cho Frigate (copy thành config.yml, không commit)
sim/             Camera giả lập chạy trong Docker cho Frigate
mosquitto/       Config broker MQTT
scripts/         Dựng luồng RTSP thử nghiệm, kiểm tra Frigate
mediamtx.yml     Config máy chủ RTSP thử nghiệm (chạy trên host, không cần Frigate)
docker-compose.yml  Frigate + Mosquitto + camera giả lập
```

## Phân công

| Thành viên | Vai trò |
|---|---|
| Kỳ Đạt | Kiến trúc & Module AI — pipeline, tracking, luật, tích hợp Frigate API + MQTT, Docker Compose |
| Nam Trường | Dữ liệu & mô hình — dataset, gán nhãn, train PPE + cháy, logic ngã (pose), TensorRT, đo độ chính xác |
| Long Uy | Quản trị Frigate & cảnh báo — cài Frigate, Mosquitto, Telegram, Streamlit, video kiểm thử |

## Lưu ý

- Repo public: không commit `.env`, ảnh người lao động, hay `config.yml` có mật khẩu camera.
- Hình ảnh người lao động là dữ liệu cá nhân theo Luật Bảo vệ dữ liệu cá nhân (hiệu lực 01/01/2026).
- Hệ thống chỉ hỗ trợ, không thay thế hệ thống báo cháy đạt chuẩn PCCC.
- Frigate: giấy phép MIT. Ultralytics YOLO: AGPL-3.0 — dùng cho đồ án thoải mái, thương mại hoá cần giấy phép riêng.
