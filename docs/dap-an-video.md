# Đáp án cho video kiểm thử

Video đặt tên theo [bo-video-kiem-thu.md](bo-video-kiem-thu.md): `test_<ma_canh>_<so>_<ngay|dem>.mp4`,
để trong `data/videos_test/`. Video `train_*` không dùng để chấm (script tự bỏ qua).

**Cách chính: một bảng `nhan.csv` cho cả thư mục.** `scripts/kiem_tra_video_test.py --nhan` tạo sẵn
khung, mỗi video một dòng, gợi ý nhãn theo tên tệp; mình chỉ điền giờ. Một video có nhiều khoảng vi
phạm thì thêm dòng; video không vi phạm thì để trống cột nhãn:

```csv
tep,nhan,bat_dau_giay,ket_thuc_giay,ghi_chu
test_khong_mu_01_ngay.mp4,no_helmet,5,20,nguoi ao xanh
test_khong_mu_01_ngay.mp4,no_helmet,0:30,0:41,quay lai
test_binh_thuong_01_ngay.mp4,,,,
```

Khung mẫu đã gợi ý nhãn mà quên điền giờ thì script chấm báo lỗi, không chấm nhầm.

**Cách phụ:** tệp `.csv` **cùng tên** với video, để cạnh video (mẫu: [mau-dap-an.csv](mau-dap-an.csv)),
dùng khi chấm một video lẻ ngoài thư mục chung. Có cả hai thì tệp riêng được ưu tiên.

```csv
nhan,bat_dau,ket_thuc,ghi_chu
no_helmet,5,20,nguoi ao xanh khong doi mu
no_vest,0:25,0:41,nguoi ao xanh coi ao phan quang
fall,0:52,1:00,nga tren nem roi nam yen
```

- **nhan**: một trong `no_helmet`, `no_vest`, `fire`, `smoke`, `fall`, `danger_zone`
  (`danger_zone` do Frigate xử lý, script chấm của module bỏ qua dòng này).
- **bat_dau, ket_thuc**: thời gian trong video, ghi bằng giây (`5`, `12.5`) hoặc phút:giây (`0:52`).
  Tính từ lúc vi phạm **bắt đầu thấy được trên hình** tới lúc **hết**: người không đội mũ đi vào
  khung ở giây 5, ra khỏi khung ở giây 20 thì ghi `5,20`. Không cần chính xác tới từng khung,
  sai một hai giây không sao.
- Hai người cùng vi phạm cùng lúc: ghi hai dòng. Một người vi phạm, đi ra rồi quay lại: ghi hai dòng.
- **Video không có vi phạm nào** (vd mọi người đội mũ, mặc áo đầy đủ): tệp chỉ có dòng tiêu đề.
  Loại video này quan trọng không kém, vì nó đo báo nhầm.
- **ghi_chu**: tuỳ ý, để người khác đọc hiểu.

Mở bằng Notepad hay Excel đều được; lưu lại đúng dạng CSV, mã hoá UTF-8.

## Chấm

```powershell
.venv\Scripts\python.exe scripts\evaluate_videos.py data\test_videos\*.mp4 `
    --extra-model models\ppe_v1_finetune_best.pt --extra-model models\fire_v1_yolo26s.pt
```

Kết quả ra hai mức:

- **Theo sự cố**: báo đúng / báo nhầm / bỏ sót / báo lặp, Precision, Recall, F1, độ trễ báo,
  số lần báo nhầm mỗi giờ. Sự cố tới muộn tối đa 5 giây sau khi hết khoảng vi phạm vẫn tính
  là đúng, vì luật cần thấy vi phạm 3 giây mới báo.
- **Theo đoạn 2 giây**: AUROC và AUPRC, kèm đường cong ROC và PR (`roc_pr.png`). Mức này không
  phụ thuộc ngưỡng, dùng để so các bản mô hình với nhau.

Video dùng để chấm (`test_*`) **không bao giờ đưa vào huấn luyện**; huấn luyện dùng `train_*`.
Video người lao động không đẩy lên repo (thư mục `data/` đã nằm trong `.gitignore`).
