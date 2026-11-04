# Chấm trên bộ dữ liệu công khai

Video tự quay của nhóm chỉ có cảnh đồ bảo hộ. Luật ngã và mô hình cháy chưa gặp cú ngã hay đám
lửa thật nào, nên chấm trên bộ công khai có đáp án của tác giả. Video tải về để trong
`data/public/` (đã gitignore), **không đưa lên repo**: giấy phép cho dùng nhưng không cho nhóm
phát lại dưới tên mình.

| Bộ | Nội dung | Giấy phép | Dung lượng |
|---|---|---|---|
| [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) (ĐH Rzeszów, Kwolek & Kępski 2014) | 30 video ngã, 40 video sinh hoạt (ngồi, cúi nhặt đồ, nằm lên ghế dài), nhãn **từng khung** | CC BY-NC-SA 4.0 (phi thương mại) | 94 MB (camera 0) |
| [FIRESENSE](https://zenodo.org/records/836749) (Dimitropoulos và cộng sự, dự án FP7 FIRESENSE) | 11 video lửa + 16 video dễ nhầm; 13 video khói + 9 video dễ nhầm, nhãn **từng video** | CC BY 4.0 | 820 MB |

## Luật ngã trên UR Fall (04/11)

```powershell
.venv\Scripts\python.exe scripts\evaluate_urfall.py data\public\urfall
```

Mô hình `yolo26n-pose`, luật mặc định (thân ≥ 60°, chân ≥ 45° nếu thấy, nằm liên tục 3 s),
10 khung/giây. Tác giả gán mỗi khung: không nằm, đang ngã (bỏ ra khi chấm, như tác giả làm),
nằm dưới đất.

**Từng video.** Video của bộ này dừng ngay sau cú ngã: người nằm 0,2–2,1 giây rồi hết, nên trên
video gốc luật đòi 3 giây **không kịp báo lần nào (0/30)**. Đây là giới hạn của bộ dữ liệu, không
nói được luật tốt hay kém. Để chấm được, giữ khung cuối thêm 4 giây cho **mọi** video: mô phỏng
người ngã nằm yên tại chỗ (bất tỉnh), đúng ca nguy hiểm nhất.

| | Kết quả |
|---|---|
| Video ngã bắt được | 8/30 |
| Video sinh hoạt báo nhầm | 2/40, cả hai là người **nằm trên ghế dài** (adl-10, adl-11) |
| Precision / Recall / F1 | 80,0% / 26,7% / 40,0% |
| Độ trễ trung vị từ lúc bắt đầu ngã | 4,4 s (gồm 3 s chờ của luật) |

**Từng khung**, để biết sót ở đâu (3290 khung, 796 khung nằm):

| | Recall | AUROC | AUPRC (ngẫu nhiên) |
|---|---|---|---|
| Mọi khung | 17,2% (precision 100%) | 0,51 | 0,56 (0,24) |
| Chỉ khung mô hình thấy người | 28,5% | **0,81** | **0,76** (0,17) |

Hai nguồn sót, gần bằng nhau:

1. **Mô hình không thấy người đang nằm: 40% số khung nằm.** Camera của bộ này đặt ngang ngực,
   người ngã gần camera bị cắt ở mép dưới khung hình hoặc nằm dọc hướng nhìn (bị co ngắn). Ở nửa
   sau bộ (fall-16..30) tỉ lệ này là 64%. Camera nhà xưởng treo cao nhìn xuống nên ca này ít hơn,
   nhưng phải có video quay góc cao mới khẳng định được.
2. **Thấy người mà luật chưa nhận là nằm.** Điểm nghiêng phân biệt tốt (AUROC 0,81), nhưng ngưỡng
   60° quá chặt: đã báo thì đúng 100%, chỉ bắt được 28,5%. Người ngã ngồi bệt dựa tường, nằm co
   gối, nằm nghiêng một phần đều có thân dưới 60°.

**Thử hạ ngưỡng (chưa áp dụng).** Chia đôi bộ, chọn ngưỡng trên nửa đầu (fall-01..15,
adl-01..20) rồi chấm trên nửa sau để không tự chấm lại trên dữ liệu đã dùng để chọn:

| Ngưỡng (thân / chân) | F1 khung, nửa chọn | F1 khung, nửa chấm | Precision nửa chấm |
|---|---|---|---|
| 60° / 45° (đang dùng) | 0,60 | 0,08 | 100% |
| 35° / 30° | 0,74 | 0,21 | 98% |

Hạ ngưỡng có lợi ở cả hai nửa mà gần như không thêm báo nhầm, nhưng nửa sau vẫn thấp vì lỗi
chính ở đó là mô hình không thấy người (mục 1). Chưa đổi luật vì ngưỡng 60° được chọn từ video nhà
kho (người đi lại nghiêng tới 41°); cần video ngã quay góc cao của nhóm để chốt.

Kết quả ghi ở `runs/eval/urfall-*`: `tong_hop.csv`, `tung_video.csv`, `tung_khung.csv` (góc thân,
góc chân từng khung, dùng để quét ngưỡng không cần chạy lại mô hình).
