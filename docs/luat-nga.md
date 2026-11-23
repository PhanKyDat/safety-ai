# Luật ngã: chốt ngày 17/11 (tuần 11)

Luật nằm ở `safety/rules.py` (`FallConfig`, `FallRule`). Tài liệu này ghi lý do của từng điều
kiện và số đo, để cắt vào báo cáo cuối kỳ.

## 1. Vấn đề

Luật cũ chỉ dựa vào góc: thân lệch ≥ 60° so với phương đứng, chân ≥ 45°, giữ liên tục 3 giây.
Camera góc cao nhìn chéo xuống làm người nằm bị co ngắn trên ảnh. Ở 3 video ngã của nhóm
(`test_nga_01..03_dem`, Uy quay 16/11: ngã ngửa, nghiêng, sấp), góc thân chỉ còn 31°, 3° và
40°, nên luật cũ **sót cả 3**. Hạ ngưỡng góc cũng không cứu được (Uy thử: bắt 1/3).

## 2. Luật mới: thêm nhánh "khung bè", ghép với nhánh góc theo kiểu "hoặc"

Nhánh góc giữ nguyên. Nhánh khung bè chỉ báo khi đủ cả các điều kiện sau. Mỗi điều kiện được
thêm vào để chặn đúng một loại báo nhầm đo được.

| Điều kiện | Ngưỡng | Chặn loại báo nhầm nào |
|---|---|---|
| Khung người rộng/cao ≥ 0,7, thấy rõ 2 vai và 2 hông (độ tin cậy ≥ 0,5) | `wide_ratio`, `wide_kp_conf` | Gợi ý của Uy. Lúc đứng ≤ 0,43, lúc nằm ≥ 0,57 trên 3 video ngã |
| Thân ngang mà chân đứng thì là cúi, không xét khung bè | (nhánh góc) | Người cúi nhặt đồ nhìn ngang có khung bè |
| ≥ 80% số khung trong lúc đếm thật sự thấy người nằm | `min_lying_ratio` | Vài khung bè lác đác xen giữa các khung không rõ vai, hông |
| Nằm yên: tâm khung xê dịch ≤ 0,10 lần chiều cao khung | `wide_max_motion` | Người đứng sát mép hoặc ngay dưới camera, đang đi hoặc đang cởi đồ: ngã 0,03–0,04; báo nhầm 0,19–1,07 |
| Đã tụt chiều cao: ≤ 0,75 lần khung cao nhất ở cùng chỗ trong 6 s trước | `wide_max_height_ratio`, `wide_history` | Người đứng với khung bè từ đầu: ngã 0,37–0,66; báo nhầm 0,73–1,44 |
| Khung không chạm mép ảnh (cách mép ≥ 1%) | `edge_margin` | Người đứng sát mép dưới bị cắt nửa người, khung thấp và bè như ngã |

Chiều cao được so **theo chỗ**, không theo mã người, vì người ngã hay bị đổi mã lúc chạm sàn.

## 3. Số đo

| Phiên bản | 3 video ngã của nhóm | Báo ngã nhầm trên 12 video người đi lại của nhóm | UR Fall: bắt / báo nhầm |
|---|---|---|---|
| Luật cũ (chỉ góc) | 0/3 | 0 | 8/30, 2/40 |
| + khung bè (chỉ điều kiện của Uy) | 3/3 | 13 | 9/30, 9/40 |
| + ≥ 80% khung nằm | 3/3 | 13 | 9/30, 9/40 |
| + nằm yên + tụt chiều cao | 3/3 | 1 | 8/30, 7/40 |
| **+ bỏ khung chạm mép (chốt)** | **3/3, báo sau khi ngã ~5 s** | **0** | **8/30, 4/40** |

Ghi chú khi viết báo cáo:

* **3 video ngã là dữ liệu dò ngưỡng, không phải dữ liệu kiểm thử.** Số 3/3 cho thấy luật xử
  lý được góc camera cao, không phải recall thật. Muốn có recall thật cần quay thêm vài cú ngã
  ở chỗ khác, người khác.
* UR Fall quay ngang ngực, người nằm nằm ngang trên ảnh nên nhánh góc đã đủ. Nhánh khung bè
  **không thêm cú ngã nào nhưng cũng không thêm báo nhầm thật**: 4/40 gồm adl-10 và adl-11
  (người nằm xuống sàn hoặc giường; tác giả bộ dữ liệu cũng đánh dấu các khung đó là "nằm",
  luật cũ cũng báo), và adl-01, adl-02 (báo sau khi video đã hết, trong 4 s giữ khung cuối
  của cách chấm: khung đứng im thì "nằm yên" luôn đúng; đây là lỗi do cách chấm).
* Còn thiếu cảnh ngồi xổm hoặc ngồi bệt lâu ở góc cao. Ngồi bệt xuống sàn cũng tụt chiều cao,
  khung bè và nằm yên, nên dễ bị báo. Với xưởng thì đây là chỗ nên kiểm tra tiếp.
* 12 video người đi lại có cả cảnh tháo mũ, cởi áo, người đứng sát mép ảnh và cảnh ban đêm có
  2 người.

Chạy lại: `scripts/evaluate_videos.py data/videos_test/test_*.mp4 ...` (kết quả
`runs/eval/nga_chot`), `scripts/evaluate_urfall.py data/public/urfall` (`runs/eval/urfall_chot`).

## 4. Kiểm thử trên video mới (20/11, Uy quay)

Đợt quay thứ hai, **chưa dùng để dò ngưỡng**: chỗ khác, máy quay ngang 1280×720, đặt cao chúi
xuống, 5–14 s mỗi clip. Đáp án trong `data/videos_test/nhan.csv`.

| Clip | Cảnh | Luật đã chốt |
|---|---|---|
| `test_nga_04_dem` | ngã ngửa | **bắt**, báo sau khi ngã 5 s |
| `test_nga_05_dem` | ngã nghiêng | **sót** |
| `test_nga_06_dem` | ngã sấp về phía camera | **sót** |
| `test_ngoi_01_dem` | ngồi xổm suốt clip | không báo (đúng) |
| `test_ngoi_02_dem` | cúi rồi ngồi xổm nhặt đồ ở bàn | không báo (đúng) |
| `test_cui_01_dem`, `test_cui_02_dem` | cúi nhặt đồ dưới sàn | không báo (đúng) |

**Recall thật trên video góc cao: 1/3; báo nhầm 0 trên 4 clip ngồi/cúi.** Kết quả:
`runs/eval/nga_dot2` (chỉ 7 clip mới), `runs/eval/nga_tat_ca_goc` (cả 10 clip).

Vì sao sót (soi từng khung, mô hình tư thế thấy rõ vai/hông ≥ 0,9 ở cả hai):

* **`nga_05`**: khung bè 1,15–1,20, không chạm mép, chiều cao đã tụt (262 so với 433 px lúc đứng
  = 0,60), nằm im tuyệt đối từ giây 5,6. Trượt ở **"nằm yên"**: `_wide_confirmed` đo xê dịch trên
  **mọi khung từ lúc bắt đầu đếm**, gồm cả lúc đang lăn xuống sàn (≈ 87 px, ngưỡng 0,10 × 262 =
  26 px). Lần đếm không bị ngắt nên phần xê dịch đó ở lại mãi — nằm im bao lâu cũng không đạt.
* **`nga_06`**: ngã về phía camera, khung người **chạm mép dưới ảnh** từ giây 5,2 nên điều kiện
  `edge_margin` bỏ qua. Đây đúng là loại báo nhầm điều kiện này chặn (người đứng sát mép dưới bị
  cắt nửa người), nên khó tách bằng khung người; về triển khai thì nên đặt camera sao cho vùng
  sàn cần giám sát nằm trọn trong khung, cách mép.

Thử sửa "nằm yên" chỉ đo trên ~3 s gần nhất (giả lập, **chưa sửa `rules.py`**):

| | Ngã bắt được (6 clip) | Báo nhầm (4 clip ngồi/cúi) |
|---|---|---|
| Luật đã chốt | 4/6 (`nga_05`, `nga_06` sót) | 0 |
| "Nằm yên" đo trên 3 s gần nhất | 5/6 (thêm `nga_05`) | **1** (`ngoi_02`: ngồi xổm nhặt đồ ở bàn) |

Ngồi xổm lâu cũng bè, tụt chiều cao và đứng yên, nên nới điều kiện này đổi một cú ngã lấy một
báo nhầm. Để nguyên luật đã chốt và ghi recall 1/3 là trung thực; nếu sửa thì phải tìm thêm đặc
trưng tách ngồi xổm (ví dụ hông thấp hơn gối, hoặc khung cao/rộng của người ngồi) và đo lại trên
cả 12 video người đi lại của nhóm.

## 5. Quyết định (23/11)

Đạt chấm lại trên cả 22 video (`runs/eval/dot2_kiem_lai`), ra cùng số với Uy:
- ngã: 4/6 (`nga_01..04` bắt; `nga_05`, `nga_06` sót);
- 0 báo ngã nhầm trên 12 video người đi lại và 4 clip ngồi/cúi.

**Giữ nguyên luật đã chốt** vì hai lý do:
- Nới "nằm yên" chỉ đổi một cú ngã lấy một báo nhầm ngồi xổm.
- Chỉnh tiếp theo 7 clip đợt 2 thì chúng thành dữ liệu dò ngưỡng, nhóm mất bộ kiểm thử độc lập
  duy nhất. Không còn thời gian quay đợt 3.

Báo cáo ghi:
- **Recall thật ở góc cao: 1/3** trên video chưa dùng dò ngưỡng.
- **Khuyến nghị lắp đặt:** vùng sàn cần giám sát nằm trọn trong khung hình, cách mép ảnh
  (`nga_06` sót vì người ngã chạm mép dưới).
- **Hướng làm tiếp:** đặc trưng tách ngồi xổm khỏi nằm (hông so với gối), hoặc huấn luyện riêng
  một lớp "người nằm" thay vì suy từ tư thế.

Lưu ý khi đọc bảng của `evaluate_videos.py` trên cả 22 video: cột thiếu mũ/thiếu áo sẽ ra 10 lần
"nhầm". Đó là người trong clip ngã/ngồi/cúi không đội mũ, mặc áo; clip chỉ gán đáp án cho ngã,
nên mọi sự cố đồ bảo hộ ở đó đều bị tính là nhầm. Số đồ bảo hộ phải lấy trên 12 video đồ bảo hộ:
6/8 thiếu mũ, 7/7 thiếu áo, 0 báo nhầm.
