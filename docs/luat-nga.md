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
