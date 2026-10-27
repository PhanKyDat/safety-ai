"""Bảng thống kê sự cố an toàn lao động (việc tuần 6).

    .venv\\Scripts\\streamlit.exe run dashboard/app.py
    .venv\\Scripts\\streamlit.exe run dashboard/app.py -- --db data/incidents_demo.db

Đọc thẳng bảng `incidents` (docs/data-contract.md mục 3b) — cùng bảng mà module của
Đạt và cầu nối MQTT ghi vào. Mọi truy vấn nằm trong safety/stats.py để kiểm thử được;
tệp này chỉ lo phần hiển thị.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from safety.stats import (  # noqa: E402
    LABEL_VI,
    SEVERITY_VI,
    Filter,
    danh_dau_bao_nham,
    danh_sach,
    dem_theo,
    dem_theo_ngay,
    gia_tri_loc,
    gio_dia_phuong,
    link_frigate,
    ton_tai,
    tong_quan,
)

GOC = Path(__file__).resolve().parent.parent


def tham_so():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--db", default=None)
    args, _ = p.parse_known_args()
    return args


st.set_page_config(page_title="Thống kê sự cố an toàn", page_icon="🦺", layout="wide")
load_dotenv(GOC / ".env")

args = tham_so()
DB = Path(args.db or os.environ.get("INCIDENT_DB", "data/incidents.db"))
if not DB.is_absolute():
    DB = GOC / DB
FRIGATE_UI = os.environ.get("FRIGATE_UI")

st.title("Thống kê sự cố an toàn lao động")

if not ton_tai(DB):
    st.warning(f"Chưa có cơ sở dữ liệu `{DB}`.")
    st.markdown(
        "Chạy module hoặc cầu nối MQTT để sinh sự cố thật, hoặc tạo dữ liệu mẫu để xem thử:\n\n"
        "```powershell\n"
        ".venv\\Scripts\\python.exe scripts/seed_dashboard_demo.py --db data/incidents_demo.db\n"
        ".venv\\Scripts\\streamlit.exe run dashboard/app.py -- --db data/incidents_demo.db\n"
        "```"
    )
    st.stop()

# --- Bộ lọc ---------------------------------------------------------------
gt = gia_tri_loc(DB)

with st.sidebar:
    st.header("Bộ lọc")
    hom_nay = date.today()
    khoang = st.date_input(
        "Khoảng ngày",
        value=(hom_nay - timedelta(days=13), hom_nay),
        max_value=hom_nay,
        format="DD/MM/YYYY",
    )
    # Người dùng bấm ngày đầu xong, Streamlit chạy lại ngay khi chưa chọn ngày sau.
    tu_ngay, den_ngay = khoang if isinstance(khoang, tuple) and len(khoang) == 2 else (khoang[0], hom_nay)

    cameras = st.multiselect("Camera", gt["cameras"], default=[])
    labels = st.multiselect(
        "Loại sự cố",
        gt["labels"],
        default=[],
        format_func=lambda n: LABEL_VI.get(n, n),
    )
    an_bao_nham = st.checkbox("Ẩn sự cố đã đánh dấu báo nhầm", value=False)
    gioi_han = st.slider("Số dòng tối đa", 20, 500, 100, step=20)
    st.caption(f"Nguồn: `{DB.name}`")
    if st.button("Tải lại"):
        st.rerun()

loc = Filter(
    tu_ts=time.mktime(datetime.combine(tu_ngay, datetime.min.time()).timetuple()),
    den_ts=time.mktime(datetime.combine(den_ngay + timedelta(days=1), datetime.min.time()).timetuple()),
    cameras=tuple(cameras),
    labels=tuple(labels),
    an_bao_nham=an_bao_nham,
)

# --- Số liệu tổng quan ----------------------------------------------------
tq = tong_quan(DB, loc)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng sự cố", tq["tong"])
c2.metric("Nghiêm trọng", tq["nghiem_trong"])
c3.metric("Báo nhầm", tq["bao_nham"], f"{tq['ti_le_bao_nham']:.0%}")
if tq["moi_nhat"]:
    gio, ngay = gio_dia_phuong(tq["moi_nhat"]).split()
    c4.metric("Sự cố gần nhất", f"{ngay[:5]} {gio[:5]}")
else:
    c4.metric("Sự cố gần nhất", "—")

if tq["tong"] == 0:
    st.info("Không có sự cố nào khớp bộ lọc.")
    st.stop()

# --- Biểu đồ --------------------------------------------------------------
t1, t2 = st.columns([2, 1])
with t1:
    st.subheader("Theo ngày")
    theo_ngay = dem_theo_ngay(DB, loc)
    st.bar_chart(pd.Series(theo_ngay, name="Số sự cố"), color="#d62728")
with t2:
    st.subheader("Theo loại")
    theo_loai = dem_theo(DB, "label", loc)
    st.bar_chart(
        pd.Series({LABEL_VI.get(k, k): v for k, v in theo_loai.items()}, name="Số sự cố"),
        horizontal=True,
        color="#ff7f0e",
    )

st.subheader("Theo camera")
st.bar_chart(pd.Series(dem_theo(DB, "camera", loc), name="Số sự cố"), horizontal=True)

# --- Danh sách sự cố ------------------------------------------------------
st.subheader("Danh sách sự cố")
rows = danh_sach(DB, loc, gioi_han=gioi_han)

for r in rows:
    nhan = LABEL_VI.get(r["label"], r["label"])
    muc = SEVERITY_VI.get(r["severity"], r["severity"])
    dau = "⚠️" if r["false_alarm"] else {"critical": "🔴", "high": "🟠"}.get(r["severity"], "🟡")
    tieu_de = f"{dau} {gio_dia_phuong(r['ts'])} · {nhan} · {r['camera']}"
    if r["false_alarm"]:
        tieu_de += " · ĐÃ ĐÁNH DẤU BÁO NHẦM"

    with st.expander(tieu_de):
        trai, phai = st.columns([1, 2])
        with trai:
            anh = Path(r["snapshot"]) if r["snapshot"] else None
            if anh and not anh.is_absolute():
                anh = GOC / anh
            if anh and anh.is_file():
                st.image(str(anh), use_container_width=True)
            else:
                st.caption("Không có ảnh")
        with phai:
            st.write(f"**Mức độ:** {muc}  ·  **Độ tin cậy:** {r['score']:.2f}")
            if r["sub_label"]:
                st.write(f"**Chi tiết:** {r['sub_label']}")
            if r["track_id"] is not None:
                st.write(f"**Mã bám vết:** {r['track_id']}")

            url = link_frigate(FRIGATE_UI, r["frigate_event_id"])
            if url:
                st.link_button("Xem trong Frigate", url)
            elif r["frigate_event_id"]:
                st.caption(f"Mã sự cố Frigate: `{r['frigate_event_id']}` (đặt FRIGATE_UI trong .env để có link)")

            # Nút báo nhầm: đánh dấu được và bỏ đánh dấu được, vì bấm nhầm là chuyện thường.
            if r["false_alarm"]:
                if st.button("Bỏ đánh dấu báo nhầm", key=f"bo_{r['id']}"):
                    danh_dau_bao_nham(DB, r["id"], False)
                    st.rerun()
            else:
                if st.button("Báo nhầm", key=f"nham_{r['id']}", type="primary"):
                    danh_dau_bao_nham(DB, r["id"], True)
                    st.rerun()

if len(rows) == gioi_han:
    st.caption(f"Đang hiển thị {gioi_han} sự cố mới nhất — tăng giới hạn ở thanh bên nếu cần.")
