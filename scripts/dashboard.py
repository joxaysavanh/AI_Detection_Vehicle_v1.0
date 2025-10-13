# แนะนำให้ใช้ Grafana ในการสร้าง Dashboard แทน Streamlit
# เนื่องจาก Grafana มีฟีเจอร์ที่เหมาะสมกับการสร้าง Dashboard และ ง่ายต่อการใช้งานมากกว่า เพราะ Grafana จะเชื่อมต่อกับฐานข้อมูลโดยตรง
# แต่ถ้าต้องการใช้ Streamlit จริงๆ ควรปรับปรุงโค้ดให้เหมาะสมกับการใช้งานจริงมากกว่านี้

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="🚗 Vehicle Detection Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS
# ========================
st.markdown("""
    <style>
    /* Main theme */
    :root {
        --primary-color: #FF6B35;
        --secondary-color: #004E89;
        --accent-color: #F7931E;
        --success-color: #06A77D;
        --danger-color: #D62828;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Header styling */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 40px;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 16px;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    /* Status indicators */
    .status-active {
        color: #06A77D;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #D62828;
        font-weight: bold;
    }
    
    /* Custom dividers */
    .divider {
        margin: 20px 0;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# CONFIGURATION
# ========================
API_BASE_URL = "http://localhost:8020/api"  # เปลี่ยนเป็น URL ของ API ที่ใช้งานจริง

# ========================
# API FETCH FUNCTIONS
# ========================
@st.cache_data(ttl=5)
def fetch_counts(from_time: Optional[datetime] = None, to_time: Optional[datetime] = None, camera_id: Optional[int] = None):
    try:
        params = {}
        if from_time:
            params["from_time"] = from_time.isoformat()
        if to_time:
            params["to_time"] = to_time.isoformat()
        if camera_id:
            params["camera_id"] = camera_id

        response = requests.get(f"{API_BASE_URL}/counts", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching counts: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_cameras():
    try:
        response = requests.get(f"{API_BASE_URL}/cameras", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching cameras: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return []


@st.cache_data(ttl=5)
def fetch_detections(from_time: Optional[datetime] = None, to_time: Optional[datetime] = None, camera_id: Optional[int] = None, limit: int = 100):
    try:
        params = {'limit': limit}
        if from_time:
            params["from_time"] = from_time.isoformat()
        if to_time:
            params["to_time"] = to_time.isoformat()
        if camera_id:
            params["camera_id"] = camera_id

        response = requests.get(f"{API_BASE_URL}/detections", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching detections: {response.text}")
            return []
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return []


def get_vehicle_category(vehicle_class: str) -> str:
    mapping = {
        "motorcycle": "motorcycle_tuk_tuk",
        "tuk-tuk": "motorcycle_tuk_tuk",
        "sedan": "sedan_pickup_suv",
        "single-pick-up": "sedan_pickup_suv",
        "van": "van",
        "bus": "minibus_bus",
        "minibus": "minibus_bus",
        "trailer": "truck6_truck10_trailer",
        "truck6": "truck6_truck10_trailer",
        "truck10": "truck6_truck10_trailer",
    }
    return mapping.get(vehicle_class, "unknown")

# ========================
# CHART CREATION FUNCTIONS
# ========================
def create_vehicle_counts_chart(counts_data: List[Dict]):
    if not counts_data:
        return None

    df = pd.DataFrame(counts_data)
    counts_sum = {
        "🏍 Motorcycle/Tuk-Tuk": df["motorcycle_tuk_tuk"].sum(),
        "🚗 Sedan/Pickup/SUV": df["sedan_pickup_suv"].sum(),
        "🚙 Van": df["van"].sum(),
        "🚌 Minibus/Bus": df["minibus_bus"].sum(),
        "🚛 Truck/Trailer": df["truck6_truck10_trailer"].sum(),
    }

    fig = px.bar(
        x=list(counts_sum.keys()),
        y=list(counts_sum.values()),
        title="นับจำนวนของแต่ละกลุ่มยานพาหนะ",
        labels={"x": "Vehicle Type", "y": "Count"},
        color=list(counts_sum.values()),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=400,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=100)
    )
    fig.update_traces(textposition='outside', text=list(counts_sum.values()))
    return fig


def create_time_series_chart(counts_data: List[Dict]):
    if not counts_data:
        return None

    df = pd.DataFrame(counts_data)
    df["start_ts"] = pd.to_datetime(df["start_ts"])
    df = df.sort_values("start_ts")

    fig = go.Figure()

    vehicle_types = [
        ("motorcycle_tuk_tuk", "🏍 Motorcycle/Tuk-Tuk", "#FF6B35"),
        ("sedan_pickup_suv", "🚗 Sedan/Pickup/SUV", "#004E89"),
        ("van", "🚙 Van", "#F7931E"),
        ("minibus_bus", "🚌 Minibus/Bus", "#06A77D"),
        ("truck6_truck10_trailer", "🚛 Truck/Trailer", "#D62828"),
    ]

    for col, name, color in vehicle_types:
        fig.add_trace(go.Scatter(
            x=df["start_ts"],
            y=df[col],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3),
            marker=dict(size=6),
            fill='tozeroy' if col == "motorcycle_tuk_tuk" else None,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y} vehicles<extra></extra>",  # ✅ ตัดวินาทีออก
        ))

    # ✅ ปรับแกน X ให้ไม่แสดงวินาที และจัด tick สวยขึ้น
    fig.update_xaxes(
        tickformat="%H:%M",  # แสดงเฉพาะ ชั่วโมง:นาที
        showgrid=True,
        gridcolor="LightGrey",
        title_text="Time",
        title_standoff=10
    )

    # ✅ ปรับแกน Y เล็กน้อยให้อ่านง่าย
    fig.update_yaxes(
        title_text="Count",
        showgrid=True,
        gridcolor="LightGrey",
        rangemode="tozero"
    )

    fig.update_layout(
        title="นับยานพาหนะแต่ละช่วงเวลา",
        xaxis_title="Time",
        yaxis_title="Count",
        height=400,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=40),  # เพิ่ม margin ล่างเล็กน้อย
    )

    return fig



def create_vehicle_distribution_pie(counts_data: List[Dict]):
    if not counts_data:
        return None

    df = pd.DataFrame(counts_data)
    counts_sum = {
        "Motorcycle/Tuk-Tuk": df["motorcycle_tuk_tuk"].sum(),
        "Sedan/Pickup/SUV": df["sedan_pickup_suv"].sum(),
        "Van": df["van"].sum(),
        "Minibus/Bus": df["minibus_bus"].sum(),
        "Truck/Trailer": df["truck6_truck10_trailer"].sum(),
    }
    
    # Remove zero values
    counts_sum = {k: v for k, v in counts_sum.items() if v > 0}

    fig = px.pie(
        values=list(counts_sum.values()),
        names=list(counts_sum.keys()),
        title="Vehicle Distribution",
        color_discrete_sequence=["#FF6B35", "#004E89", "#F7931E", "#06A77D", "#D62828"],
    )
    fig.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
    return fig


def create_hourly_trends(counts_data: List[Dict]):
    if not counts_data:
        return None

    df = pd.DataFrame(counts_data)
    df["start_ts"] = pd.to_datetime(df["start_ts"])
    df["hour"] = df["start_ts"].dt.hour
    
    hourly_totals = df.groupby("hour")[[
        "motorcycle_tuk_tuk", "sedan_pickup_suv", "van", "minibus_bus", "truck6_truck10_trailer"
    ]].sum().reset_index()
    hourly_totals["total"] = hourly_totals[[
        "motorcycle_tuk_tuk", "sedan_pickup_suv", "van", "minibus_bus", "truck6_truck10_trailer"
    ]].sum(axis=1)

    fig = px.area(
        hourly_totals,
        x="hour",
        y="total",
        title="Total Vehicles by Hour",
        labels={"hour": "Hour of Day", "total": "Total Count"},
        color_discrete_sequence=["#667eea"],
    )
    fig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ========================
# MAIN DASHBOARD
# ========================
def main():
    # Header Section (full width)
    with st.container():
        st.markdown("""
            <div class="header-section" style="width: 100%; box-sizing: border-box;">
                <div class="header-title">🚗 หน้าหลัก การตรวจจับยานพาหนะ</div>
            </div>
        """, unsafe_allow_html=True)
        # Live indicator placeholder (right aligned)
        live_status_placeholder = st.empty()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        st.markdown("---")

        # Time Range Filter
        st.markdown("*ช่วงเวลา*")
        time_range = st.selectbox(
            "Select time range",
            ["Live (Real-time)", "เลือกวัน",],
            label_visibility="collapsed"
        )

        if time_range == "Live (Real-time)":
            # ใช้ session state เพื่อป้องกันการ refresh ที่ไม่จำเป็น
            if "live_from_time" not in st.session_state or "live_to_time" not in st.session_state:
                st.session_state["live_from_time"] = datetime.now() - timedelta(hours=24)
                st.session_state["live_to_time"] = datetime.now()
            
            from_time = st.session_state["live_from_time"]
            to_time = st.session_state["live_to_time"]
            is_realtime = True
        elif time_range == "เลือกวัน":
            st.markdown("*เลือกวันที่ต้องการ*")

            # --- Step 1: เลือกวัน ---
            selected_date = st.date_input(
                "Pick a day",
                datetime.now().date(),
                label_visibility="collapsed"
            )

            # --- Step 2: สร้าง interval ราย 15 นาที + ตัวเลือก "ทั้งหมดของวัน" ---
            intervals = ["ทั้งหมดของวัน"]
            for h in range(24):
                for m in [0, 15, 30, 45]:
                    start = f"{h:02d}:{m:02d}"
                    end_h, end_m = h, m + 15
                    if end_m == 60:
                        end_h += 1
                        end_m = 0
                    if end_h < 24:
                        end = f"{end_h:02d}:{end_m:02d}"
                    else:
                        end = "23:59"
                    intervals.append(f"{start} - {end}")

            # --- Step 3: ให้ผู้ใช้เลือกช่วงเวลา ---
            selected_interval = st.selectbox(
                "เลือกช่วงเวลา",
                intervals,
                index=0,  # ค่าเริ่มต้นคือ "ทั้งหมดของวัน"
                placeholder="เลือกช่วงเวลาที่ต้องการ..."
            )

            # --- Step 4: แปลงช่วงเวลาเป็น datetime range ---
            if selected_interval == "ทั้งหมดของวัน":
                from_time = datetime.combine(selected_date, datetime.strptime("00:00", "%H:%M").time())
                to_time = datetime.combine(selected_date, datetime.strptime("23:59:59", "%H:%M:%S").time())
            else:
                start_str, end_str = selected_interval.split(" - ")
                from_time = datetime.combine(selected_date, datetime.strptime(start_str, "%H:%M").time())
                if end_str == "23:59":
                    to_time = datetime.combine(selected_date, datetime.strptime("23:59:59", "%H:%M:%S").time())
                else:
                    to_time = datetime.combine(selected_date, datetime.strptime(end_str, "%H:%M").time())

            is_realtime = False


        st.markdown("---")

        # Camera Filter
        st.markdown("*เลือกกล้องที่ต้องการ*")
        cameras = fetch_cameras()
        camera_list = ["เลือกทั้งหมด"] + [f"{c['camera_id']}: {c['name']}" for c in cameras]


        # Live (Real-time) always use latest camera_id, no dropdown
        if time_range == "Live (Real-time)" and cameras:
            max_camera = max(cameras, key=lambda c: c['camera_id'])
            camera_id = max_camera['camera_id']
            st.markdown(f"**กล้อง:** {max_camera['camera_id']}: {max_camera['name']}")
            # Reset session state for camera selection when switching to Live
            st.session_state["selected_camera"] = f"{max_camera['camera_id']}: {max_camera['name']}"
        else:
            # Use dropdown for other time ranges, keep last selected camera
            if "selected_camera" not in st.session_state or st.session_state.get("last_time_range") == "Live (Real-time)":
                st.session_state["selected_camera"] = camera_list[0]
            selected_camera = st.selectbox(
                "Select camera",
                camera_list,
                index=camera_list.index(st.session_state["selected_camera"]) if st.session_state["selected_camera"] in camera_list else 0,
                label_visibility="collapsed",
                key="camera_selectbox"
            )
            st.session_state["selected_camera"] = selected_camera
            camera_id = None
            if selected_camera != "เลือกทั้งหมด":
                camera_id = int(selected_camera.split(":")[0])
        # Track last time_range to help reset on switch
        if "last_time_range" not in st.session_state:
            st.session_state["last_time_range"] = time_range
        
        # Reset live time when switching to Live mode
        if time_range == "Live (Real-time)" and st.session_state.get("last_time_range") != "Live (Real-time)":
            st.session_state["live_from_time"] = datetime.now() - timedelta(hours=24)
            st.session_state["live_to_time"] = datetime.now()
        
        st.session_state["last_time_range"] = time_range

        st.markdown("---")

        # Auto Refresh
        st.markdown("*Settings*")
        auto_refresh = st.checkbox("🔄 Real-time Updates (5s)", value=False)
        show_advanced = False

        if auto_refresh:
            time.sleep(5)
            st.rerun()

    # Fetch Data
    use_live_metrics = time_range == "Live (Real-time)"
    
    if use_live_metrics:
        # สำหรับ Live mode ให้ดึงข้อมูลล่าสุดโดยไม่ใช้ช่วงเวลา
        detections = fetch_detections(None, None, camera_id, limit=1000)
        counts_data = []  # ไม่ใช้ counts_data สำหรับ Live mode
        df = pd.DataFrame()
        
        if detections:
            det_df = pd.DataFrame(detections)
            det_df['category'] = det_df['vehicle_class'].apply(get_vehicle_category)
            category_counts = det_df.groupby('category').size()
            total_vehicles = len(detections)
        else:
            total_vehicles = 0
    else:
        # สำหรับโหมดอื่นๆ ใช้ counts_data ตามปกติ
        counts_data = fetch_counts(from_time, to_time, camera_id)
        
        if not counts_data:
            st.warning("📭 ไม่มีข้อมูลสำหรับฟิลเตอร์ที่เลือก โปรดลองใช้ช่วงเวลาหรือกล้องอื่น.")
            return
            
        df = pd.DataFrame(counts_data)
        vehicle_columns = [
            "motorcycle_tuk_tuk", "sedan_pickup_suv", "van", "minibus_bus", "truck6_truck10_trailer"
        ]
        if not df.empty:
            total_vehicles = df[vehicle_columns].sum().sum()

    # Update live status indicator
    if auto_refresh:
        live_status_placeholder.markdown("""
            <div style="background: #06A77D; padding: 10px 15px; border-radius: 8px; text-align: center; color: white; font-weight: bold;">
                🔴 LIVE
            </div>
        """, unsafe_allow_html=True)
    else:
        pass
    
    # Summary Metrics Row
    st.markdown("### 📈  ตัวเลขชี้วัด")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("จำนวนทั้งหมดยานพาหนะ", f"{int(total_vehicles):,}", help="Total vehicles detected")

    with col2:
        if use_live_metrics and detections:
            count = int(category_counts.get('motorcycle_tuk_tuk', 0))
        else:
            count = int(df['motorcycle_tuk_tuk'].sum()) if not df.empty else 0
        st.metric("🏍 Motorcycles, Tuk-tuk", f"{count:,}")

    with col3:
        if use_live_metrics and detections:
            count = int(category_counts.get('sedan_pickup_suv', 0))
        else:
            count = int(df['sedan_pickup_suv'].sum()) if not df.empty else 0
        st.metric("🚗 Sedan_Pickup_SUV", f"{count:,}")

    with col4:
        if use_live_metrics and detections:
            count = int(category_counts.get('van', 0))
        else:
            count = int(df['van'].sum()) if not df.empty else 0
        st.metric("🚌 Van", f"{count:,}")
    
    with col5:
        if use_live_metrics and detections:
            count = int(category_counts.get('minibus_bus', 0))
        else:
            count = int(df['minibus_bus'].sum()) if not df.empty else 0
        st.metric("🚌 Minibus_Bus", f"{count:,}")

    with col6:
        if use_live_metrics and detections:
            count = int(category_counts.get('truck6_truck10_trailer', 0))
        else:
            count = int(df['truck6_truck10_trailer'].sum()) if not df.empty else 0
        st.metric("🚛 Trucks", f"{count:,}")

    st.markdown("---")

    # Charts Section
    st.markdown("### 📊 การแสดงของกราฟ")
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_counts = create_vehicle_counts_chart(counts_data) if not use_live_metrics else None
        if not fig_counts and use_live_metrics and 'detections' in locals():
            # Create bar from detections
            if detections:
                det_df = pd.DataFrame(detections)
                det_df['category'] = det_df['vehicle_class'].apply(get_vehicle_category)
                cat_sums = det_df.groupby('category').size()
                display_names = {
                    "motorcycle_tuk_tuk": "🏍 Motorcycle/Tuk-Tuk",
                    "sedan_pickup_suv": "🚗 Sedan/Pickup/SUV",
                    "van": "🚙 Van",
                    "minibus_bus": "🚌 Minibus/Bus",
                    "truck6_truck10_trailer": "🚛 Truck/Trailer",
                }
                sums_dict = {display_names.get(cat, cat): count for cat, count in cat_sums.items()}
                fig_counts = px.bar(
                    x=list(sums_dict.keys()), y=list(sums_dict.values()),
                    title="นับจำนวนของแต่ละกลุ่มยานพาหนะ",
                    labels={"x": "Vehicle Type", "y": "Count"},
                    color=list(sums_dict.values()),
                    color_continuous_scale="Viridis",
                )
                fig_counts.update_layout(
                    xaxis_tickangle=-45, showlegend=False, height=400,
                    template="plotly_white", hovermode="x unified",
                    margin=dict(l=0, r=0, t=40, b=100)
                )
                fig_counts.update_traces(textposition='outside', text=list(sums_dict.values()))
        if fig_counts:
            st.plotly_chart(fig_counts, use_container_width=True, key="counts_chart")

    with chart_col2:
        fig_pie = create_vehicle_distribution_pie(counts_data) if not use_live_metrics else None
        if not fig_pie and use_live_metrics and 'detections' in locals():
            if detections:
                det_df = pd.DataFrame(detections)
                det_df['category'] = det_df['vehicle_class'].apply(get_vehicle_category)
                cat_sums = det_df.groupby('category').size()
                display_names = {
                    "motorcycle_tuk_tuk": "Motorcycle/Tuk-Tuk",
                    "sedan_pickup_suv": "Sedan/Pickup/SUV",
                    "van": "Van",
                    "minibus_bus": "Minibus/Bus",
                    "truck6_truck10_trailer": "Truck/Trailer",
                }
                sums_dict = {display_names.get(cat, cat): count for cat, count in cat_sums.items() if count > 0}
                fig_pie = px.pie(
                    values=list(sums_dict.values()), names=list(sums_dict.keys()),
                    title="การกระจายยานพาหนะ",
                    color_discrete_sequence=["#FF6B35", "#004E89", "#F7931E", "#06A77D", "#D62828"],
                )
                fig_pie.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    # Time Series Chart
    fig_time = create_time_series_chart(counts_data)
    if fig_time:
        st.plotly_chart(fig_time, use_container_width=True, key="time_chart")

    # Advanced Analytics Section
    if show_advanced:
        st.markdown("---")
        st.markdown("### 🔬 Advanced Analytics")

        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            fig_hourly = create_hourly_trends(counts_data)
            if fig_hourly:
                st.plotly_chart(fig_hourly, use_container_width=True, key="hourly_chart")

        with adv_col2:
            # Statistics table
            st.markdown("*Detection Statistics*")
            stats_df = pd.DataFrame({
                "Vehicle Type": ["Motorcycle/Tuk-Tuk", "Sedan/Pickup/SUV", "Van", "Minibus/Bus", "Truck/Trailer"],
                "Count": [
                    int(df["motorcycle_tuk_tuk"].sum()) if not df.empty else 0,
                    int(df["sedan_pickup_suv"].sum()) if not df.empty else 0,
                    int(df["van"].sum()) if not df.empty else 0,
                    int(df["minibus_bus"].sum()) if not df.empty else 0,
                    int(df["truck6_truck10_trailer"].sum()) if not df.empty else 0
                ]
            })
            stats_df["Percentage"] = (stats_df["Count"] / stats_df["Count"].sum() * 100).round(2).astype(str) + "%"
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Detailed Data Table
    st.markdown("---")
    st.markdown("### 📋 ข้อมูลการนับโดยละเอียด")
    
    with st.expander("ดูข้อมูลโดยละเอียด", expanded=False):
        if use_live_metrics and detections:
            # สำหรับ Live mode แสดงข้อมูล detections
            display_df = pd.DataFrame(detections)
            if not display_df.empty:
                display_df['ts'] = pd.to_datetime(display_df['ts']).dt.strftime("%Y-%m-%d %H:%M:%S")
                display_df = display_df[['ts', 'vehicle_class', 'conf', 'direction', 'snapshot_path']].sort_values('ts', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            # สำหรับโหมดอื่นแสดงข้อมูล counts
            display_df = df.copy()
            if not display_df.empty:
                if "start_ts" in display_df.columns:
                    display_df["start_ts"] = pd.to_datetime(display_df["start_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                if "end_ts" in display_df.columns:
                    display_df["end_ts"] = pd.to_datetime(display_df["end_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with footer_col2:
        if time_range == "Live (Real-time)":
            st.caption("📍 Live Mode - Real-time data")
        else:
            st.caption(f"📍 Time range: {from_time.strftime('%Y-%m-%d %H:%M')} to {to_time.strftime('%Y-%m-%d %H:%M')}")
    with footer_col3:
        if camera_id:
            st.caption(f"📹 Camera: {camera_id}")
        else:
            st.caption("📹 All cameras")


# ========================
# RUN APP
# ========================
if __name__ == "__main__":
    main()