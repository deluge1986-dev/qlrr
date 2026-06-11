import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# 1. CẤU HÌNH PAGE STREAMLIT ĐẦU TIÊN
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# 2. IMPORT & CÁC HÀM CACHE DÙNG CHUNG
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo tính hashable cho st.cache_data"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        return None

# Định nghĩa danh sách các biến đặc trưng dựa theo notebook
X_COLS = [f"X_{i}" for i in range(1, 15)]
Y_COL = "default"

# 3. SIDEBAR (TP1) - VÙNG CẤU HÌNH
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu mẫu hoặc dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Tệp dữ liệu chứa các cột từ X_1 đến X_14 và cột nhãn 'default'"
    )
    
    st.divider()
    
    # Lựa chọn mô hình AI
    model_choice = st.selectbox(
        "Chọn thuật toán phân loại",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Lựa chọn mô hình toán học phù hợp để huấn luyện phát hiện gian lận."
    )
    
    st.subheader("Tham số mô hình AI")
    params = {}
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10)
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=15)
        params['random_state'] = st.number_input("Cài đặt Random State", value=42, step=1)
    
    elif model_choice == "Decision Tree":
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=10)
        params['criterion'] = st.selectbox("Tiêu chí đo lường (criterion)", options=["gini", "entropy", "log_loss"], index=0)
        params['random_state'] = st.number_input("Cài đặt Random State", value=42, step=1)
        
    elif model_choice == "Logistic Regression":
        params['max_iter'] = st.slider("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=5000, value=1000, step=100)
        params['C'] = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
        params['random_state'] = st.number_input("Cài đặt Random State", value=42, step=1)

    st.divider()
    test_size = st.slider("Tỷ lệ tập kiểm thử (Test size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    st.write("")
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# 4. HEADER (TP2) - VÙNG ĐỊNH HƯỚNG
st.title("🛡️ Ứng Dụng Phát Hiện Giao Dịch Gian Lận & Rủi Ro")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, tự động tìm kiếm hành vi bất thường và dự đoán rủi ro gian lận dựa trên Machine Learning.")

# Kiểm tra trạng thái tải file của người dùng
if uploaded_file is None:
    st.info("💡 Vui lòng tải lên file dữ liệu (.csv hoặc .xlsx) ở thanh cấu hình bên trái để bắt đầu khám phá và huấn luyện mô hình.")
    st.stop()

# Đọc dữ liệu từ file upload
df_global = load_data(uploaded_file.getvalue(), uploaded_file.name)

# Kiểm tra tính hợp lệ của DataFrame
if df_global is None or not isinstance(df_global, pd.DataFrame):
    st.error("❌ Không thể phân tích tệp dữ liệu. Vui lòng kiểm tra lại định dạng hoặc cấu trúc file đầu vào.")
    st.stop()

# Kiểm tra schema dữ liệu bắt buộc
missing_cols = [col for col in X_COLS + [Y_COL] if col not in df_global.columns]
if missing_cols:
    st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang sử dụng tệp: **{uploaded_file.name}** | Quy mô dữ liệu: **{df_global.shape[0]:,}** dòng, **{df_global.shape[1]}** cột.")
st.divider()

# 5. KHỐI HUẤN LUYỆN (Chạy khi bấm nút, lưu toàn bộ kết quả vào session_state)
if btn_train:
    with st
