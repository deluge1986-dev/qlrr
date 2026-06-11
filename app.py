import streamlit as st
import pandas as pd
import numpy as np
import io  
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ==========================================
# 1. CẤU HÌNH TRANG & TIÊU ĐỀ ỨNG DỤNG (CHỈ XUẤT HIỆN 1 LẦN)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện gian lận tại Agribank",
    page_icon="😊"
)

st.title("*Ứng Dụng Phát Hiện Giao Dịch Gian Lận & Rủi Ro*")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, tự động tìm kiếm hành vi bất thường và dự đoán rủi ro gian lận dựa trên Machine Learning.")
st.divider()

# ==========================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU CACHE
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes thông qua io.BytesIO để Pandas phân tích chính xác"""
    try:
        data_stream = io.BytesIO(file_bytes)
        if file_name.endswith('.csv'):
            df = pd.read_csv(data_stream)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_stream)
        else:
            return None
        return df
    except Exception as e:
        return None

# Định nghĩa danh sách các biến đặc trưng dựa theo notebook
X_COLS = [f"X_{i}" for i in range(1, 15)]
Y_COL = "default"

# ==========================================
# 3. SIDEBAR - VÙNG CẤU HÌNH THAM SỐ
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Tệp dữ liệu chứa các cột từ X_1 đến X_14 và cột nhãn 'default'"
    )
    
    st.divider()
    
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

# ==========================================
# 4. KIỂM TRA TRẠNG THÁI FILE & VALIDATE SCHEMA
# ==========================================
if uploaded_file is None:
    st.info("💡 Vui lòng tải lên file dữ liệu (.csv hoặc .xlsx) ở thanh cấu hình bên trái để bắt đầu khám phá và huấn luyện mô hình.")
    st.stop()

# Đọc dữ liệu từ file upload
df_global = load_data(uploaded_file.getvalue(), uploaded_file.name)

if df_global is None or not isinstance(df_global, pd.DataFrame):
    st.error("❌ Không thể phân tích tệp dữ liệu. Vui lòng kiểm tra lại định dạng hoặc cấu trúc file đầu vào.")
    st.stop()

missing_cols = [col for col in X_COLS + [Y_COL] if col not in df_global.columns]
if missing_cols:
    st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
    st.stop()

st.success(f"📁 Đang sử dụng tệp: **{uploaded_file.name}** | Quy mô dữ liệu: **{df_global.shape[0]:,}** dòng, **{df_global.shape[1]}** cột.")

# ==========================================
# 5. KHỐI LOGIC HUẤN LUYỆN MÔ HÌNH
# ==========================================
if btn_train:
    with st.spinner("⏳ Đang xử lý dữ liệu và huấn luyện mô hình..."):
        X = df_global[X_COLS]
        y = df_global[Y_COL]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(params.get('random_state', 42)), stratify=y
        )
        
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=int(params['random_state']))
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=params['max_depth'], criterion=params['criterion'], random_state=int(params['random_state']))
        else:
            model = LogisticRegression(max_iter=params['max_iter'], C=params['C'], random_state=int(params['random_state']))
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu trữ vào bộ nhớ phiên làm việc
        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_choice
        st.session_state['features_list'] = X_COLS
        st.session_state['show_success_alert'] = True # Cờ đánh dấu hiển thị thông báo hợp lý
        
        st.session_state['metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'cm': confusion_matrix(y_test, y_pred),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_probs': y_probs.tolist() if y_probs is not None else None
        }

# ==========================================
# 6. GIAO DIỆN CHÍNH - PHÂN PHÒNG CÁC TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Đánh giá mô hình", 
    "🔮 Triển khai dự báo từ mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    st.subheader("📝 Phân tích cấu trúc dữ liệu thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tổng số dòng (Số lượng giao dịch)", f"{df_global.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng biến độc lập (X)", len(X_COLS))
    with col_m3:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("Dung lượng tệp dữ liệu", f"{file_size_mb:.2f} MB")
        
    st.write("#### 🔎 Xem trước dữ liệu (5 dòng đầu tiên)")
    st.dataframe(df_global.head(5), use_container_width=True)
    
    st.write("#### 📉 Thống kê mô tả các biến đặc trưng đưa vào mô hình")
    st.dataframe(df_global[X_COLS + [Y_COL]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("📊 Phân bổ dữ liệu & Sự tương quan")
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.write("##### Phân phối nhãn mục tiêu (default)")
        target_counts = df_global[Y_COL].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: '0: Hợp lệ', 1: '1: Gian lận'})
        fig_target = px.bar(target_counts, x='Trạng thái', y='Số lượng', color='Trạng thái', text_auto=True, height=350)
        st.plotly_chart(fig_target, use_container_width=True)
        
    with row1_col2:
        st.write("##### Biểu đồ phân bổ của biến quan trọng bậc nhất (X_1)")
        fig_x1 = px.histogram(df_global, x="X_1", color=Y_COL, barmode="overlay", marginal="box", height=350)
        st.plotly_chart(fig_x1, use_container_width=True)

    st.write("##### 🛠️ Tùy chọn xem biểu đồ phân bổ phân tán các biến đặc trưng khác")
    selected_features = st.multiselect("Chọn các biến độc lập để vẽ phân bổ:", options=X_COLS, default=["X_2", "X_3"])
    
    if len(selected_features) > 0:
        row2_cols = st.columns(len(selected_features))
        for idx, feat in enumerate(selected_features):
            with row2_cols[idx % len(selected_features)]:
                st.write(f"##### Biểu đồ phân bổ biến {feat}")
                fig_feat = px.histogram(df_global, x=feat, color=Y_COL, marginal="violin", barmode="histogram", height=350)
                st.plotly_chart(fig_feat, use_container_width=True)

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng thuật toán trên tập Test")
    
    if 'metrics' not in st.session_state:
        st.info("📢 Vui lòng bấm vào nút **[🚀 Huấn luyện mô hình]** tại thanh công cụ Sidebar bên trái để chạy thuật toán và xem báo cáo kiểm định.")
        st.stop()
    
    # Đưa thông báo thành công vào đúng vị trí đầu trang kết quả kiểm định
    if st.session_state.get('show_success_alert', False):
        st.success(f"🎉 Huấn luyện thành công mô hình **{st.session_state['model_name']}**!")
        st.session_state['show_success_alert'] = False # Tắt cờ sau khi hiển thị xong
        
    metrics = st.session_state['metrics']
    current_model_name = st.session_state['model_name']
    st.markdown(f"Đang hiển thị kết quả kiểm định của mô hình: **{current_model_name}**")
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1: st.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
    with c_m2: st.metric("Độ chính xác xác thực (Precision)", f"{metrics['precision']:.4f}")
    with c_m3: st.metric("Tỷ lệ bỏ sót rủi ro (Recall)", f"{metrics['recall']:.4f}")
    with c_m4: st.metric("Chỉ số F1-Score", f"{metrics['f1']:.4f}")
        
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.write("##### 📊 Ma Trận Nhầm Lẫn (Confusion Matrix)")
        fig_cm = px.imshow(metrics['cm'], text_auto=True, x=['0 (Hợp lệ)', '1 (Gian lận)'], y=['0 (Hợp lệ)', '1 (Gian lận)'], color_continuous_scale="Blues", height=380)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with res_col2:
        st.write("##### 📈 Đường cong ROC-AUC")
        if metrics['y_probs'] is not None:
            fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_probs'])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Ngẫu nhiên'))
            fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_roc, use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ---
with tab4:
    st.subheader("🔮 Phòng ngừa rủi ro real-time / Dự báo hàng loạt")
    
    if 'trained_model' not in st.session_state:
        st.info("📢 Vui lòng bấm vào nút **[🚀 Huấn luyện mô hình]** tại thanh công cụ Sidebar bên trái trước khi thực hiện chức năng dự báo rủi ro giao dịch.")
        st.stop()
    
    model = st.session_state['trained_model']
    predict_mode = st.radio("Phương thức nhập dữ liệu đầu vào:", options=["Nhập thông số trực tiếp (Đơn lẻ)", "Tải tệp danh sách cần chấm điểm (Hàng loạt)"], horizontal=True)
    
    if predict_mode == "Nhập thông số trực tiếp (Đơn lẻ)":
        st.write("##### 🛠️ Điền các chỉ số kỹ thuật của giao dịch mới")
        input_data = {}
        form_cols = st.columns(3)
        for idx, col_name in enumerate(X_COLS):
            col_pos = form_cols[idx % 3]
            with col_pos:
                default_val = float(df_global[col_name].median())
                input_data[col_name] = st.number_input(f"Giá trị đặc trưng {col_name}", value=default_val, format="%.5f")
        
        st.write("")
        if st.button("🔍 Kiểm tra rủi ro giao dịch", type="primary"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            
            if prediction == 1:
                st.error("🚨 **CẢNH BÁO: Giao dịch có dấu hiệu GIAN LẬN hoặc RỦI RO CAO!**")
            else:
                st.success("✅ **AN TOÀN: Giao dịch được đánh giá hợp lệ.**")
            if prob is not None:
                st.metric(label="Xác suất rủi ro", value=f"{prob*100:.2f} %")
    else:
        batch_file = st.file_uploader("Tải tệp cần chấm điểm dự báo", type=["csv", "xlsx"])
        if batch_file is not None:
            df_batch = load_data(batch_file.getvalue(), batch_file.name)
            if df_batch is not None:
                missing_batch_cols = [c for c in X_COLS if c not in df_batch.columns]
                if missing_batch_cols:
                    st.error(f"❌ Tệp thiếu các cột biến đặc trưng bắt buộc: {', '.join(missing_batch_cols)}")
                else:
                    X_batch = df_batch[X_COLS]
                    batch_preds = model.predict(X_batch)
                    df_result = df_batch.copy()
                    df_result['Du_Bao_Default'] = batch_preds
                    if hasattr(model, "predict_proba"):
                        df_result['Xac_Suat_Gian_Lan'] = model.predict_proba(X_batch)[:, 1]
                    st.success("⚡ Đã chấm điểm thành công cho danh sách giao dịch mới!")
                    st.dataframe(df_result, use_container_width=True)
