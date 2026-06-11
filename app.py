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
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
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
    
    # Lựa chọn mô hình AI (Notebook có 3 mô hình)
    model_choice = st.selectbox(
        "Chọn thuật toán phân loại",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Lựa chọn mô hình toán học phù hợp để huấn luyện phát hiện gian lận."
    )
    
    st.subheader("Tham số mô hình AI")
    # Thay đổi tham số động theo mô hình được lựa chọn
    params = {}
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=300, value=100, step=10, help="Số lượng cây quyết định trong rừng.")
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=15, help="Độ sâu tối đa của mỗi cây quyết định.")
        params['random_state'] = st.number_input("Cài đặt Random State", value=42, step=1, help="Đảm bảo kết quả huấn luyện có thể tái lập.")
    
    elif model_choice == "Decision Tree":
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=50, value=10, help="Độ sâu giới hạn của cây quyết định nhằm tránh overfitting.")
        params['criterion'] = st.selectbox("Tiêu chí đo lường (criterion)", options=["gini", "entropy", "log_loss"], index=0, help="Hàm đo lường chất lượng phân tách.")
        params['random_state'] = st.number_input("Cài đặt Random State", value=42, step=1, help="Đảm bảo kết quả huấn luyện có thể tái lập.")
        
    elif model_choice == "Logistic Regression":
        params['max_iter'] = st.slider("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=5000, value=1000, step=100, help="Số lượng vòng lặp tối đa cho thuật toán tối ưu hội tụ.")
        params['C'] = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05, help="Giá trị C càng nhỏ thì phạt chính quy hóa càng mạnh.")
        params['random_state'] = st.number_input("Cài đặt Random State", value=42, step=1, help="Đảm bảo kết quả huấn luyện có thể tái lập.")

    st.divider()
    
    # Tỷ lệ chia Train/Test tập dữ liệu
    test_size = st.slider("Tỷ lệ tập kiểm thử (Test size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05, help="Tỷ lệ chia dữ liệu để đánh giá chất lượng mô hình.")
    
    # Nút bấm hành động duy nhất để kích hoạt huấn luyện
    st.write("")
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True, help="Bấm để bắt đầu trích xuất đặc trưng và fit mô hình.")

# 4. HEADER (TP2) - VÙNG ĐỊNH HƯỚNG
st.title("🛡️ Ứng Dụng Phát Hiện Giao Dịch Gian Lận & Rủi Ro")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, tự động tìm kiếm hành vi bất thường và dự đoán rủi ro gian lận dựa trên Machine Learning.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên file dữ liệu (.csv hoặc .xlsx) ở thanh cấu hình bên trái để bắt đầu khám phá và huấn luyện mô hình.")
    st.stop()

# Đọc dữ liệu khi đã upload thành công
df_data = load_data(uploaded_file.getvalue(), uploaded_file.name)

if df_data is None:
    st.error("Không thể đọc tệp dữ liệu. Vui lòng kiểm tra lại định dạng file.")
    st.stop()

# Kiểm tra schema dữ liệu bắt buộc
missing_cols = [col for col in X_COLS + [Y_COL] if col not in df_data.columns]
if missing_cols:
    st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang sử dụng tệp: **{uploaded_file.name}** | Quy mô dữ liệu: **{df_data.shape[0]:,}** dòng, **{df_data.shape[1]}** cột.")
st.divider()

# 5. KHỐI HUẤN LUYỆN (Chạy khi bấm nút, lưu kết quả vào session_state)
if btn_train:
    with st.spinner("⏳ Đang xử lý dữ liệu và huấn luyện mô hình..."):
        # Chuẩn bị X, y
        X = df_data[X_COLS]
        y = df_data[Y_COL]
        
        # Chia tập dữ liệu giống quy trình trong notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(params.get('random_state', 42)), stratify=y
        )
        
        # Khởi tạo mô hình theo lựa chọn
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=int(params['random_state'])
            )
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=params['max_depth'],
                criterion=params['criterion'],
                random_state=int(params['random_state'])
            )
        else:
            model = LogisticRegression(
                max_iter=params['max_iter'],
                C=params['C'],
                random_state=int(params['random_state'])
            )
        
        # Fit mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán và tính toán metrics
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu trữ mọi thông tin cần thiết vào session_state
        st.session_state['trained_model'] = model
        st.session_state['model_name'] = model_choice
        st.session_state['features_list'] = X_COLS
        
        # Tính toán bộ chỉ số kiểm định
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
        st.success(f"🎉 Huấn luyện thành công mô hình **{model_choice}**! Hãy chuyển sang các Tab bên dưới để xem kết quả chi tiết.")

# 6. KHỞI TẠO CÁC TABS GIAO DIỆN CHÍNH
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
        st.metric("Tổng số dòng (Số lượng giao dịch)", f"{df_data.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng biến độc lập (X)", len(X_COLS))
    with col_m3:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("Dung lượng tệp dữ liệu", f"{file_size_mb:.2f} MB")
        
    st.write("#### 🔎 Xem trước dữ liệu (5 dòng đầu tiên)")
    st.dataframe(df_data.head(5), use_container_width=True)
    
    st.write("#### 📉 Thống kê mô tả các biến đặc trưng đưa vào mô hình")
    # Chỉ mô tả các biến mô hình quan tâm theo quy tắc thiết kế
    st.dataframe(df_data[X_COLS + [Y_COL]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("📊 Phân bổ dữ liệu & Sự tương quan")
    
    # 1. Vẽ biến mục tiêu trước (Quy tắc ưu tiên biến y)
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.write("##### Phân phối nhãn mục tiêu (default)")
        target_counts = df_data[Y_COL].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: '0: Hợp lệ', 1: '1: Gian lận'})
        fig_target = px.bar(
            target_counts, x='Trạng thái', y='Số lượng', 
            color='Trạng thái', color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=True, height=350
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with row1_col2:
        st.write("##### Biểu đồ phân bổ của biến quan trọng bậc nhất (X_1)")
        fig_x1 = px.histogram(
            df_data, x="X_1", color=Y_COL, 
            barmode="overlay", marginal="box",
            color_discrete_sequence=["#2b5c8f", "#d95f02"], height=350
        )
        st.plotly_chart(fig_x1, use_container_width=True)

    # Cho người dùng tùy chọn biến để hiển thị lưới 2x2 linh hoạt
    st.write("##### 🛠️ Tùy chọn xem biểu đồ phân bổ phân tán các biến đặc trưng khác")
    selected_features = st.multiselect(
        "Chọn các biến độc lập để vẽ phân bổ (Mặc định chọn 2 biến đầu):",
        options=X_COLS,
        default=["X_2", "X_3"]
    )
    
    if len(selected_features) > 0:
        row2_cols = st.columns(len(selected_features))
        for idx, feat in enumerate(selected_features):
            with row2_cols[idx % len(selected_features)]:
                st.write(f"##### Biểu đồ phân bổ biến {feat}")
                fig_feat = px.histogram(
                    df_data, x=feat, color=Y_COL,
                    marginal="violin", barmode="histogram", height=350
                )
                st.plotly_chart(fig_feat, use_container_width=True)

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab3:
    st.subheader("🎯 Đánh giá hiệu năng thuật toán trên tập Test")
    
    # Kiểm tra trạng thái đã bấm huấn luyện chưa
    if 'metrics' not in st.session_state:
        st.info("📢 Vui lòng bấm vào nút **[🚀 Huấn luyện mô hình]** tại thanh công cụ Sidebar bên trái để chạy thuật toán và xem báo cáo kiểm định.")
    else:
        metrics = st.session_state['metrics']
        current_model_name = st.session_state['model_name']
        
        st.markdown(f"Đang hiển thị kết quả kiểm định của mô hình: **{current_model_name}**")
        
        # Trình bày chỉ tiêu vô hướng qua st.metric
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        with c_m2:
            st.metric("Độ chính xác xác thực (Precision)", f"{metrics['precision']:.4f}")
        with c_m3:
            st.metric("Tỷ lệ bỏ sót rủi ro (Recall)", f"{metrics['recall']:.4f}")
        with c_m4:
            st.metric("Chỉ số F1-Score", f"{metrics['f1']:.4f}")
            
        st.divider()
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write("##### 📊 Ma Trận Nhầm Lẫn (Confusion Matrix)")
            cm_data = metrics['cm']
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['0 (Hợp lệ)', '1 (Gian lận)'],
                y=['0 (Hợp lệ)', '1 (Gian lận)'],
                color_continuous_scale="Blues",
                height=380
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with res_col2:
            st.write("##### 📈 Đường cong ROC-AUC")
            if metrics['y_probs'] is not None:
                fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_probs'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})', line=dict(color='darkorange', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Dự đoán ngẫu nhiên', line=dict(color='navy', width=2, dash='dash')))
                fig_roc.update_layout(
                    xaxis_title='Tỷ lệ Dương tính giả (FPR)',
                    yaxis_title='Tỷ lệ Dương tính thật (TPR)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=380,
                    legend=dict(x=0.5, y=0.1)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("Mô hình được chọn không hỗ trợ xuất xác suất phân lớp để vẽ ROC-AUC.")

# --- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ---
with tab4:
    st.subheader("🔮 Phòng ngừa rủi ro real-time / Dự báo hàng loạt")
    
    if 'trained_model' not in st.session_state:
        st.info("📢 Vui lòng bấm huấn luyện mô hình ở Sidebar trước khi thực hiện chức năng dự báo rủi ro giao dịch.")
        st.stop()
        
    model = st.session_state['trained_model']
    
    # Lựa chọn chế độ dự báo bằng st.radio
    predict_mode = st.radio(
        "Phương thức nhập dữ liệu đầu vào:",
        options=["Nhập thông số trực tiếp (Đơn lẻ)", "Tải tệp danh sách cần chấm điểm (Hàng loạt)"],
        horizontal=True
    )
    
    if predict_mode == "Nhập thông số trực tiếp (Đơn lẻ)":
        st.write("##### 🛠️ Điền các chỉ số kỹ thuật của giao dịch mới")
        
        # Tạo giao diện nhập liệu tự động cho 14 biến, giá trị mặc định lấy từ median của tập dữ liệu
        input_data = {}
        
        # Bố trí các ô nhập liệu dạng lưới 3 cột cho gọn gàng
        form_cols = st.columns(3)
        for idx, col_name in enumerate(X_COLS):
            col_pos = form_cols[idx % 3]
            with col_pos:
                default_val = float(df_data[col_name].median())
                min_val = float(df_data[col_name].min())
                max_val = float(df_data[col_name].max())
                
                input_data[col_name] = st.number_input(
                    f"Giá trị đặc trưng {col_name}",
                    min_value=min_val - 10.0,
                    max_value=max_value + 10.0,
                    value=default_val,
                    format="%.5f",
                    help=f"Nhập thông số cho biến {col_name}. Khoảng dữ liệu gốc: [{min_val:.2f} -> {max_val:.2f}]"
                )
        
        st.write("")
        btn_predict_single = st.button("🔍 Kiểm tra rủi ro giao dịch", type="primary")
        
        if btn_predict_single:
            # Chuyển dữ liệu sang DataFrame đúng định dạng
            input_df = pd.DataFrame([input_data])
            
            # Dự báo trực tiếp từ mô hình
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            
            st.subheader("Kết quả phân tích:")
            if prediction == 1:
                st.error("🚨 **CẢNH BÁO: Giao dịch này có dấu hiệu GIAN LẬN hoặc RỦI RO CAO!**")
            else:
                st.success("✅ **AN TOÀN: Giao dịch được đánh giá là HỢP LỆ.**")
                
            if prob is not None:
                st.metric(label="Xác suất rủi ro được tính toán", value=f"{prob*100:.2f} %")

    else:
        st.write("##### 📁 Tải lên file chứa danh sách giao dịch mới để xử lý hàng loạt")
        st.caption("Lưu ý: Tệp tải lên phải định dạng .xlsx hoặc .csv và chứa đầy đủ 14 cột biến đặc trưng từ `X_1` đến `X_14` giống cấu trúc gốc.")
        
        batch_file = st.file_uploader("Tải tệp cần chấm điểm dự báo", type=["csv", "xlsx"], key="batch_uploader")
        
        if batch_file is not None:
            df_batch = load_data(batch_file.getvalue(), batch_file.name)
            if df_batch is not None:
                # Kiểm tra schema đặc trưng X
                missing_batch_cols = [c for c in X_COLS if c not in df_batch.columns]
                if missing_batch_cols:
                    st.error(f"❌ Tệp tải lên thiếu các cột biến độc lập sau: {', '.join(missing_batch_cols)}")
                else:
                    # Tiến hành dự báo hàng loạt
                    X_batch = df_batch[X_COLS]
                    batch_preds = model.predict(X_batch)
                    
                    # Thêm cột kết quả vào dataframe
                    df_result = df_batch.copy()
                    df_result['Du_Bao_Default'] = batch_preds
                    df_result['Trạng_Thái_Giao_Dịch'] = df_result['Du_Bao_Default'].map({0: 'Hợp lệ', 1: '🚨 Gian lận/Rủi ro'})
                    
                    if hasattr(model, "predict_proba"):
                        df_result['Xac_Suat_Gian_Lan'] = model.predict_proba(X_batch)[:, 1]
                    
                    st.success(f"⚡ Đã chấm điểm thành công cho toàn bộ {df_result.shape[0]} bản ghi giao dịch!")
                    
                    # Thống kê nhanh kết quả vừa dự báo
                    fraud_count = int((batch_preds == 1).sum())
                    st.metric("Phát hiện số giao dịch rủi ro tiềm ẩn", f"{fraud_count} / {len(batch_preds)}")
                    
                    # Hiển thị kết quả trong container cuộn gọn gàng
                    st.write("##### Xem bảng kết quả chi tiết:")
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Xuất kết quả trả về dạng CSV cho người dùng tải về
                    csv_data = df_result.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Tải xuống kết quả dự báo (.CSV)",
                        data=csv_data,
                        file_name="ket_qua_du_bao_gian_lan.csv",
                        mime="text/csv"
                    )
