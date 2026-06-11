# 🛡️ Ứng Dụng Web Phát Hiện Giao Dịch Gian Lận & Dự Báo Rủi Ro Tài Chính

Ứng dụng web này được phát triển bằng framework **Streamlit**, chuyển đổi hoàn chỉnh từ quy trình nghiên cứu, xây dựng và huấn luyện mô hình Machine Learning trong Notebook (`phat_hien_giao_dich_gian_lan.ipynb`). 

Hệ thống cho phép người dùng vận hành, phân tích dữ liệu giao dịch tài chính tự động, trực quan hóa và kiểm tra rủi ro trực tuyến theo mô hình AI.

## 🧮 Các mô hình được hỗ trợ trong ứng dụng
Dựa trên cấu trúc thực nghiệm từ mã nguồn notebook ban đầu, ứng dụng tích hợp cả 3 thuật toán phân loại:
1. **Random Forest Classifier** (Mặc định cấu hình tối ưu nhất)
2. **Decision Tree Classifier**
3. **Logistic Regression**

## 💻 Hướng dẫn Cài đặt & Chạy ứng dụng

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo bạn đã cài đặt phiên bản **Python (từ bản 3.9 đến 3.12)** trên hệ thống của mình.

### Bước 2: Cài đặt các thư viện phụ thuộc bắt buộc
Mở terminal hoặc command prompt tại thư mục chứa mã nguồn này và chạy lệnh:
```bash
pip install -r requirements.txt
