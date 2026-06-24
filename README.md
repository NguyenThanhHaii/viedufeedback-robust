# ViEduFeedback-Robust

## Tên đề tài

**Đánh giá độ bền của PhoBERT trong phân tích cảm xúc và phân loại chủ đề phản hồi sinh viên tiếng Việt phi chuẩn**

## Mục tiêu

Dự án tập trung vào bài toán phân loại phản hồi sinh viên tiếng Việt với hai tác vụ chính:

1. Phân tích cảm xúc.
2. Phân loại chủ đề.

Mô hình chính được sử dụng là PhoBERT. Ngoài đánh giá trên dữ liệu sạch, dự án tạo các tập kiểm thử nhiễu có kiểm soát để phân tích mức suy giảm hiệu năng khi văn bản đầu vào có các hiện tượng phi chuẩn như thiếu dấu, viết tắt, teencode, lỗi gõ và kéo dài ký tự.

## Phạm vi

Dự án tập trung vào core NLP:

- Dataset verification
- EDA
- Baseline truyền thống
- Fine-tuning PhoBERT
- Noisy test generation
- Robustness evaluation
- Error analysis

Dự án không tập trung vào frontend, backend, Docker hoặc deploy trong phạm vi core.

## Cấu trúc thư mục

`	ext
viedufeedback-robust/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── noisy/
├── notebooks/
├── kaggle/
├── src/
├── scripts/
├── outputs/
└── report/
Môi trường chạy

Local dùng cho:

Kiểm tra dữ liệu
EDA
Baseline
Tạo noisy test
Tổng hợp kết quả
Viết báo cáo

Kaggle T4x2 dùng cho:

Fine-tune PhoBERT
Evaluate PhoBERT trên clean/noisy test
Trạng thái
 Stage 0: Repo setup
 Stage 1: Dataset verification
 Stage 2: EDA
 Stage 3: Baseline
 Stage 4: PhoBERT clean
 Stage 5: Noisy test
 Stage 6: Robustness evaluation
 Stage 7: Error analysis
 Stage 8: Report and slides
