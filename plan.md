# Kế hoạch thực nghiệm

| Stage | Nội dung | Output chính |
|---|---|---|
| 1 | Kiểm tra và chuẩn hóa dataset | split summary, label mapping |
| 2 | EDA | phân phối nhãn, độ dài văn bản |
| 3 | Baseline | Majority, TF-IDF word SVM, TF-IDF char SVM |
| 4 | Noisy test | clean, no_accent, typo, teencode, mixed |
| 5 | Baseline robustness | Macro-F1 drop theo từng noise |
| 6 | PhoBERT | clean result, noisy result, prediction |
| 7 | Final comparison | bảng so sánh, error examples |
| 8 | Robust inference | PhoBERT + char SVM fallback |
| Demo | Streamlit app | nhập phản hồi và dự đoán nhãn |
