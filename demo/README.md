# Streamlit Prediction Demo

Demo app triển khai dạng nhập phản hồi và nhận dự đoán:

- Cảm xúc: negative / neutral / positive.
- Chủ đề: lecturer / training_program / facility / others.
- Auto Router: dùng PhoBERT mặc định, fallback sang TF-IDF char SVM nếu phát hiện thiếu dấu.

## 1. Điều kiện model artifacts

PhoBERT cần có:

```text
outputs/models/phobert/sentiment/best/
outputs/models/phobert/topic/best/
```

TF-IDF char SVM cần có:

```text
outputs/models/baseline/sentiment_tfidf_char_svm.joblib
outputs/models/baseline/topic_tfidf_char_svm.joblib
```

Nếu thiếu SVM, chạy:

```powershell
python scripts/export_char_svm_models.py
```

## 2. Cài thư viện

```powershell
conda activate viedufeedback
pip install streamlit torch transformers sentencepiece joblib scikit-learn
```

## 3. Chạy app

```powershell
streamlit run demo/app.py
```

## 4. Ghi chú

- App dự đoán thật cho câu người dùng nhập.
- Với Auto Router:
  - Câu có dấu bình thường: dùng PhoBERT.
  - Câu có khả năng thiếu dấu: dùng TF-IDF char SVM.
- Confidence của PhoBERT là softmax probability.
- Confidence của LinearSVC chỉ là pseudo-score từ decision margin, không phải xác suất xác thực.
