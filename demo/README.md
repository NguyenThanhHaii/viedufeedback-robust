# Demo

Streamlit app dùng để nhập phản hồi sinh viên và dự đoán:

- cảm xúc phản hồi
- chủ đề phản hồi
- câu có bị nghi thiếu dấu hay không
- mô hình được dùng trong chế độ tự động

## Chuẩn bị

Cài thư viện:

```powershell
conda activate viedufeedback
pip install -r demo/requirements_demo.txt
```

Cần có PhoBERT fine-tuned:

```text
outputs/models/phobert/sentiment/best/
outputs/models/phobert/topic/best/
```

Cần có TF-IDF char SVM fallback:

```text
outputs/models/baseline/sentiment_tfidf_char_svm.joblib
outputs/models/baseline/topic_tfidf_char_svm.joblib
```

Nếu thiếu SVM fallback:

```powershell
python scripts/export_char_svm_models.py
```

## Chạy app

```powershell
streamlit run demo/app.py
```

Hoặc:

```powershell
.\demo\run_demo.ps1
```

## Ví dụ

```text
giảng viên rất nhiệt tình, bài giảng dễ hiểu
```

```text
giang vien rat nhiet tinh nhung slide hoi kho hieu
```

Chế độ mặc định là **Tự động chọn mô hình**: PhoBERT được dùng cho câu bình thường, TF-IDF char SVM được dùng khi câu có khả năng thiếu dấu.
