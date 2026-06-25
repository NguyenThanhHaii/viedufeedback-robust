# ViEduFeedback Robust

Dự án đánh giá độ bền của các mô hình NLP trong bài toán phân tích phản hồi sinh viên tiếng Việt, tập trung vào hai tác vụ:

- **Phân tích cảm xúc**: negative, neutral, positive.
- **Phân loại chủ đề**: lecturer, training_program, facility, others.

Dự án so sánh các baseline truyền thống với PhoBERT, sau đó đánh giá độ bền của mô hình khi văn bản đầu vào bị nhiễu như thiếu dấu, typo, teencode và nhiễu kết hợp. Kết quả cuối cùng được dùng để xây dựng một chiến lược suy luận bền vững: **dùng PhoBERT mặc định, nhưng chuyển sang TF-IDF char SVM khi phát hiện văn bản có khả năng thiếu dấu**.

---

## 1. Mục tiêu dự án

Dự án trả lời các câu hỏi chính:

1. PhoBERT có vượt các baseline truyền thống trên dữ liệu sạch không?
2. Loại nhiễu nào làm mô hình suy giảm mạnh nhất?
3. PhoBERT có luôn bền hơn baseline trên văn bản phi chuẩn không?
4. Có thể cải thiện hệ thống suy luận bằng cách kết hợp PhoBERT và TF-IDF char SVM không?

---

## 2. Dữ liệu

Dự án sử dụng bộ dữ liệu phản hồi sinh viên tiếng Việt `uitnlp/vietnamese_students_feedback`.

Sau bước chuẩn hóa, dữ liệu gồm ba split:

| Split      | Số mẫu |
| ---------- | -----: |
| Train      | 11,426 |
| Validation |  1,583 |
| Test       |  3,166 |

Schema chuẩn hóa chính:

| Cột               | Ý nghĩa                     |
| ----------------- | --------------------------- |
| `text`            | Nội dung phản hồi sinh viên |
| `sentiment_label` | Nhãn cảm xúc                |
| `sentiment_name`  | Tên nhãn cảm xúc            |
| `topic_label`     | Nhãn chủ đề                 |
| `topic_name`      | Tên nhãn chủ đề             |

Nhãn cảm xúc:

|  ID | Label    |
| --: | -------- |
|   0 | negative |
|   1 | neutral  |
|   2 | positive |

Nhãn chủ đề:

|  ID | Label            |
| --: | ---------------- |
|   0 | lecturer         |
|   1 | training_program |
|   2 | facility         |
|   3 | others           |

---

## 3. Pipeline thực nghiệm

Dự án được tổ chức theo các stage:

| Stage   | Nội dung                                                |
| ------- | ------------------------------------------------------- |
| Stage 1 | Dataset verification                                    |
| Stage 2 | EDA                                                     |
| Stage 3 | Baseline models                                         |
| Stage 4 | Noisy test generation                                   |
| Stage 5 | Baseline robustness evaluation                          |
| Stage 6 | PhoBERT fine-tuning and robustness evaluation           |
| Stage 7 | Final comparison and error analysis                     |
| Stage 8 | Robust inference strategy                               |
| Demo    | Streamlit app nhập phản hồi và dự đoán cảm xúc + chủ đề |

---

## 4. Cấu trúc thư mục

```text
viedufeedback-robust/
├── configs/
│   ├── eda.yaml
│   ├── noise.yaml
│   ├── robustness.yaml
│   ├── phobert_sentiment.yaml
│   ├── phobert_topic.yaml
│   ├── final_analysis.yaml
│   └── robust_inference.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── noisy/
│
├── demo/
│   ├── app.py
│   ├── inference_utils.py
│   ├── README.md
│   ├── requirements_demo.txt
│   └── run_demo.ps1
│
├── kaggle/
│   └── stage6_train_phobert.py
│
├── notebooks/
│   ├── 01_dataset_verification.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_train_baselines.ipynb
│   ├── 04_noisy_test_generation.ipynb
│   ├── 05_baseline_robustness.ipynb
│   ├── 06_phobert_kaggle_training_clean.ipynb
│   ├── 06_phobert_results_analysis.ipynb
│   ├── 07_final_comparison_error_analysis.ipynb
│   └── 08_robust_inference_strategy.ipynb
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   ├── reports/
│   └── tables/
│
├── scripts/
│   ├── generate_noisy_validation.py
│   ├── evaluate_robust_inference.py
│   └── export_char_svm_models.py
│
└── src/
    ├── baseline.py
    ├── eda.py
    ├── final_analysis.py
    ├── noise.py
    ├── robust_inference.py
    └── robustness.py
```

---

## 5. Cài đặt môi trường

Tạo môi trường Conda:

```powershell
conda create -n viedufeedback python=3.10 -y
conda activate viedufeedback
```

Cài thư viện cơ bản:

```powershell
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml joblib
pip install transformers torch sentencepiece accelerate
pip install streamlit
```

Hoặc riêng cho demo:

```powershell
pip install -r demo/requirements_demo.txt
```

---

## 6. Chạy pipeline chính

### 6.1. Dataset verification

```powershell
python scripts/verify_dataset.py
```

Output chính:

```text
data/raw/
data/processed/
outputs/tables/
outputs/reports/
```

### 6.2. EDA

```powershell
python scripts/run_eda.py
```

Output chính:

```text
outputs/tables/eda_*.csv
outputs/figures/*.png
```

### 6.3. Train baseline

```powershell
python scripts/train_baselines.py
```

Baseline gồm:

- Majority classifier.
- TF-IDF word-level + LinearSVC.
- TF-IDF char-level + LinearSVC.

### 6.4. Sinh noisy test

```powershell
python scripts/generate_noisy_tests.py
```

Các biến thể nhiễu:

```text
clean
no_accent
typo_light
typo_medium
teencode_light
mixed_light
mixed_no_accent
```

### 6.5. Đánh giá baseline robustness

```powershell
python scripts/evaluate_baseline_robustness.py
```

### 6.6. Fine-tune PhoBERT

PhoBERT được train trên Kaggle bằng script:

```text
kaggle/stage6_train_phobert.py
```

Notebook chạy Kaggle:

```text
notebooks/06_phobert_kaggle_training_clean.ipynb
```

Sau khi train xong, cần có các output:

```text
outputs/models/phobert/sentiment/best/
outputs/models/phobert/topic/best/
outputs/tables/phobert_clean_results.csv
outputs/tables/phobert_robustness_results.csv
outputs/predictions/phobert_robustness_predictions.csv
```

### 6.7. Final comparison and error analysis

```powershell
python scripts/run_final_comparison.py
```

Output chính:

```text
outputs/tables/final_model_comparison.csv
outputs/tables/final_robustness_comparison.csv
outputs/tables/final_per_class_comparison.csv
outputs/tables/error_examples_sentiment.csv
outputs/tables/error_examples_topic.csv
outputs/reports/final_comparison_report.md
```

### 6.8. Robust inference strategy

```powershell
python scripts/generate_noisy_validation.py
python scripts/evaluate_robust_inference.py
```

Output chính:

```text
outputs/tables/noisy_validation_detection_thresholds.csv
outputs/tables/robust_inference_results.csv
outputs/tables/robust_inference_drop.csv
outputs/tables/robust_inference_comparison.csv
outputs/tables/robust_inference_detector_by_variant.csv
outputs/predictions/robust_inference_predictions.csv
outputs/reports/robust_inference_report.md
```

---

## 7. Kết quả chính

### 7.1. Clean test

| Task      | Model tốt nhất | Accuracy | Macro-F1 |
| --------- | -------------- | -------: | -------: |
| Sentiment | PhoBERT        |   0.9315 |   0.8228 |
| Topic     | PhoBERT        |   0.8964 |   0.8001 |

Kết luận: PhoBERT vượt các baseline truyền thống trên dữ liệu sạch ở cả hai tác vụ.

### 7.2. Robustness với văn bản thiếu dấu

Trên các biến thể thiếu dấu, PhoBERT suy giảm mạnh, trong khi TF-IDF char SVM bền hơn.

| Task      | Variant         | PhoBERT Macro-F1 | TF-IDF char SVM Macro-F1 |
| --------- | --------------- | ---------------: | -----------------------: |
| Sentiment | no_accent       |           0.3399 |                   0.4140 |
| Sentiment | mixed_no_accent |           0.3405 |                   0.4110 |
| Topic     | no_accent       |           0.1866 |                   0.2957 |
| Topic     | mixed_no_accent |           0.1870 |                   0.3003 |

Kết luận: PhoBERT không tự động bền hơn baseline trong mọi điều kiện. Với văn bản thiếu dấu, char-level SVM có lợi thế rõ ràng.

### 7.3. Robust inference

Stage 8 triển khai chiến lược:

```text
Nếu input bình thường:
    dùng PhoBERT

Nếu input có khả năng thiếu dấu:
    dùng TF-IDF char SVM
```

Kết quả Robust Router so với PhoBERT-only:

| Task      | Variant         | PhoBERT-only Macro-F1 | Robust Router Macro-F1 |    Gain |
| --------- | --------------- | --------------------: | ---------------------: | ------: |
| Sentiment | no_accent       |                0.3399 |                 0.4148 | +0.0749 |
| Sentiment | mixed_no_accent |                0.3405 |                 0.4107 | +0.0702 |
| Topic     | no_accent       |                0.1866 |                 0.3000 | +0.1134 |
| Topic     | mixed_no_accent |                0.1870 |                 0.3041 | +0.1171 |

Detector thiếu dấu có route rate trên test:

| Variant         | Route sang char SVM |
| --------------- | ------------------: |
| clean           |               0.22% |
| typo_light      |               0.22% |
| typo_medium     |               0.22% |
| teencode_light  |               0.22% |
| mixed_light     |               0.28% |
| no_accent       |              95.39% |
| mixed_no_accent |              95.17% |

Kết luận: Robust Router gần như giữ nguyên hiệu năng trên clean/noise nhẹ, đồng thời cải thiện rõ rệt trên các biến thể thiếu dấu.

---

## 8. Demo app

Demo được xây dựng bằng Streamlit.

App cho phép người dùng nhập phản hồi sinh viên, sau đó dự đoán:

- Cảm xúc phản hồi.
- Chủ đề phản hồi.
- Detector có phát hiện thiếu dấu không.
- Model nào được dùng: PhoBERT hoặc TF-IDF char SVM.

### 8.1. Chuẩn bị model

PhoBERT cần tồn tại tại:

```text
outputs/models/phobert/sentiment/best/
outputs/models/phobert/topic/best/
```

TF-IDF char SVM cần tồn tại tại:

```text
outputs/models/baseline/sentiment_tfidf_char_svm.joblib
outputs/models/baseline/topic_tfidf_char_svm.joblib
```

Nếu thiếu SVM, chạy:

```powershell
python scripts/export_char_svm_models.py
```

### 8.2. Chạy demo

```powershell
streamlit run demo/app.py
```

Hoặc trên Windows PowerShell:

```powershell
.\demo
un_demo.ps1
```

### 8.3. Ví dụ nhập thử

Phản hồi có dấu:

```text
giảng viên rất nhiệt tình, bài giảng dễ hiểu và có nhiều ví dụ thực tế
```

Phản hồi thiếu dấu:

```text
giang vien rat nhiet tinh nhung slide hoi kho hieu
```

Phản hồi về cơ sở vật chất:

```text
phòng học nóng, máy chiếu mờ và âm thanh rất khó nghe
```

---

## 9. Ghi chú về model artifacts

Không nên commit các model PhoBERT nếu dung lượng quá lớn.

Có thể thêm vào `.gitignore`:

```gitignore
outputs/models/phobert/
*.zip
```

TF-IDF char SVM thường nhẹ hơn, có thể commit nếu cần demo nhanh.

---

## 10. Hạn chế

Dự án còn một số hạn chế:

1. Bộ dữ liệu thuộc miền phản hồi sinh viên, chưa kiểm chứng trên miền văn bản tiếng Việt khác.
2. Noisy data được sinh bằng rule, chưa bao phủ toàn bộ lỗi ngôn ngữ tự nhiên của người dùng thật.
3. Robust Router cải thiện hệ thống suy luận, không cải thiện bản thân PhoBERT.
4. Confidence của LinearSVC trong demo chỉ là pseudo-score từ decision margin, không phải xác suất hiệu chuẩn.
5. Chưa thực hiện kiểm định thống kê như bootstrap confidence interval hoặc McNemar test.

---

## 11. Hướng phát triển

Một số hướng phát triển tiếp theo:

1. Fine-tune PhoBERT với noisy data augmentation.
2. Thêm mô hình khôi phục dấu tiếng Việt trước khi phân loại.
3. Hiệu chuẩn confidence score cho LinearSVC.
4. Triển khai API bằng FastAPI.
5. Đóng gói demo bằng Docker.
6. Kiểm thử thêm trên dữ liệu phản hồi sinh viên thực tế ngoài tập UIT-VSFC.

---

## 12. Kết luận

Dự án cho thấy PhoBERT đạt hiệu năng cao trên văn bản tiếng Việt chuẩn, nhưng không bền vững trong mọi điều kiện phi chuẩn. Đặc biệt, văn bản thiếu dấu làm PhoBERT suy giảm mạnh. Trong điều kiện này, TF-IDF char SVM lại cho kết quả ổn định hơn.

Từ phát hiện đó, dự án đề xuất chiến lược Robust Inference: sử dụng PhoBERT làm mô hình mặc định và chuyển sang TF-IDF char SVM khi phát hiện văn bản thiếu dấu. Kết quả thực nghiệm cho thấy chiến lược này cải thiện rõ rệt trên dữ liệu thiếu dấu mà gần như không làm giảm hiệu năng trên dữ liệu sạch và nhiễu nhẹ.
