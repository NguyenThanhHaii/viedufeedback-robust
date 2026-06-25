# ViEduFeedback Robust

Dự án này đánh giá độ bền của mô hình phân loại phản hồi sinh viên tiếng Việt. Hai tác vụ được xử lý song song:

- **Sentiment**: `negative`, `neutral`, `positive`
- **Topic**: `lecturer`, `training_program`, `facility`, `others`

Mô hình chính là PhoBERT. Baseline gồm majority classifier, TF-IDF word SVM và TF-IDF char SVM. Ngoài test set sạch, dự án tạo thêm các biến thể nhiễu như thiếu dấu, lỗi gõ, teencode và nhiễu kết hợp.

Điểm chính của dự án không chỉ là so sánh PhoBERT với baseline, mà còn kiểm tra mô hình có bền khi đầu vào tiếng Việt không chuẩn hay không.

## Dữ liệu

Dataset: `uitnlp/vietnamese_students_feedback`

| Split | Số mẫu |
|---|---:|
| Train | 11,426 |
| Validation | 1,583 |
| Test | 3,166 |

Schema sau chuẩn hóa:

| Cột | Ý nghĩa |
|---|---|
| `text` | Nội dung phản hồi |
| `sentiment_label`, `sentiment_name` | Nhãn cảm xúc |
| `topic_label`, `topic_name` | Nhãn chủ đề |

## Pipeline

| Stage | Nội dung |
|---|---|
| 1 | Kiểm tra dataset và chuẩn hóa schema |
| 2 | EDA |
| 3 | Train baseline |
| 4 | Tạo noisy test set |
| 5 | Đánh giá robustness của baseline |
| 6 | Fine-tune PhoBERT |
| 7 | So sánh cuối và phân tích lỗi |
| 8 | Robust inference: PhoBERT + char SVM fallback |
| Demo | Streamlit app nhập phản hồi và dự đoán nhãn |

## Kết quả chính

### Clean test

| task      | model   |   accuracy |   macro_f1 |
|:----------|:--------|-----------:|-----------:|
| sentiment | phobert |   0.931459 |   0.822815 |
| topic     | phobert |   0.896399 |   0.800073 |

PhoBERT đạt Macro-F1 cao nhất trên dữ liệu sạch ở cả hai tác vụ.

### Văn bản thiếu dấu

| task      | variant         | model          |   accuracy |   macro_f1 |   macro_f1_drop |
|:----------|:----------------|:---------------|-----------:|-----------:|----------------:|
| sentiment | mixed_no_accent | phobert        |   0.371762 |   0.340462 |        0.482353 |
| sentiment | mixed_no_accent | tfidf_char_svm |   0.469678 |   0.411    |        0.327784 |
| sentiment | no_accent       | phobert        |   0.375237 |   0.339899 |        0.482916 |
| sentiment | no_accent       | tfidf_char_svm |   0.474416 |   0.414024 |        0.32476  |
| topic     | mixed_no_accent | phobert        |   0.244157 |   0.187029 |        0.613044 |
| topic     | mixed_no_accent | tfidf_char_svm |   0.432091 |   0.300282 |        0.431925 |
| topic     | no_accent       | phobert        |   0.252369 |   0.1866   |        0.613473 |
| topic     | no_accent       | tfidf_char_svm |   0.436197 |   0.295702 |        0.436505 |

PhoBERT giảm mạnh trên `no_accent` và `mixed_no_accent`. Trong hai biến thể này, TF-IDF char SVM ổn định hơn do đặc trưng ký tự ít phụ thuộc vào token/subword có dấu.

### Robust inference

Stage 8 dùng rule phát hiện thiếu dấu để chọn mô hình:

```text
Nếu phản hồi không bị nghi thiếu dấu:
    dùng PhoBERT

Nếu phản hồi có khả năng thiếu dấu:
    dùng TF-IDF char SVM
```

Kết quả của Robust Router trên hai biến thể thiếu dấu:

| task      | variant         |   accuracy |   macro_f1 |   macro_f1_gain_vs_phobert |   accuracy_gain_vs_phobert |
|:----------|:----------------|-----------:|-----------:|---------------------------:|---------------------------:|
| sentiment | mixed_no_accent |   0.469994 |   0.410707 |                   0.070245 |                   0.098232 |
| sentiment | no_accent       |   0.475679 |   0.414838 |                   0.074939 |                   0.100442 |
| topic     | mixed_no_accent |   0.437776 |   0.304132 |                   0.117103 |                   0.193619 |
| topic     | no_accent       |   0.442198 |   0.300043 |                   0.113443 |                   0.189829 |

Detector gần như không route nhầm dữ liệu sạch, nhưng phát hiện được phần lớn mẫu thiếu dấu:

| variant         |   route_to_char_svm |   route_to_char_rate | true_no_accent_variant   |
|:----------------|--------------------:|---------------------:|:-------------------------|
| clean           |                   7 |             0.002211 | False                    |
| mixed_light     |                   9 |             0.002843 | False                    |
| mixed_no_accent |                3013 |             0.951674 | True                     |
| no_accent       |                3020 |             0.953885 | True                     |
| teencode_light  |                   7 |             0.002211 | False                    |
| typo_light      |                   7 |             0.002211 | False                    |
| typo_medium     |                   7 |             0.002211 | False                    |

## Cài đặt

```powershell
conda create -n viedufeedback python=3.10 -y
conda activate viedufeedback
pip install -r requirements.txt
```

## Chạy lại các bước chính

```powershell
python scripts/verify_dataset.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/generate_noisy_tests.py
python scripts/evaluate_baseline_robustness.py
```

PhoBERT được fine-tune trên Kaggle bằng:

```text
notebooks/06_phobert_kaggle_training_clean.ipynb
kaggle/stage6_train_phobert.py
```

Sau khi có output PhoBERT, chạy phần tổng hợp:

```powershell
python scripts/run_final_comparison.py
python scripts/generate_noisy_validation.py
python scripts/evaluate_robust_inference.py
```

## Chạy demo

Demo cần model đã export:

```text
outputs/models/phobert/sentiment/best/
outputs/models/phobert/topic/best/
outputs/models/baseline/sentiment_tfidf_char_svm.joblib
outputs/models/baseline/topic_tfidf_char_svm.joblib
```

Nếu chưa có SVM fallback:

```powershell
python scripts/export_char_svm_models.py
```

Chạy app:

```powershell
streamlit run demo/app.py
```

Ví dụ nhập thử:

```text
giảng viên rất nhiệt tình, bài giảng dễ hiểu và có nhiều ví dụ thực tế
```

```text
giang vien rat nhiet tinh nhung slide hoi kho hieu
```

## Cấu trúc thư mục

```text
configs/      cấu hình thực nghiệm
data/         raw, processed và noisy data
demo/         Streamlit app
kaggle/       script train PhoBERT trên Kaggle
notebooks/    notebook kiểm tra và phân tích kết quả
outputs/      bảng, hình, report và model artifacts
scripts/      entrypoint chạy từng stage
src/          logic chính của pipeline
```

## Ghi chú

- `outputs/models/phobert/` thường lớn, không nên commit trực tiếp nếu repo dùng để nộp code.
- `outputs/predictions/` có thể sinh lại từ pipeline nên được ignore mặc định.
- Robust Router cải thiện chiến lược suy luận, không thay đổi trọng số của PhoBERT.
- Confidence của LinearSVC trong demo là pseudo-score từ margin, không phải xác suất đã hiệu chuẩn.

## Kết luận

PhoBERT là mô hình tốt nhất trên dữ liệu sạch và nhiễu nhẹ. Tuy nhiên, khi phản hồi bị thiếu dấu, PhoBERT giảm mạnh hơn TF-IDF char SVM. Vì vậy dự án dùng chiến lược suy luận lai: PhoBERT làm mô hình mặc định, còn char SVM xử lý các câu bị nghi thiếu dấu. Cách này giữ gần nguyên hiệu năng trên dữ liệu sạch và tăng rõ Macro-F1 trên các biến thể thiếu dấu.
