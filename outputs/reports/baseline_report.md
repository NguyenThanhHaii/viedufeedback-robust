# Baseline Report

## 1. Mục tiêu

Giai đoạn này huấn luyện và đánh giá các baseline truyền thống cho hai tác vụ sentiment classification và topic classification.

Các mô hình gồm:

- Majority Class
- TF-IDF word-level + Linear SVM
- TF-IDF char-level + Linear SVM

## 2. Kết quả tổng hợp

| task      | model          | split      |   accuracy |   macro_f1 |   weighted_f1 |
|:----------|:---------------|:-----------|-----------:|-----------:|--------------:|
| sentiment | tfidf_char_svm | test       |   0.875237 |   0.738784 |      0.876528 |
| sentiment | tfidf_word_svm | test       |   0.891977 |   0.728904 |      0.887    |
| sentiment | majority       | test       |   0.502211 |   0.222876 |      0.335793 |
| sentiment | tfidf_word_svm | validation |   0.910929 |   0.768048 |      0.908651 |
| sentiment | tfidf_char_svm | validation |   0.899558 |   0.766947 |      0.902882 |
| sentiment | majority       | validation |   0.508528 |   0.224735 |      0.342852 |
| topic     | tfidf_word_svm | test       |   0.858497 |   0.750922 |      0.859847 |
| topic     | tfidf_char_svm | test       |   0.833544 |   0.732207 |      0.840575 |
| topic     | majority       | test       |   0.72331  |   0.209861 |      0.607178 |
| topic     | tfidf_word_svm | validation |   0.864814 |   0.761481 |      0.865695 |
| topic     | tfidf_char_svm | validation |   0.834491 |   0.728565 |      0.840343 |
| topic     | majority       | validation |   0.7271   |   0.210497 |      0.612211 |

## 3. Ghi chú

- Macro-F1 là metric chính vì dữ liệu mất cân bằng.
- Accuracy chỉ dùng làm metric phụ.
- Kết quả baseline là mốc đối chứng trước khi fine-tune PhoBERT.
- Chưa kết luận về PhoBERT trong giai đoạn này.
