# Robust Inference Report

## Thiết kế

Stage 8 dùng PhoBERT làm mô hình mặc định. Nếu detector phát hiện câu có khả năng thiếu dấu, hệ thống chuyển sang TF-IDF char SVM.

Detector dựa trên tỷ lệ ký tự tiếng Việt có dấu trong câu:

```text
accented_ratio = marked_vietnamese_chars / alphabetic_chars
```

Ngưỡng được chọn trên noisy validation, không chọn trên test.

## Ngưỡng detector

|   threshold |   precision |   recall |       f1 |   false_positive_rate |   false_negative_rate |
|------------:|------------:|---------:|---------:|----------------------:|----------------------:|
|       0     |    1        | 0.923563 | 0.960263 |              0        |              0.076437 |
|       0.005 |    1        | 0.923563 | 0.960263 |              0        |              0.076437 |
|       0.01  |    1        | 0.926406 | 0.961797 |              0        |              0.073594 |
|       0.02  |    1        | 0.936513 | 0.967216 |              0        |              0.063487 |
|       0.05  |    0.998338 | 0.948515 | 0.972789 |              0.000632 |              0.051485 |
|       0.1   |    0.996701 | 0.954201 | 0.974988 |              0.001263 |              0.045799 |

Ngưỡng được chọn: `0.1`.

## Route rate trên test

| variant         |   route_to_char_svm |   route_to_char_rate | true_no_accent_variant   |
|:----------------|--------------------:|---------------------:|:-------------------------|
| clean           |                   7 |             0.002211 | False                    |
| mixed_light     |                   9 |             0.002843 | False                    |
| mixed_no_accent |                3013 |             0.951674 | True                     |
| no_accent       |                3020 |             0.953885 | True                     |
| teencode_light  |                   7 |             0.002211 | False                    |
| typo_light      |                   7 |             0.002211 | False                    |
| typo_medium     |                   7 |             0.002211 | False                    |

Detector route rất ít mẫu clean/noise nhẹ sang char SVM, nhưng route hơn 95% mẫu thiếu dấu.

## Kết quả Robust Router

| task      | variant         |   accuracy |   macro_f1 |   macro_f1_gain_vs_phobert |   accuracy_gain_vs_phobert |
|:----------|:----------------|-----------:|-----------:|---------------------------:|---------------------------:|
| sentiment | mixed_no_accent |   0.469994 |   0.410707 |                   0.070245 |                   0.098232 |
| sentiment | no_accent       |   0.475679 |   0.414838 |                   0.074939 |                   0.100442 |
| topic     | mixed_no_accent |   0.437776 |   0.304132 |                   0.117103 |                   0.193619 |
| topic     | no_accent       |   0.442198 |   0.300043 |                   0.113443 |                   0.189829 |

Robust Router gần như giữ nguyên kết quả trên clean/noise nhẹ và cải thiện rõ trên `no_accent`, `mixed_no_accent`.

## Kết luận

Chiến lược suy luận lai phù hợp với kết quả thực nghiệm: PhoBERT xử lý tốt văn bản chuẩn, còn TF-IDF char SVM ổn định hơn khi văn bản thiếu dấu. Cách kết hợp này cải thiện hệ thống mà không cần fine-tune lại PhoBERT.
