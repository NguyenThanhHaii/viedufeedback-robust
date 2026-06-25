# Báo cáo tổng hợp kết quả

## Clean test

| task      | model   |   accuracy |   macro_f1 |
|:----------|:--------|-----------:|-----------:|
| sentiment | phobert |   0.931459 |   0.822815 |
| topic     | phobert |   0.896399 |   0.800073 |

PhoBERT là mô hình tốt nhất trên clean test ở cả sentiment và topic.

## Robustness trên văn bản thiếu dấu

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

Kết quả chính:

- Với sentiment, TF-IDF char SVM cao hơn PhoBERT khoảng 0.07 Macro-F1 trên hai biến thể thiếu dấu.
- Với topic, TF-IDF char SVM cao hơn PhoBERT khoảng 0.11 Macro-F1.
- `no_accent` và `mixed_no_accent` là hai biến thể gây giảm mạnh nhất.

## Hướng lỗi chính của PhoBERT

### Sentiment

| variant         | y_true_name   | y_pred_name   |   count |   percent_within_wrong |
|:----------------|:--------------|:--------------|--------:|-----------------------:|
| mixed_light     | positive      | negative      |      57 |                22.9839 |
| mixed_light     | negative      | positive      |      53 |                21.371  |
| mixed_light     | neutral       | positive      |      52 |                20.9677 |
| mixed_no_accent | negative      | neutral       |     797 |                40.0704 |
| mixed_no_accent | positive      | neutral       |     620 |                31.1714 |
| mixed_no_accent | negative      | positive      |     291 |                14.6305 |
| no_accent       | negative      | neutral       |     792 |                40.0404 |
| no_accent       | positive      | neutral       |     596 |                30.1314 |
| no_accent       | negative      | positive      |     315 |                15.9252 |

Khi thiếu dấu, nhiều mẫu `negative` và `positive` bị kéo về `neutral`.

### Topic

| variant         | y_true_name      | y_pred_name      |   count |   percent_within_wrong |
|:----------------|:-----------------|:-----------------|--------:|-----------------------:|
| mixed_light     | training_program | lecturer         |     109 |                28.9894 |
| mixed_light     | lecturer         | training_program |     102 |                27.1277 |
| mixed_light     | others           | lecturer         |      43 |                11.4362 |
| mixed_no_accent | lecturer         | others           |    1355 |                56.6235 |
| mixed_no_accent | training_program | others           |     440 |                18.387  |
| mixed_no_accent | lecturer         | training_program |     320 |                13.3723 |
| no_accent       | lecturer         | others           |    1325 |                55.978  |
| no_accent       | training_program | others           |     443 |                18.7157 |
| no_accent       | lecturer         | training_program |     319 |                13.477  |

Ở task topic, nhiều phản hồi thuộc `lecturer` và `training_program` bị dự đoán thành `others`.

## Nhận xét

Kết quả Stage 7 cho thấy PhoBERT mạnh trên văn bản chuẩn, nhưng không bền trong mọi tình huống. Với văn bản thiếu dấu, baseline char-level lại có lợi thế hơn. Đây là cơ sở để xây dựng chiến lược suy luận ở Stage 8.
