# Baseline Robustness Report

Baseline được đánh giá trên clean test và các biến thể noisy test. Bảng dưới tập trung vào các biến thể thiếu dấu.

| task      | model          | variant         |   accuracy |   macro_f1 |   weighted_f1 |
|:----------|:---------------|:----------------|-----------:|-----------:|--------------:|
| sentiment | tfidf_char_svm | clean           |   0.875237 |   0.738784 |      0.876528 |
| sentiment | tfidf_word_svm | clean           |   0.891977 |   0.728904 |      0.887    |
| sentiment | tfidf_char_svm | mixed_no_accent |   0.469678 |   0.411    |      0.519813 |
| sentiment | tfidf_word_svm | mixed_no_accent |   0.401453 |   0.357145 |      0.442646 |
| sentiment | tfidf_char_svm | no_accent       |   0.474416 |   0.414024 |      0.522812 |
| sentiment | tfidf_word_svm | no_accent       |   0.406822 |   0.359685 |      0.447424 |
| topic     | tfidf_char_svm | clean           |   0.833544 |   0.732207 |      0.840575 |
| topic     | tfidf_word_svm | clean           |   0.858497 |   0.750922 |      0.859847 |
| topic     | tfidf_char_svm | mixed_no_accent |   0.432091 |   0.300282 |      0.496427 |
| topic     | tfidf_word_svm | mixed_no_accent |   0.327858 |   0.261696 |      0.40485  |
| topic     | tfidf_char_svm | no_accent       |   0.436197 |   0.295702 |      0.499572 |
| topic     | tfidf_word_svm | no_accent       |   0.329754 |   0.261359 |      0.406455 |

Nhận xét ngắn:

- Char SVM giảm ít hơn word SVM khi bỏ dấu.
- Với tiếng Việt thiếu dấu, đặc trưng ký tự có lợi thế hơn đặc trưng từ.
