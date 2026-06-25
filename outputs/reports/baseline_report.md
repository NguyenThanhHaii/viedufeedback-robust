# Baseline Report

Baseline gồm:

- Majority classifier
- TF-IDF word SVM
- TF-IDF char SVM

## Test result

| task      | model          |   accuracy |   macro_f1 |   weighted_f1 |
|:----------|:---------------|-----------:|-----------:|--------------:|
| sentiment | tfidf_char_svm |   0.875237 |   0.738784 |      0.876528 |
| sentiment | tfidf_word_svm |   0.891977 |   0.728904 |      0.887    |
| sentiment | majority       |   0.502211 |   0.222876 |      0.335793 |
| topic     | tfidf_word_svm |   0.858497 |   0.750922 |      0.859847 |
| topic     | tfidf_char_svm |   0.833544 |   0.732207 |      0.840575 |
| topic     | majority       |   0.72331  |   0.209861 |      0.607178 |

Với sentiment, char SVM có Macro-F1 cao nhất trong nhóm baseline. Với topic, word SVM tốt nhất trên clean test.
