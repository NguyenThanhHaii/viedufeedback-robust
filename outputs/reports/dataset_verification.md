# Dataset Verification Report

Dataset: `uitnlp/vietnamese_students_feedback`

## 1. Official Schema

- Text column: `sentence`
- Sentiment column: `sentiment`
- Topic column: `topic`
- Standardized text column: `text`
- Standardized sentiment columns: `sentiment_label`, `sentiment_name`
- Standardized topic columns: `topic_label`, `topic_name`

## 2. Data Lineage

- Raw snapshot directory: `data/raw`
- Processed standardized directory: `data/processed`
- Raw files keep the original HuggingFace schema.
- Processed files use the project-wide standardized schema.

## 3. Split Summary

### Split: `train`

- Rows: 11426
- Columns: 3
- Duplicated rows: 0

Columns:
- `sentence`
- `sentiment`
- `topic`

Missing values:
- `sentence`: 0
- `sentiment`: 0
- `topic`: 0

### Split: `validation`

- Rows: 1583
- Columns: 3
- Duplicated rows: 0

Columns:
- `sentence`
- `sentiment`
- `topic`

Missing values:
- `sentence`: 0
- `sentiment`: 0
- `topic`: 0

### Split: `test`

- Rows: 3166
- Columns: 3
- Duplicated rows: 0

Columns:
- `sentence`
- `sentiment`
- `topic`

Missing values:
- `sentence`: 0
- `sentiment`: 0
- `topic`: 0

## 4. Label Mapping

### Sentiment

- 0: `negative`
- 1: `neutral`
- 2: `positive`

### Topic

- 0: `lecturer`
- 1: `training_program`
- 2: `facility`
- 3: `others`

## 5. Column Candidates

Các cột dưới đây là kết quả kiểm tra tự động theo tên cột.

- text_candidates: ['sentence']
- sentiment_candidates: ['sentiment']
- topic_candidates: ['topic']

## 6. Quality Summary

| split | num_rows | empty_text | missing_text | duplicated_text | missing_sentiment_label | missing_sentiment_name | missing_topic_label | missing_topic_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 11426 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| validation | 1583 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 3166 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 7. Ghi chú kiểm tra

- Dataset có đủ 3 split: train, validation, test.
- Dataset có đủ nhãn cho cả sentiment classification và topic classification.
- Bản raw snapshot được lưu tại `data/raw/`.
- Các file chuẩn hóa được lưu tại `data/processed/*_standardized.csv`.
- Chưa train model ở giai đoạn này.
