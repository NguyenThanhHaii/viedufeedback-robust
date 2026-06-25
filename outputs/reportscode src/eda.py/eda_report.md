# EDA Report

## 1. Basic Summary

| split      |   num_rows |   num_unique_text |   empty_text |   duplicated_text |   sentiment_num_classes |   topic_num_classes |
|:-----------|-----------:|------------------:|-------------:|------------------:|------------------------:|--------------------:|
| test       |       3166 |              3166 |            0 |                 0 |                       3 |                   4 |
| train      |      11426 |             11425 |            0 |                 1 |                       3 |                   4 |
| validation |       1583 |              1583 |            0 |                 0 |                       3 |                   4 |
| all        |      16175 |             16174 |            0 |                 1 |                       3 |                   4 |

## 2. Sentiment Label Distribution

| split      |   label_id | label_name   |   count |   percent |
|:-----------|-----------:|:-------------|--------:|----------:|
| test       |          0 | negative     |    1409 |   44.5041 |
| test       |          1 | neutral      |     167 |    5.2748 |
| test       |          2 | positive     |    1590 |   50.2211 |
| train      |          0 | negative     |    5325 |   46.6042 |
| train      |          1 | neutral      |     458 |    4.0084 |
| train      |          2 | positive     |    5643 |   49.3874 |
| validation |          0 | negative     |     705 |   44.5357 |
| validation |          1 | neutral      |      73 |    4.6115 |
| validation |          2 | positive     |     805 |   50.8528 |

## 3. Topic Label Distribution

| split      |   label_id | label_name       |   count |   percent |
|:-----------|-----------:|:-----------------|--------:|----------:|
| test       |          0 | lecturer         |    2290 |   72.331  |
| test       |          1 | training_program |     572 |   18.067  |
| test       |          2 | facility         |     145 |    4.5799 |
| test       |          3 | others           |     159 |    5.0221 |
| train      |          0 | lecturer         |    8166 |   71.4686 |
| train      |          1 | training_program |    2201 |   19.2631 |
| train      |          2 | facility         |     497 |    4.3497 |
| train      |          3 | others           |     562 |    4.9186 |
| validation |          0 | lecturer         |    1151 |   72.71   |
| validation |          1 | training_program |     267 |   16.8667 |
| validation |          2 | facility         |      70 |    4.422  |
| validation |          3 | others           |      95 |    6.0013 |

## 4. Length Summary by Split

| split      | metric                |    mean |     std |   min |   max |
|:-----------|:----------------------|--------:|--------:|------:|------:|
| test       | char_count            | 58.8364 | 44.1283 |     4 |   411 |
| test       | whitespace_word_count | 14.2208 | 10.2424 |     2 |    98 |
| test       | phobert_subword_count | 16.5474 | 10.5533 |     4 |   102 |
| train      | char_count            | 59.0849 | 43.0852 |     4 |   660 |
| train      | whitespace_word_count | 14.3088 | 10.0897 |     2 |   159 |
| train      | phobert_subword_count | 16.604  | 10.3808 |     4 |   163 |
| validation | char_count            | 56.362  | 42.7654 |     4 |   718 |
| validation | whitespace_word_count | 13.6715 |  9.9128 |     2 |   161 |
| validation | phobert_subword_count | 15.9097 | 10.1567 |     4 |   164 |

## 5. Whitespace Word Count Percentiles

| scope      | metric                |   percentile |   value |
|:-----------|:----------------------|-------------:|--------:|
| test       | whitespace_word_count |            0 |    2    |
| test       | whitespace_word_count |           25 |    8    |
| test       | whitespace_word_count |           50 |   11    |
| test       | whitespace_word_count |           75 |   17    |
| test       | whitespace_word_count |           90 |   26    |
| test       | whitespace_word_count |           95 |   34    |
| test       | whitespace_word_count |           99 |   52.35 |
| test       | whitespace_word_count |          100 |   98    |
| train      | whitespace_word_count |            0 |    2    |
| train      | whitespace_word_count |           25 |    8    |
| train      | whitespace_word_count |           50 |   11    |
| train      | whitespace_word_count |           75 |   17    |
| train      | whitespace_word_count |           90 |   26    |
| train      | whitespace_word_count |           95 |   33    |
| train      | whitespace_word_count |           99 |   52    |
| train      | whitespace_word_count |          100 |  159    |
| validation | whitespace_word_count |            0 |    2    |
| validation | whitespace_word_count |           25 |    8    |
| validation | whitespace_word_count |           50 |   11    |
| validation | whitespace_word_count |           75 |   17    |
| validation | whitespace_word_count |           90 |   25    |
| validation | whitespace_word_count |           95 |   31    |
| validation | whitespace_word_count |           99 |   46    |
| validation | whitespace_word_count |          100 |  161    |
| all        | whitespace_word_count |            0 |    2    |
| all        | whitespace_word_count |           25 |    8    |
| all        | whitespace_word_count |           50 |   11    |
| all        | whitespace_word_count |           75 |   17    |
| all        | whitespace_word_count |           90 |   26    |
| all        | whitespace_word_count |           95 |   33    |
| all        | whitespace_word_count |           99 |   52    |
| all        | whitespace_word_count |          100 |  161    |

## 6. PhoBERT Subword Count Percentiles

| scope      | metric                |   percentile |   value |
|:-----------|:----------------------|-------------:|--------:|
| test       | phobert_subword_count |            0 |    4    |
| test       | phobert_subword_count |           25 |   10    |
| test       | phobert_subword_count |           50 |   13    |
| test       | phobert_subword_count |           75 |   20    |
| test       | phobert_subword_count |           90 |   29    |
| test       | phobert_subword_count |           95 |   37    |
| test       | phobert_subword_count |           99 |   55    |
| test       | phobert_subword_count |          100 |  102    |
| train      | phobert_subword_count |            0 |    4    |
| train      | phobert_subword_count |           25 |   10    |
| train      | phobert_subword_count |           50 |   14    |
| train      | phobert_subword_count |           75 |   20    |
| train      | phobert_subword_count |           90 |   29    |
| train      | phobert_subword_count |           95 |   36    |
| train      | phobert_subword_count |           99 |   55    |
| train      | phobert_subword_count |          100 |  163    |
| validation | phobert_subword_count |            0 |    4    |
| validation | phobert_subword_count |           25 |   10    |
| validation | phobert_subword_count |           50 |   13    |
| validation | phobert_subword_count |           75 |   19    |
| validation | phobert_subword_count |           90 |   27    |
| validation | phobert_subword_count |           95 |   35    |
| validation | phobert_subword_count |           99 |   52.18 |
| validation | phobert_subword_count |          100 |  164    |
| all        | phobert_subword_count |            0 |    4    |
| all        | phobert_subword_count |           25 |   10    |
| all        | phobert_subword_count |           50 |   14    |
| all        | phobert_subword_count |           75 |   20    |
| all        | phobert_subword_count |           90 |   29    |
| all        | phobert_subword_count |           95 |   36    |
| all        | phobert_subword_count |           99 |   55    |
| all        | phobert_subword_count |          100 |  164    |

## 7. max_length Candidate Coverage

|   max_length_candidate |   covered_samples |   total_samples |   coverage_percent |
|-----------------------:|------------------:|----------------:|-------------------:|
|                     64 |             16085 |           16175 |            99.4436 |
|                     96 |             16164 |           16175 |            99.932  |
|                    128 |             16173 |           16175 |            99.9876 |
|                    160 |             16173 |           16175 |            99.9876 |
|                    192 |             16175 |           16175 |           100      |
|                    256 |             16175 |           16175 |           100      |

## 8. Notes

- Giai đoạn này chỉ phân tích dữ liệu, chưa train model.
- `whitespace_word_count` chỉ là thống kê theo khoảng trắng, không phải tách từ tiếng Việt chuẩn.
- `phobert_subword_count` dùng tokenizer PhoBERT để ước lượng độ dài đầu vào.
- Việc chọn `max_length` cho PhoBERT nên dựa trên bảng coverage, không chọn cảm tính.
