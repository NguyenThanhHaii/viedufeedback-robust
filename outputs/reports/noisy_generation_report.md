# Noisy Test Generation Report

Test set được tạo thêm các biến thể nhiễu để đánh giá robustness.

Các biến thể chính:

- `clean`: dữ liệu gốc
- `no_accent`: bỏ dấu tiếng Việt
- `typo_light`, `typo_medium`: lỗi gõ
- `teencode_light`: viết tắt phổ biến
- `mixed_light`: kết hợp teencode và typo nhẹ
- `mixed_no_accent`: bỏ dấu kèm nhiễu nhẹ

## Summary

| variant         | noise_type        | noise_level   |   num_rows |   changed_percent |   avg_char_change_ratio |
|:----------------|:------------------|:--------------|-----------:|------------------:|------------------------:|
| clean           | clean             | none          |       3166 |            0      |                0        |
| mixed_light     | mixed             | light         |       3166 |           40.6822 |                0.015794 |
| mixed_no_accent | mixed             | medium        |       3166 |           99.8737 |                0.219922 |
| no_accent       | remove_diacritics | medium        |       3166 |           99.8737 |                0.211668 |
| teencode_light  | teencode          | light         |       3166 |           25.4264 |                0.012741 |
| typo_light      | typo              | light         |       3166 |           21.9204 |                0.002927 |
| typo_medium     | typo              | medium        |       3166 |           40.4611 |                0.006423 |

Train set không bị làm nhiễu. Noisy test chỉ dùng để đánh giá robustness.
