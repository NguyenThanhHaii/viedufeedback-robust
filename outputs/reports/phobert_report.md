# PhoBERT Report

PhoBERT được fine-tune riêng cho hai tác vụ: sentiment và topic.

## Clean test

| task      |   accuracy |   macro_f1 |   weighted_f1 |   best_epoch |
|:----------|-----------:|-----------:|--------------:|-------------:|
| sentiment |   0.931459 |   0.822815 |      0.929772 |            4 |
| topic     |   0.896399 |   0.800073 |      0.894241 |            4 |

## Robustness snapshot

| task      | variant         |   accuracy |   macro_f1 |   weighted_f1 |
|:----------|:----------------|-----------:|-----------:|--------------:|
| sentiment | clean           |   0.931459 |   0.822815 |      0.929772 |
| sentiment | typo_light      |   0.929248 |   0.820352 |      0.927437 |
| sentiment | mixed_light     |   0.921668 |   0.813839 |      0.920467 |
| sentiment | no_accent       |   0.375237 |   0.339899 |      0.429524 |
| sentiment | mixed_no_accent |   0.371762 |   0.340462 |      0.430592 |
| topic     | clean           |   0.896399 |   0.800073 |      0.894241 |
| topic     | typo_light      |   0.893557 |   0.795476 |      0.891452 |
| topic     | mixed_light     |   0.881238 |   0.777107 |      0.879876 |
| topic     | no_accent       |   0.252369 |   0.1866   |      0.320529 |
| topic     | mixed_no_accent |   0.244157 |   0.187029 |      0.31136  |

PhoBERT giữ kết quả tốt với typo/teencode nhẹ, nhưng giảm mạnh khi văn bản bị bỏ dấu.
