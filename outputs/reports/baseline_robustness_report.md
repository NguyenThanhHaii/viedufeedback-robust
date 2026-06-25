# Baseline Robustness Evaluation Report

## 1. Mục tiêu

Giai đoạn này đánh giá các baseline đã train trên clean training set trên các phiên bản clean/noisy test set.

Không train lại model trong giai đoạn này.

## 2. Clean baseline reference

```text
     task          model variant noise_type noise_level  num_samples  accuracy  macro_f1  weighted_f1
sentiment tfidf_char_svm   clean      clean        none         3166  0.875237  0.738784     0.876528
    topic tfidf_word_svm   clean      clean        none         3166  0.858497  0.750922     0.859847
```

## 3. Full robustness results

```text
     task          model         variant        noise_type noise_level  num_samples  accuracy  macro_f1  weighted_f1
sentiment       majority           clean             clean        none         3166  0.502211  0.222876     0.335793
sentiment       majority     mixed_light             mixed       light         3166  0.502211  0.222876     0.335793
sentiment       majority mixed_no_accent             mixed      medium         3166  0.502211  0.222876     0.335793
sentiment       majority       no_accent remove_diacritics      medium         3166  0.502211  0.222876     0.335793
sentiment       majority  teencode_light          teencode       light         3166  0.502211  0.222876     0.335793
sentiment       majority      typo_light              typo       light         3166  0.502211  0.222876     0.335793
sentiment       majority     typo_medium              typo      medium         3166  0.502211  0.222876     0.335793
sentiment tfidf_char_svm           clean             clean        none         3166  0.875237  0.738784     0.876528
sentiment tfidf_char_svm     mixed_light             mixed       light         3166  0.864498  0.724830     0.867596
sentiment tfidf_char_svm mixed_no_accent             mixed      medium         3166  0.469678  0.411000     0.519813
sentiment tfidf_char_svm       no_accent remove_diacritics      medium         3166  0.474416  0.414024     0.522812
sentiment tfidf_char_svm  teencode_light          teencode       light         3166  0.869236  0.733458     0.871656
sentiment tfidf_char_svm      typo_light              typo       light         3166  0.873342  0.731545     0.874915
sentiment tfidf_char_svm     typo_medium              typo      medium         3166  0.870499  0.731212     0.872558
sentiment tfidf_word_svm           clean             clean        none         3166  0.891977  0.728904     0.887000
sentiment tfidf_word_svm     mixed_light             mixed       light         3166  0.881870  0.717428     0.876828
sentiment tfidf_word_svm mixed_no_accent             mixed      medium         3166  0.401453  0.357145     0.442646
sentiment tfidf_word_svm       no_accent remove_diacritics      medium         3166  0.406822  0.359685     0.447424
sentiment tfidf_word_svm  teencode_light          teencode       light         3166  0.885660  0.721304     0.880904
sentiment tfidf_word_svm      typo_light              typo       light         3166  0.887871  0.722617     0.882547
sentiment tfidf_word_svm     typo_medium              typo      medium         3166  0.881870  0.714089     0.877121
    topic       majority           clean             clean        none         3166  0.723310  0.209861     0.607178
    topic       majority     mixed_light             mixed       light         3166  0.723310  0.209861     0.607178
    topic       majority mixed_no_accent             mixed      medium         3166  0.723310  0.209861     0.607178
    topic       majority       no_accent remove_diacritics      medium         3166  0.723310  0.209861     0.607178
    topic       majority  teencode_light          teencode       light         3166  0.723310  0.209861     0.607178
    topic       majority      typo_light              typo       light         3166  0.723310  0.209861     0.607178
    topic       majority     typo_medium              typo      medium         3166  0.723310  0.209861     0.607178
    topic tfidf_char_svm           clean             clean        none         3166  0.833544  0.732207     0.840575
    topic tfidf_char_svm     mixed_light             mixed       light         3166  0.821226  0.718579     0.830109
    topic tfidf_char_svm mixed_no_accent             mixed      medium         3166  0.432091  0.300282     0.496427
    topic tfidf_char_svm       no_accent remove_diacritics      medium         3166  0.436197  0.295702     0.499572
    topic tfidf_char_svm  teencode_light          teencode       light         3166  0.822805  0.719644     0.831580
    topic tfidf_char_svm      typo_light              typo       light         3166  0.826911  0.723079     0.834723
    topic tfidf_char_svm     typo_medium              typo      medium         3166  0.825963  0.721417     0.834351
    topic tfidf_word_svm           clean             clean        none         3166  0.858497  0.750922     0.859847
    topic tfidf_word_svm     mixed_light             mixed       light         3166  0.846178  0.736205     0.848679
    topic tfidf_word_svm mixed_no_accent             mixed      medium         3166  0.327858  0.261696     0.404850
    topic tfidf_word_svm       no_accent remove_diacritics      medium         3166  0.329754  0.261359     0.406455
    topic tfidf_word_svm  teencode_light          teencode       light         3166  0.855022  0.749003     0.856682
    topic tfidf_word_svm      typo_light              typo       light         3166  0.855338  0.749665     0.857454
    topic tfidf_word_svm     typo_medium              typo      medium         3166  0.848705  0.739693     0.851713
```

## 4. Robustness drop from clean

```text
     task          model         variant        noise_type noise_level  clean_accuracy  variant_accuracy  accuracy_drop  accuracy_relative_drop_percent  clean_macro_f1  variant_macro_f1  macro_f1_drop  macro_f1_relative_drop_percent  clean_weighted_f1  variant_weighted_f1  weighted_f1_drop  weighted_f1_relative_drop_percent
sentiment       majority           clean             clean        none        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment       majority     mixed_light             mixed       light        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment       majority mixed_no_accent             mixed      medium        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment       majority       no_accent remove_diacritics      medium        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment       majority  teencode_light          teencode       light        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment       majority      typo_light              typo       light        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment       majority     typo_medium              typo      medium        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment tfidf_char_svm           clean             clean        none        0.875237          0.875237       0.000000                          0.0000        0.738784          0.738784       0.000000                          0.0000           0.876528             0.876528          0.000000                             0.0000
sentiment tfidf_char_svm     mixed_light             mixed       light        0.875237          0.864498       0.010739                          1.2270        0.738784          0.724830       0.013954                          1.8888           0.876528             0.867596          0.008931                             1.0189
sentiment tfidf_char_svm mixed_no_accent             mixed      medium        0.875237          0.469678       0.405559                         46.3371        0.738784          0.411000       0.327784                         44.3680           0.876528             0.519813          0.356714                            40.6963
sentiment tfidf_char_svm       no_accent remove_diacritics      medium        0.875237          0.474416       0.400821                         45.7957        0.738784          0.414024       0.324760                         43.9587           0.876528             0.522812          0.353716                            40.3542
sentiment tfidf_char_svm  teencode_light          teencode       light        0.875237          0.869236       0.006001                          0.6857        0.738784          0.733458       0.005327                          0.7210           0.876528             0.871656          0.004872                             0.5558
sentiment tfidf_char_svm      typo_light              typo       light        0.875237          0.873342       0.001895                          0.2165        0.738784          0.731545       0.007240                          0.9799           0.876528             0.874915          0.001613                             0.1840
sentiment tfidf_char_svm     typo_medium              typo      medium        0.875237          0.870499       0.004738                          0.5413        0.738784          0.731212       0.007572                          1.0250           0.876528             0.872558          0.003969                             0.4529
sentiment tfidf_word_svm           clean             clean        none        0.891977          0.891977       0.000000                          0.0000        0.728904          0.728904       0.000000                          0.0000           0.887000             0.887000          0.000000                             0.0000
sentiment tfidf_word_svm     mixed_light             mixed       light        0.891977          0.881870       0.010107                          1.1331        0.728904          0.717428       0.011476                          1.5744           0.887000             0.876828          0.010172                             1.1468
sentiment tfidf_word_svm mixed_no_accent             mixed      medium        0.891977          0.401453       0.490524                         54.9929        0.728904          0.357145       0.371759                         51.0025           0.887000             0.442646          0.444353                            50.0962
sentiment tfidf_word_svm       no_accent remove_diacritics      medium        0.891977          0.406822       0.485155                         54.3909        0.728904          0.359685       0.369219                         50.6540           0.887000             0.447424          0.439575                            49.5575
sentiment tfidf_word_svm  teencode_light          teencode       light        0.891977          0.885660       0.006317                          0.7082        0.728904          0.721304       0.007600                          1.0426           0.887000             0.880904          0.006096                             0.6873
sentiment tfidf_word_svm      typo_light              typo       light        0.891977          0.887871       0.004106                          0.4603        0.728904          0.722617       0.006287                          0.8626           0.887000             0.882547          0.004453                             0.5020
sentiment tfidf_word_svm     typo_medium              typo      medium        0.891977          0.881870       0.010107                          1.1331        0.728904          0.714089       0.014815                          2.0325           0.887000             0.877121          0.009879                             1.1138
    topic       majority           clean             clean        none        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic       majority     mixed_light             mixed       light        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic       majority mixed_no_accent             mixed      medium        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic       majority       no_accent remove_diacritics      medium        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic       majority  teencode_light          teencode       light        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic       majority      typo_light              typo       light        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic       majority     typo_medium              typo      medium        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic tfidf_char_svm           clean             clean        none        0.833544          0.833544       0.000000                          0.0000        0.732207          0.732207       0.000000                          0.0000           0.840575             0.840575          0.000000                             0.0000
    topic tfidf_char_svm     mixed_light             mixed       light        0.833544          0.821226       0.012318                          1.4778        0.732207          0.718579       0.013628                          1.8613           0.840575             0.830109          0.010466                             1.2451
    topic tfidf_char_svm mixed_no_accent             mixed      medium        0.833544          0.432091       0.401453                         48.1622        0.732207          0.300282       0.431925                         58.9894           0.840575             0.496427          0.344148                            40.9420
    topic tfidf_char_svm       no_accent remove_diacritics      medium        0.833544          0.436197       0.397347                         47.6696        0.732207          0.295702       0.436505                         59.6150           0.840575             0.499572          0.341003                            40.5679
    topic tfidf_char_svm  teencode_light          teencode       light        0.833544          0.822805       0.010739                          1.2884        0.732207          0.719644       0.012563                          1.7158           0.840575             0.831580          0.008995                             1.0701
    topic tfidf_char_svm      typo_light              typo       light        0.833544          0.826911       0.006633                          0.7958        0.732207          0.723079       0.009128                          1.2466           0.840575             0.834723          0.005852                             0.6961
    topic tfidf_char_svm     typo_medium              typo      medium        0.833544          0.825963       0.007581                          0.9094        0.732207          0.721417       0.010790                          1.4736           0.840575             0.834351          0.006224                             0.7404
    topic tfidf_word_svm           clean             clean        none        0.858497          0.858497       0.000000                          0.0000        0.750922          0.750922       0.000000                          0.0000           0.859847             0.859847          0.000000                             0.0000
    topic tfidf_word_svm     mixed_light             mixed       light        0.858497          0.846178       0.012318                          1.4349        0.750922          0.736205       0.014717                          1.9599           0.859847             0.848679          0.011169                             1.2989
    topic tfidf_word_svm mixed_no_accent             mixed      medium        0.858497          0.327858       0.530638                         61.8102        0.750922          0.261696       0.489226                         65.1500           0.859847             0.404850          0.454997                            52.9160
    topic tfidf_word_svm       no_accent remove_diacritics      medium        0.858497          0.329754       0.528743                         61.5894        0.750922          0.261359       0.489563                         65.1949           0.859847             0.406455          0.453392                            52.7294
    topic tfidf_word_svm  teencode_light          teencode       light        0.858497          0.855022       0.003474                          0.4047        0.750922          0.749003       0.001919                          0.2555           0.859847             0.856682          0.003165                             0.3681
    topic tfidf_word_svm      typo_light              typo       light        0.858497          0.855338       0.003159                          0.3679        0.750922          0.749665       0.001258                          0.1675           0.859847             0.857454          0.002393                             0.2783
    topic tfidf_word_svm     typo_medium              typo      medium        0.858497          0.848705       0.009792                          1.1405        0.750922          0.739693       0.011230                          1.4955           0.859847             0.851713          0.008134                             0.9459
```

## 5. Worst drop per model

```text
     task          model         variant        noise_type noise_level  clean_accuracy  variant_accuracy  accuracy_drop  accuracy_relative_drop_percent  clean_macro_f1  variant_macro_f1  macro_f1_drop  macro_f1_relative_drop_percent  clean_weighted_f1  variant_weighted_f1  weighted_f1_drop  weighted_f1_relative_drop_percent
sentiment       majority     mixed_light             mixed       light        0.502211          0.502211       0.000000                          0.0000        0.222876          0.222876       0.000000                          0.0000           0.335793             0.335793          0.000000                             0.0000
sentiment tfidf_char_svm mixed_no_accent             mixed      medium        0.875237          0.469678       0.405559                         46.3371        0.738784          0.411000       0.327784                         44.3680           0.876528             0.519813          0.356714                            40.6963
sentiment tfidf_word_svm mixed_no_accent             mixed      medium        0.891977          0.401453       0.490524                         54.9929        0.728904          0.357145       0.371759                         51.0025           0.887000             0.442646          0.444353                            50.0962
    topic       majority     mixed_light             mixed       light        0.723310          0.723310       0.000000                          0.0000        0.209861          0.209861       0.000000                          0.0000           0.607178             0.607178          0.000000                             0.0000
    topic tfidf_char_svm       no_accent remove_diacritics      medium        0.833544          0.436197       0.397347                         47.6696        0.732207          0.295702       0.436505                         59.6150           0.840575             0.499572          0.341003                            40.5679
    topic tfidf_word_svm       no_accent remove_diacritics      medium        0.858497          0.329754       0.528743                         61.5894        0.750922          0.261359       0.489563                         65.1949           0.859847             0.406455          0.453392                            52.7294
```

## 6. Notes

- Macro-F1 là metric chính vì dữ liệu mất cân bằng.
- Robustness drop được tính so với clean test cùng model và cùng task.
- Nếu noisy variant có Macro-F1 cao hơn clean, drop có thể âm.
- Kết quả này là mốc đối chứng trước khi đánh giá PhoBERT.
