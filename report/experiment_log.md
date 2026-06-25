# Experiment Log

| Stage | Script / Notebook | Ghi chú |
|---|---|---|
| 1 | `scripts/verify_dataset.py` | Kiểm tra split và chuẩn hóa schema |
| 2 | `scripts/run_eda.py` | Sinh bảng và biểu đồ EDA |
| 3 | `scripts/train_baselines.py` | Train baseline |
| 4 | `scripts/generate_noisy_tests.py` | Tạo noisy test |
| 5 | `scripts/evaluate_baseline_robustness.py` | Đánh giá baseline trên noisy test |
| 6 | `kaggle/stage6_train_phobert.py` | Fine-tune PhoBERT trên Kaggle |
| 7 | `scripts/run_final_comparison.py` | Tổng hợp kết quả và lỗi |
| 8 | `scripts/evaluate_robust_inference.py` | Đánh giá robust router |
