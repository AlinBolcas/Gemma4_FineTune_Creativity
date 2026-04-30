# Fine-Tune Report: all_domains_20260427_202423_e4b_v3_obvious

**Source:** training_runs
**Model dir:** `/Users/arvolve/GDRIVE/05_CODE/Gemma4_FineTune_Creativity/Gemma4_FineTune_Creativity/data/output/models/all_domains_20260427_202423_e4b_v3_obvious`
**Run dir:** `/Users/arvolve/GDRIVE/05_CODE/Gemma4_FineTune_Creativity/Gemma4_FineTune_Creativity/data/output/training_runs/all_domains_20260427_202423_e4b_v3_obvious`

## Summary


## Config

- `model.alias`: e4b
- `model.hf_model_id`: google/gemma-4-E4B-it
- `training.backend`: unsloth
- `training.output_dir`: data/output/models/all_domains_20260427_202423_e4b_v3_obvious
- `training.max_seq_length`: 1536
- `data.train_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/train/all_domains_20260427_202423_train.jsonl
- `data.eval_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/eval/all_domains_20260427_202423_eval.jsonl
- `data.test_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/test/all_domains_20260427_202423_test.jsonl
- `global_step`: 6444
- `max_steps`: 6444
- `num_train_epochs`: 6
- `best_metric`: None

## Train Metrics

- `train_runtime`: 40843.8248
- `train_samples_per_second`: 0.631
- `train_steps_per_second`: 0.158
- `total_flos`: 4.563096108839021e+17
- `train_loss`: 1.0634978382375895
- `epoch`: 6.0

## Eval Metrics

- `eval_skipped`: True
- `reason`: post_train_eval disabled in config

## Last Logged Steps

| Step | Loss | Eval Loss | LR | Grad Norm |
|---|---:|---:|---:|---:|
| 6436 | 0.09524694085121155 |  | 4.066312167657179e-07 | 2.0534474849700928 |
| 6437 | 0.06666336208581924 |  | 3.753518923991242e-07 | 1.507986307144165 |
| 6438 | 0.10150570422410965 |  | 3.4407256803253055e-07 | 1.8672086000442505 |
| 6439 | 0.06376361846923828 |  | 3.127932436659368e-07 | 1.5540369749069214 |
| 6440 | 0.07137979567050934 |  | 2.8151391929934315e-07 | 1.6501911878585815 |
| 6441 | 0.07888671010732651 |  | 2.502345949327495e-07 | 1.756588339805603 |
| 6442 | 0.07030801475048065 |  | 2.1895527056615578e-07 | 1.4584845304489136 |
| 6443 | 0.11432378739118576 |  | 1.876759461995621e-07 | 2.0186767578125 |
| 6444 | 0.09974384307861328 |  | 1.563966218329684e-07 | 3.657963275909424 |
| 6444 |  |  |  |  |

## Plot Files

- `loss.png`
- `learning_rate.png`
- `grad_norm.png`
