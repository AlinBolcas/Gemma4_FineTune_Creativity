# Fine-Tune Report: all_domains_augmented_20260417_155341_e4b_v2_strong

**Source:** training_runs
**Model dir:** `/Users/arvolve/GDRIVE/05_CODE/Gemma4_FineTune_Creativity/Gemma4_FineTune_Creativity/data/output/models/all_domains_augmented_20260417_155341_e4b_v2_strong`
**Run dir:** `/Users/arvolve/GDRIVE/05_CODE/Gemma4_FineTune_Creativity/Gemma4_FineTune_Creativity/data/output/training_runs/all_domains_augmented_20260417_155341_e4b_v2_strong`

## Summary


## Config

- `model.alias`: e4b
- `model.hf_model_id`: google/gemma-4-E4B-it
- `training.backend`: unsloth
- `training.output_dir`: data/output/models/all_domains_augmented_20260417_155341_e4b_v2_strong
- `training.max_seq_length`: 1024
- `data.train_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/train/all_domains_augmented_20260417_155341_train.jsonl
- `data.eval_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/eval/all_domains_augmented_20260417_155341_eval.jsonl
- `data.test_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/test/all_domains_augmented_20260417_155341_test.jsonl
- `global_step`: 672
- `max_steps`: 672
- `num_train_epochs`: 8
- `best_metric`: None

## Train Metrics

- `train_runtime`: 5084.256
- `train_samples_per_second`: 0.526
- `train_steps_per_second`: 0.132
- `total_flos`: 4.661227462285056e+16
- `train_loss`: 0.8195443242279691
- `epoch`: 8.0

## Eval Metrics

- `eval_skipped`: True
- `error`: CUDA out of memory. Tried to allocate 1.51 GiB. GPU 0 has a total capacity of 14.56 GiB of which 831.81 MiB is free. Including non-PyTorch memory, this process has 13.75 GiB memory in use. Of the allocated memory 13.44 GiB is allocated by PyTorch, and 153.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## Last Logged Steps

| Step | Loss | Eval Loss | LR | Grad Norm |
|---|---:|---:|---:|---:|
| 664 | 0.027839094400405884 |  | 5.438066465256798e-06 | 0.663092315196991 |
| 665 | 0.011619411408901215 |  | 4.984894259818731e-06 | 0.5261902213096619 |
| 666 | 0.01229323260486126 |  | 4.531722054380664e-06 | 0.6221637725830078 |
| 667 | 0.013344500213861465 |  | 4.078549848942597e-06 | 0.4886311888694763 |
| 668 | 0.014296982437372208 |  | 3.6253776435045313e-06 | 0.7095783352851868 |
| 669 | 0.008884385228157043 |  | 3.172205438066465e-06 | 0.43530604243278503 |
| 670 | 0.016221947968006134 |  | 2.719033232628399e-06 | 0.7975277900695801 |
| 671 | 0.017952045425772667 |  | 2.265861027190332e-06 | 0.8160852193832397 |
| 672 | 0.008323514834046364 |  | 1.8126888217522657e-06 | 0.7745924592018127 |
| 672 |  |  |  |  |

## Plot Files

- `loss.png`
- `learning_rate.png`
- `grad_norm.png`
