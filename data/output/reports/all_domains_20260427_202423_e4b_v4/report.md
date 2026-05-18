# Fine-Tune Report: all_domains_20260427_202423_e4b_v4

**Source:** training_runs
**Model dir:** `/Users/arvolve/GDRIVE/05_CODE/Gemma4_FineTune_Creativity/Gemma4_FineTune_Creativity/data/output/models/all_domains_20260427_202423_e4b_v4`
**Run dir:** `/Users/arvolve/GDRIVE/05_CODE/Gemma4_FineTune_Creativity/Gemma4_FineTune_Creativity/data/output/training_runs/all_domains_20260427_202423_e4b_v4`

## Summary


## Config

- `model.alias`: e4b
- `model.hf_model_id`: google/gemma-4-E4B-it
- `training.backend`: unsloth
- `training.output_dir`: data/output/models/all_domains_20260427_202423_e4b_v4
- `training.max_seq_length`: 1536
- `data.train_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/train/all_domains_20260427_202423_train.jsonl
- `data.eval_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/eval/all_domains_20260427_202423_eval.jsonl
- `data.test_path`: /kaggle/working/Gemma4_FineTune_Creativity/data/input/test/all_domains_20260427_202423_test.jsonl
- `global_step`: 2148
- `max_steps`: 2148
- `num_train_epochs`: 2
- `best_metric`: None

## Train Metrics

- `train_runtime`: 15824.6681
- `train_samples_per_second`: 0.543
- `train_steps_per_second`: 0.136
- `total_flos`: 1.4850432765851328e+17
- `train_loss`: 0.5593280279159102
- `epoch`: 2.0

## Eval Metrics

- `eval_skipped`: True
- `reason`: post_train_eval disabled in config

## Last Logged Steps

| Step | Loss | Eval Loss | LR | Grad Norm |
|---|---:|---:|---:|---:|
| 2140 | 0.44197627902030945 |  | 5.882627289371945e-09 | 0.651481568813324 |
| 2141 | 0.445992648601532 |  | 4.764945857360648e-09 | 0.42753681540489197 |
| 2142 | 0.4831135869026184 |  | 3.764908042774851e-09 | 0.5207428336143494 |
| 2143 | 0.454450786113739 |  | 2.8825161988044193e-09 | 0.544405460357666 |
| 2144 | 0.45754313468933105 |  | 2.117772401805107e-09 | 0.4745456278324127 |
| 2145 | 0.43939948081970215 |  | 1.470678451293006e-09 | 0.4661424458026886 |
| 2146 | 0.4736816883087158 |  | 9.412358699445456e-10 | 0.44997769594192505 |
| 2147 | 0.4715613126754761 |  | 5.294459035798394e-10 | 0.4470820128917694 |
| 2148 | 2.0091793537139893 |  | 2.3530952119044103e-10 | 4.212429523468018 |
| 2148 |  |  |  |  |

## Plot Files

- `loss.png`
- `learning_rate.png`
- `grad_norm.png`
