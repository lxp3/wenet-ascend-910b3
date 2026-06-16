# AISHELL Conformer Ascend 910B3 Reproduction

This recipe records a WeNet Conformer reproduction on AISHELL-1 using Ascend 910B3 NPUs.  The goal is to keep the training setup, decoding results, and comparison with the original WeNet Conformer baseline in one place.

## Result

Evaluation set: AISHELL-1 `test`.

| decoding mode | Original Conformer CER | Ascend 910B3 CER | Delta |
| --- | ---: | ---: | ---: |
| attention decoder | 5.18 | 4.87 | -0.31 |
| ctc greedy search | 4.94 | 4.91 | -0.03 |
| ctc prefix beam search | 4.94 | 4.91 | -0.03 |
| attention rescoring | 4.61 | 4.61 | 0.00 |

The best result is `attention_rescoring` with 4.61 CER, matching the original Conformer result.  The other three decoding modes are slightly better than the original reported numbers in this run.

The reproduced WER/CER files are saved under:

```text
exp/conformer/attention/wer
exp/conformer/ctc_greedy_search/wer
exp/conformer/ctc_prefix_beam_search/wer
exp/conformer/attention_rescoring/wer
```

## Setup

Dataset:

- AISHELL-1, using the standard `train`, `dev`, and `test` splits.
- Local raw data path in this run: `/data/user/lxp/asr/datasets/aishell/raw`.
- Training data was prepared as shards with `num_utts_per_shard=1000`.
- Feature pipeline: 80-dim fbank, 16 kHz resampling, global CMVN, speed perturbation, SpecAugment, and dither `0.1`.

Hardware and distributed training:

- 1 node with 8 Ascend 910B3 NPUs.
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
- WeNet training device: `npu`.
- Distributed engine: `torch_fsdp`.
- Distributed backend: `hccl`.

System and software:

- OS: Kylin Linux Advanced Server V10 (Sword).
- Architecture: `aarch64`.
- CPU: HiSilicon Kunpeng-920, 4 sockets, 48 cores per socket, 192 CPU cores total.
- CANN toolkit: `8.2.RC1`.
- Driver / `npu-smi` version: `25.2.0`.
- Ascend package version: `25.2.0`.
- Accelerator platform: Atlas A2 / Ascend 910B series.

CPU topology:

| item | value |
| --- | --- |
| CPU op-mode | 64-bit |
| Byte order | Little Endian |
| CPU cores | 192 |
| Threads per core | 1 |
| Cores per socket | 48 |
| Sockets | 4 |
| NUMA nodes | 4 |
| CPU max MHz | 2600 |
| CPU min MHz | 200 |
| L1d cache | 12 MiB |
| L1i cache | 12 MiB |
| L2 cache | 96 MiB |
| L3 cache | 192 MiB |

NPU memory snapshot during training:

```text
(wenet) [root@pm-8ce2 s0]# npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 25.2.0                   Version: 25.2.0                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 116.2       36                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          44895/ 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 133.0       34                0    / 0             |
| 0                         | 0000:C2:00.0  | 11          0    / 0          42175/ 65536         |
+===========================+===============+====================================================+
| 2     910B3               | OK            | 103.4       36                0    / 0             |
| 0                         | 0000:81:00.0  | 3           0    / 0          49947/ 65536         |
+===========================+===============+====================================================+
| 3     910B3               | OK            | 95.9        36                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          57296/ 65536         |
+===========================+===============+====================================================+
| 4     910B3               | OK            | 125.2       40                0    / 0             |
| 0                         | 0000:01:00.0  | 10          0    / 0          44924/ 65536         |
+===========================+===============+====================================================+
| 5     910B3               | OK            | 106.1       40                0    / 0             |
| 0                         | 0000:02:00.0  | 1           0    / 0          35948/ 65536         |
+===========================+===============+====================================================+
| 6     910B3               | OK            | 130.0       41                0    / 0             |
| 0                         | 0000:41:00.0  | 8           0    / 0          48023/ 65536         |
+===========================+===============+====================================================+
| 7     910B3               | OK            | 114.1       40                0    / 0             |
| 0                         | 0000:42:00.0  | 1           0    / 0          34646/ 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 3715239       | python3                  | 41557                   |
+===========================+===============+====================================================+
| 1       0                 | 3715240       | python3                  | 38843                   |
+===========================+===============+====================================================+
| 2       0                 | 3715241       | python3                  | 46615                   |
+===========================+===============+====================================================+
| 3       0                 | 3715242       | python3                  | 53965                   |
+===========================+===============+====================================================+
| 4       0                 | 3715243       | python3                  | 41594                   |
+===========================+===============+====================================================+
| 5       0                 | 3715244       | python3                  | 32617                   |
+===========================+===============+====================================================+
| 6       0                 | 3715245       | python3                  | 44693                   |
+===========================+===============+====================================================+
| 7       0                 | 3715246       | python3                  | 31315                   |
+===========================+===============+====================================================+
```

Model:

- Config: `conf/train_conformer.yaml`.
- Encoder: 12-layer Conformer, attention dim 256, 4 heads, FFN dim 2048, CNN kernel 15, relative positional self-attention.
- Decoder: 6-layer Transformer decoder, attention dim 256, 4 heads, FFN dim 2048.
- Hybrid CTC/attention model with training `ctc_weight=0.3`.
- Model parameters logged by training: 46,197,266.

Training:

- Optimizer: Adam.
- Scheduler: warmup LR with `warmup_steps=25000`.
- Learning rate: `0.002`.
- Batch config: static batch size `64`.
- Gradient accumulation: `1`.
- Max epoch in config: `240`.
- Main training log: `train.20260615`.
- Resume log: `train.20260616`.
- The run started from epoch 0 at `2026-06-15 20:03:06`.
- Checkpoint `epoch_195.pt` was saved at `2026-06-16 10:06:48`.
- Training was resumed from `exp/conformer/epoch_195.pt`.
- Checkpoint `epoch_230.pt` was saved at `2026-06-16 16:21:21`.

Decoding:

- Decoding used the averaged checkpoint `exp/conformer/avg_30.pt`.
- Average checkpoint setting: `average_num=30`, `--val_best`.
- Decoding modes: `ctc_greedy_search`, `ctc_prefix_beam_search`, `attention`, `attention_rescoring`.
- Beam size: `10`.
- Decode batch size: `32`.
- Decode `ctc_weight=0.3`.
- Decode `reverse_weight=0.5`.

## Changes From The Original Conformer Recipe

This run keeps the original Conformer architecture but adapts the recipe for Ascend 910B3 reproduction:

- `run_npu.sh` uses NPU device training instead of CUDA GPU training.
- `torch_fsdp` is used as the training engine.
- HCCL is selected as the distributed backend.
- The AISHELL data path is changed to the local dataset path.
- Data preparation uses shard data instead of raw data for training and decoding.
- `nj` is increased to `32`.
- `stage` and `stop_stage` are provided by command line arguments.
- Training resumes from `exp/conformer/epoch_195.pt` for the later part of the run.
- `conf/train_conformer.yaml` changes the effective per-process batch setup from batch size `16` with `accum_grad=4` to batch size `64` with `accum_grad=1`.

## Reproduction Files

Files kept for this reproduction include:

- `conf/train_conformer.yaml`
- `run_npu.sh`
- `train.20260615`
- `train.20260616`
- `fusion_result.json`
- TensorBoard event files under `tensorboard/conformer/`
- Decoding outputs and CER reports under the four `exp/conformer/*` decoding directories

Large checkpoints, generated `exp/conformer/*.pt`, generated `exp/conformer/*.yaml`, and prepared `data/` files are intentionally not included.
