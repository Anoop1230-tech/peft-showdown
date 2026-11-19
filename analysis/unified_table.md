Scores: GLUE validation, x100, mean ± std over seeds 1-3 (SST-2 full: seed 1 only). Peak memory: max over all runs. Times: Tesla T4, fp16, batch 32.

| Method | SST-2 full acc | SST-2 1k acc | MRPC full acc | MRPC full F1 | MRPC 1k acc | MRPC 1k F1 | RTE full acc | RTE 1k acc | Trainable params | % trainable | Peak mem (MiB) | Time RTE full (s) | Time SST-2 full (s) | Checkpoint (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full FT | 94.0 | 90.4 ± 0.5 | 87.7 ± 1.4 | 91.2 ± 1.2 | 85.8 ± 2.2 | 89.9 ± 1.4 | 76.1 ± 2.2 | 69.9 ± 2.1 | 124,647,170 | 100.00 | 3264 | 158 | 492 | 475.5 |
| LoRA | 93.7 | 90.1 ± 0.5 | 87.1 ± 1.0 | 90.9 ± 0.6 | 84.3 ± 1.3 | 88.8 ± 0.9 | 74.0 ± 2.0 | 66.9 ± 3.2 | 887,042 | 0.71 | 1937 | 112 | 324 | 3.4 |
| Adapters | 94.2 | 90.4 ± 0.8 | 87.3 ± 0.6 | 91.0 ± 0.4 | 84.4 ± 1.0 | 89.1 ± 0.8 | 76.3 ± 1.8 | 71.2 ± 2.0 | 2,381,186 | 1.87 | 1884 | 106 | 378 | 9.1 |
| Prompt tuning | 91.2 | 73.7 ± 15.4 | 70.7 ± 1.9 | 81.5 ± 0.7 | 71.1 ± 2.5 | 81.9 ± 1.0 | 56.1 ± 2.6 | 52.8 ± 2.7 | 607,490 | 0.48 | 1923 | 117 | 385 | 2.3 |
