LR search: dev accuracy (our 10% split of the training set) x100, seed 1, full data. **Bold** = best LR on that task; \* = LR used for all final runs. GLUE validation accuracy in brackets.

| Method | LR | RTE dev (val) | MRPC dev (val) |
| --- | --- | --- | --- |
| Full FT | 1e-05 | 76.7 (76.2) | **88.3 (87.5)** |
| Full FT | 2e-05\* | **78.3 (76.5)** | 87.2 (87.5) |
| Full FT | 3e-05 | **78.3 (76.5)** | 86.9 (89.7) |
| LoRA | 0.0001 | 70.7 (69.3) | 86.9 (85.3) |
| LoRA | 0.0003 | 74.7 (74.7) | **88.6 (86.5)** |
| LoRA | 0.0005\* | **75.9 (76.2)** | 88.3 (86.5) |
| Adapters | 0.0001 | 71.1 (68.2) | 86.9 (86.3) |
| Adapters | 0.0003\* | **77.9 (78.0)** | 87.5 (87.0) |
| Adapters | 0.001 | 75.9 (76.2) | **88.0 (90.0)** |
| Prompt tuning | 0.001 | 52.2 (56.0) | 69.2 (69.6) |
| Prompt tuning | 0.01\* | **55.8 (55.6)** | 72.8 (72.3) |
| Prompt tuning | 0.03 | 52.2 (57.4) | **73.8 (69.6)** |

- Full FT: best mean dev accuracy at LR 2e-05 (final runs used 2e-05).
- LoRA: best mean dev accuracy at LR 0.0005 (final runs used 0.0005).
- Adapters: best mean dev accuracy at LR 0.0003 (final runs used 0.0003).
- Prompt tuning: best mean dev accuracy at LR 0.01 (final runs used 0.01).
