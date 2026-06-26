# Gravitational-Wave HAE — Evaluation Report
Generated: 2026-06-24 11:44

## Reconstruction metrics (MSE / Overlap / PC1)

| Model          | test1 MSE   | test1 Overlap | test1 PC1  | test2 MSE   | test2 Overlap | test2 PC1  |
|:--------------|----------:|----------:|------:|----------:|----------:|------:|
| hae            |   1.32e-04 |   0.999936 | 0.6512 |   1.29e-04 |   0.999937 | 0.6916 |
| cae            |   1.80e-05 |   0.999993 | 0.7876 |   1.90e-05 |   0.999992 | 0.8254 |
| properhae      |   1.08e-04 |   0.999949 | 0.7143 |   1.05e-04 |   0.999953 | 0.7475 |
| porthae        |   1.68e-04 |   0.999925 | 0.6266 |   1.56e-04 |   0.999927 | 0.6595 |

## Training MSE

| Model          | Train MSE   |
|:--------------|------------:|
| hae            | 0.000157 |
| cae            | 0.000026 |

## Noise robustness (test1)

| Metric                        | Value      |
|:------------------------------|----------:|
| noise_robustness_mse_hae       |   1.014732 |
| noise_robustness_mse_cae       |   1.037817 |
| noise_robustness_overlap_hae   |   0.255869 |
| noise_robustness_overlap_cae   |   0.248150 |

## Noise robustness (test2)

| Metric                        | Value      |
|:------------------------------|----------:|
| noise_robustness_mse_hae       |   1.064633 |
| noise_robustness_mse_cae       |   1.089256 |
| noise_robustness_overlap_hae   |   0.207047 |
| noise_robustness_overlap_cae   |   0.198511 |
