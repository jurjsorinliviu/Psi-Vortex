| seed | variant | init | arch_select | bic | epochs_to_thr | wall_s | val_mse | params | eff_dof | alpha_err_pct | manual_decisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | Baseline Psi-xLSTM | random | False | False | 114 | 4.1971 | 4.5448e-07 | 16305 | 5.4833 | 41.623 | 4 |
| 42 | Init-only (expert) | physics | False | False | 10 | 4.2622 | 1.3006e-10 | 16305 | 2.0185 | 16.405 | 3 |
| 42 | Init-only (auto) | auto | False | False | 10 | 3.9715 | 1.3006e-10 | 16305 | 2.0185 | 16.405 | 1 |
| 42 | BIC-only | random | False | True | 114 | 4.3872 | 4.5448e-07 | 16305 | 5.3938 | 35.864 | 3 |
| 42 | Arch-select only | random | True | False | 120 | 4.3213 | 1.2174e-06 | 4313 | 5.8282 | 57.742 | 2 |
| 42 | Init + BIC | physics | False | True | 10 | 4.2648 | 1.3006e-10 | 16305 | 2.019 | 78.19 | 1 |
| 42 | Full Psi-Vortex | auto | True | True | 5 | 4.1398 | 1.5842e-09 | 4313 | 2.0166 | 100 | 0 |
| 123 | Baseline Psi-xLSTM | random | False | False | 24 | 4.2811 | 1.8327e-06 | 16305 | 5.4569 | 90.865 | 4 |
| 123 | Init-only (expert) | physics | False | False | 7 | 4.1732 | 9.1411e-10 | 16305 | 2.0175 | 42.492 | 3 |
| 123 | Init-only (auto) | auto | False | False | 7 | 4.1203 | 9.1411e-10 | 16305 | 2.0175 | 42.492 | 1 |
| 123 | BIC-only | random | False | True | 24 | 4.1862 | 1.8327e-06 | 16305 | 4.9834 | 192.25 | 3 |
| 123 | Arch-select only | random | True | False | 47 | 4.1628 | 1.4874e-05 | 63329 | 5.2443 | 205.18 | 2 |
| 123 | Init + BIC | physics | False | True | 7 | 4.2747 | 9.1411e-10 | 16305 | 2.0146 | 205.39 | 1 |
| 123 | Full Psi-Vortex | auto | True | True | 4 | 3.9773 | 1.6385e-09 | 63329 | 2.0129 | 100 | 0 |
| 456 | Baseline Psi-xLSTM | random | False | False | 4 | 4.0392 | 2.9169e-05 | 16305 | 5.6479 | 247.95 | 4 |
| 456 | Init-only (expert) | physics | False | False | 3 | 4.0873 | 8.2916e-10 | 16305 | 2.0179 | 64.765 | 3 |
| 456 | Init-only (auto) | auto | False | False | 3 | 3.7648 | 8.2916e-10 | 16305 | 2.0179 | 64.765 | 1 |
| 456 | BIC-only | random | False | True | 4 | 4.2243 | 2.9169e-05 | 16305 | 5.5511 | 309.94 | 3 |
| 456 | Arch-select only | random | True | False | 46 | 4.0868 | 6.4382e-06 | 63329 | 5.2715 | 185.89 | 2 |
| 456 | Init + BIC | physics | False | True | 3 | 4.1779 | 8.2916e-10 | 16305 | 2.016 | 301.85 | 1 |
| 456 | Full Psi-Vortex | auto | True | True | 5 | 4.232 | 1.8891e-10 | 63329 | 2.0118 | 100 | 0 |
