| seed | control | is_null | alpha_gt | alpha_rec | latent_corr | val_mse |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | REF genuine alpha=0.08 | False | 0.08 | 0.066876 | 0.31717 | 1.3006e-10 |
| 42 | alpha=0 (no coupling) | True | 0 | 0.037339 | 0.31804 | 1.9894e-10 |
| 42 | shuffled driver | True | 0.08 | 0.053926 | 0.0087196 | 4.2244e-11 |
| 42 | victim-only (no driver) | True | 0.08 | 0.074544 | 0.87896 | 6.2548e-10 |
| 42 | fake slow drift (alpha=0) | True | 0 | 0.037192 | 0.318 | 1.7121e-10 |
| 123 | REF genuine alpha=0.08 | False | 0.08 | 0.046006 | 0.40289 | 9.1411e-10 |
| 123 | alpha=0 (no coupling) | True | 0 | -0.00051124 | 0.40232 | 6.3772e-12 |
| 123 | shuffled driver | True | 0.08 | 0.060431 | 0.015006 | 7.163e-10 |
| 123 | victim-only (no driver) | True | 0.08 | 0.048346 | 0.82364 | 8.1195e-10 |
| 123 | fake slow drift (alpha=0) | True | 0 | -0.027004 | 0.40235 | 9.8692e-12 |
| 456 | REF genuine alpha=0.08 | False | 0.08 | 0.028188 | 0.20507 | 8.2916e-10 |
| 456 | alpha=0 (no coupling) | True | 0 | -0.034781 | 0.2051 | 4.7318e-10 |
| 456 | shuffled driver | True | 0.08 | 0.064348 | 0.056037 | 1.617e-10 |
| 456 | victim-only (no driver) | True | 0.08 | 0.042214 | 0.76734 | 3.2648e-10 |
| 456 | fake slow drift (alpha=0) | True | 0 | -0.017089 | 0.2051 | 5.1284e-10 |
