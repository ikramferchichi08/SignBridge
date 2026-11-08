| Run | Model | Features | Top-1 (%) | Top-5 (%) | Macro-F1 | Params | Size (MB) |
|---|---|---|---|---|---|---|---|
| transformer_handlocal | transformer | hand-local | 31.5 ± 8.6 | 57.2 | 0.302 ± 0.084 | 348,772 | 1.41 |
| transformer_handlocal_lr1e3_e100 | transformer | hand-local | 31.3 ± 9.2 | 57.2 | 0.298 ± 0.080 | 348,772 | 1.41 |
| gru_handlocal_bidir_delta | gru | hand-local, delta, bidirectional | 29.6 ± 6.6 | 57.6 | 0.267 ± 0.058 | 644,324 | 2.58 |
| transformer_handlocal_delta | transformer | hand-local, delta | 29.3 ± 6.4 | 54.5 | 0.277 ± 0.059 | 397,540 | 1.60 |
| gru_handlocal | gru | hand-local | 22.6 ± 4.6 | 50.5 | 0.213 ± 0.035 | 273,508 | 1.10 |
| gru_bidir_delta | gru | delta, bidirectional | 20.2 ± 5.5 | 47.8 | 0.195 ± 0.057 | 644,324 | 2.58 |
| gru_bidir | gru | bidirectional | 19.9 ± 5.5 | 46.5 | 0.193 ± 0.047 | 595,556 | 2.39 |
| cnn_lstm_handlocal | cnn_lstm | hand-local | 18.9 ± 4.3 | 44.3 | 0.174 ± 0.048 | 372,068 | 1.50 |
| gru_delta | gru | delta | 18.8 ± 6.1 | 43.4 | 0.183 ± 0.065 | 322,276 | 1.29 |
| gru_baseline | gru | raw landmarks | 18.1 ± 5.3 | 41.7 | 0.171 ± 0.049 | 273,508 | 1.10 |
| gru_canonical | gru | canonical-hand | 17.7 ± 5.1 | 44.7 | 0.163 ± 0.040 | 273,508 | 1.10 |
