| method | val_F1 | test_F1 | test_P | test_R |
|---|---|---|---|---|
| BASELINE: 0.5*text + 0.5*image | 0.9087 | 0.9050 | 0.9324 | 0.9221 |
| E1 weighted fusion (w_text=0.5) | 0.9087 | 0.9050 | 0.9324 | 0.9221 |
| E1b ensemble-in-modality fusion (w_text=0.5) | 0.9082 | 0.9004 | 0.9124 | 0.9341 |
| E2 GBM on 5 cosine features | 0.9100 | 0.9063 | 0.9229 | 0.9303 |
| E3 GBM on all 16 features | 0.9106 | 0.9105 | 0.9275 | 0.9314 |
| FINAL (E3 GBM (all features) + symmetric) | 0.9112 | 0.9123 | 0.9339 | 0.9277 |