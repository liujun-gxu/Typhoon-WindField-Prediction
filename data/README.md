# Data directory

No raw or derived data are distributed with this code release. Prepare the
following files yourself before training:

```text
data/datasets/
  fixed_t0_6h_train_raw.npz
  fixed_t0_6h_val_raw.npz
  fixed_t0_6h_test_raw.npz
  fixed_t0_12h_train_raw.npz
  ...
  fixed_t0_24h_test_raw.npz
```

Every NPZ must contain `X` with shape `(N, 4, 2, 31, 31)` and `Y` with shape
`(N, 2, 31, 31)`. The year split is 2020--2021/train, 2022/validation, and
2023/test. Use `data_processing/` to reproduce the fixed-t0 target:
the target is extracted at initialization position C(t0) from the future ERA5 field.
The storm identities and per-horizon sample counts for each split are listed in 
storm_lists.md (Table S1 of the manuscript).