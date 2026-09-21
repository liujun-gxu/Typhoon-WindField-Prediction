# Storm Lists and Sample Counts

This file provides the exact train/validation/test tropical-cyclone identities and the
corresponding fixed-<i>t</i><sub>0</sub> sample counts for the 6-, 12-, and 24-h forecasting tasks, as used by
the experiments reported in the manuscript
*Multi-Horizon Typhoon Wind Field Prediction via a Lightweight CNN-LSTM Network: Error Growth and
Cross-Year Robustness at 6-24 h Lead Times* (Atmosphere). It corresponds to **Table S1 of the
Supplementary Materials**. Storm identities and sample counts were extracted directly from the
metadata CSV files used by the fixed-<i>t</i><sub>0</sub> experiments. The partition follows Section 4.1 of the
main text: no tropical cyclone case is shared among the training, validation, and testing subsets.
A value of 0 indicates that no usable sample from the corresponding storm is available for that
forecast horizon. For the 24-h forecasting task, only 23 of the 24 validation storms contribute usable samples (CMA-2022-07 contributes none), consistent with the per-horizon storm counts in Table 2 of the main text.

## Training subset (2020)

| Storm UID | CMA storm code | Storm name | 6 h samples | 12 h samples | 24 h samples |
|---|---|---|---|---|---|
| CMA-2020-01 | 2001 | Vongfong | 35 | 34 | 32 |
| CMA-2020-02 | 2002 | Nuri | 22 | 20 | 16 |
| CMA-2020-03 | 2003 | Sinlaku | 6 | 6 | 6 |
| CMA-2020-04 | 2004 | Hagupit | 30 | 30 | 30 |
| CMA-2020-05 | 2005 | Jangmi | 14 | 14 | 14 |
| CMA-2020-06 | 2006 | Mekkhala | 12 | 10 | 6 |
| CMA-2020-07 | 0 | (nameless) | 10 | 9 | 7 |
| CMA-2020-08 | 2007 | Higos | 18 | 16 | 12 |
| CMA-2020-09 | 2008 | Bavi | 24 | 24 | 24 |
| CMA-2020-10 | 2009 | Maysak | 24 | 24 | 24 |
| CMA-2020-11 | 2010 | Haishen | 26 | 26 | 26 |
| CMA-2020-12 | 2011 | Noul | 12 | 12 | 12 |
| CMA-2020-13 | 2012 | Dolphin | 23 | 23 | 23 |
| CMA-2020-15 | 2014 | Chan-hom | 48 | 48 | 48 |
| CMA-2020-16 | 2015 | Linfa | 14 | 14 | 14 |
| CMA-2020-17 | 2016 | Nangka | 14 | 14 | 14 |
| CMA-2020-18 | 0 | (nameless) | 8 | 7 | 5 |
| CMA-2020-19 | 2017 | Saudel | 23 | 23 | 23 |
| CMA-2020-20 | 0 | (nameless) | 11 | 10 | 8 |
| CMA-2020-21 | 2018 | Molave | 19 | 19 | 19 |
| CMA-2020-22 | 2019 | Goni | 41 | 41 | 39 |
| CMA-2020-23 | 2020 | Atsani | 33 | 32 | 30 |
| CMA-2020-24 | 2021 | Etau | 11 | 11 | 11 |
| CMA-2020-25 | 2022 | Vamco | 25 | 25 | 25 |
| CMA-2020-26 | 2023 | Krovanh | 17 | 17 | 17 |

**Subtotal (2020): 25 storms | 520 (6 h) | 509 (12 h) | 485 (24 h)**

## Training subset (2021)

| Storm UID | CMA storm code | Storm name | 6 h samples | 12 h samples | 24 h samples |
|---|---|---|---|---|---|
| CMA-2021-01 | 2101 | Dujuan | 24 | 23 | 21 |
| CMA-2021-02 | 2102 | Surigae | 52 | 52 | 52 |
| CMA-2021-03 | 0 | (nameless) | 9 | 8 | 6 |
| CMA-2021-04 | 2103 | Choi-wan | 31 | 30 | 28 |
| CMA-2021-05 | 2104 | Koguma | 8 | 8 | 8 |
| CMA-2021-06 | 2105 | Champi | 24 | 24 | 23 |
| CMA-2021-07 | 0 | (nameless) | 8 | 8 | 8 |
| CMA-2021-08 | 2106 | In-fa | 79 | 78 | 76 |
| CMA-2021-09 | 2107 | Cempaka | 24 | 24 | 24 |
| CMA-2021-10 | 2108 | Nepartak | 7 | 7 | 7 |
| CMA-2021-11 | 2109 | Lupit | 48 | 48 | 48 |
| CMA-2021-12 | 2110 | Mirinae | 21 | 21 | 21 |
| CMA-2021-14 | 2112 | Omais | 23 | 23 | 23 |
| CMA-2021-16 | 2113 | Conson | 23 | 23 | 23 |
| CMA-2021-17 | 2114 | Chanthu | 54 | 53 | 51 |
| CMA-2021-18 | 2115 | Dianmu | 4 | 4 | 4 |
| CMA-2021-19 | 2116 | Mindulle | 34 | 34 | 34 |
| CMA-2021-20 | 2117 | Lionrock | 18 | 18 | 18 |
| CMA-2021-21 | 2118 | Kompasu | 26 | 26 | 26 |
| CMA-2021-23 | 2120 | Malou | 24 | 24 | 24 |
| CMA-2021-24 | 0 | (nameless) | 9 | 9 | 9 |
| CMA-2021-25 | 2121 | Nyatoh | 19 | 19 | 19 |
| CMA-2021-26 | 2122 | Rai | 35 | 34 | 32 |

**Subtotal (2021): 23 storms | 604 (6 h) | 598 (12 h) | 585 (24 h)**

**Training total (2020-2021): 48 storms | 1124 (6 h) | 1107 (12 h) | 1070 (24 h)**

## Validation subset (2022)

| Storm UID | CMA storm code | Storm name | 6 h samples | 12 h samples | 24 h samples |
|---|---|---|---|---|---|
| CMA-2022-01 | 2201 | Malakas | 28 | 28 | 28 |
| CMA-2022-02 | 2202 | Megi | 11 | 10 | 8 |
| CMA-2022-03 | 2203 | Chaba | 54 | 53 | 51 |
| CMA-2022-04 | 2204 | Aere | 35 | 35 | 34 |
| CMA-2022-05 | 2205 | Songda | 18 | 17 | 15 |
| CMA-2022-06 | 2206 | Trases | 9 | 8 | 6 |
| CMA-2022-07 | 0 | (nameless) | 6 | 4 | 0 |
| CMA-2022-08 | 2207 | Mulan | 11 | 11 | 11 |
| CMA-2022-09 | 2208 | Meari | 16 | 16 | 16 |
| CMA-2022-10 | 2209 | Ma-on | 22 | 22 | 20 |
| CMA-2022-12 | 2211 | Hinnamnor | 33 | 33 | 33 |
| CMA-2022-13 | 0 | (nameless) | 6 | 5 | 3 |
| CMA-2022-14 | 2212 | Muifa | 51 | 51 | 48 |
| CMA-2022-16 | 2214 | Nanmadol | 29 | 28 | 26 |
| CMA-2022-17 | 2215 | Talas | 21 | 21 | 20 |
| CMA-2022-18 | 2216 | Noru | 24 | 24 | 24 |
| CMA-2022-19 | 2217 | Kulap | 10 | 10 | 10 |
| CMA-2022-20 | 2218 | Roke | 12 | 12 | 12 |
| CMA-2022-22 | 2219 | Sonca | 5 | 5 | 4 |
| CMA-2022-23 | 2220 | Nesat | 18 | 18 | 18 |
| CMA-2022-25 | 0 | (nameless) | 12 | 11 | 9 |
| CMA-2022-26 | 2222 | Nalgae | 34 | 32 | 28 |
| CMA-2022-27 | 2223 | Banyan | 13 | 12 | 10 |
| CMA-2022-29 | 2225 | Pakhar | 12 | 11 | 9 |

**Subtotal (2022): 24 storms | 490 (6 h) | 477 (12 h) | 443 (24 h)**

## Testing subset (2023)

| Storm UID | CMA storm code | Storm name | 6 h samples | 12 h samples | 24 h samples |
|---|---|---|---|---|---|
| CMA-2023-01 | 0 | (nameless) | 8 | 7 | 5 |
| CMA-2023-03 | 2302 | Mawar | 40 | 39 | 37 |
| CMA-2023-04 | 2303 | Guchol | 28 | 28 | 28 |
| CMA-2023-05 | 2304 | Talim | 22 | 22 | 22 |
| CMA-2023-06 | 2305 | Doksuri | 56 | 54 | 50 |
| CMA-2023-07 | 2306 | Khanun | 64 | 63 | 61 |
| CMA-2023-08 | 2307 | Lan | 27 | 27 | 27 |
| CMA-2023-10 | 2309 | Saola | 49 | 49 | 49 |
| CMA-2023-11 | 2310 | Damrey | 3 | 3 | 3 |
| CMA-2023-12 | 2311 | Haikui | 84 | 84 | 80 |
| CMA-2023-13 | 2312 | Kirogi | 15 | 14 | 12 |
| CMA-2023-14 | 2313 | Yun-yeung | 14 | 13 | 11 |
| CMA-2023-15 | 0 | (nameless) | 1 | 1 | 1 |
| CMA-2023-16 | 2314 | Koinu | 46 | 45 | 43 |
| CMA-2023-17 | 2315 | Bolaven | 9 | 9 | 9 |
| CMA-2023-18 | 2316 | Sanba | 1 | 1 | 1 |
| CMA-2023-19 | 0 | (nameless) | 14 | 13 | 11 |
| CMA-2023-20 | 2317 | Jelawat | 9 | 8 | 6 |

**Subtotal (2023): 18 storms | 490 (6 h) | 480 (12 h) | 456 (24 h)

## Per-sample indices

Per-sample indices (`sample_id`, `storm_uid`, `t0`, `target_time`) are not distributed
directly, but are reconstructed deterministically by `data_processing/prepare_tracks.py`
from the storm lists given here. The training pipeline emits one metadata CSV per split
with exactly these columns (see the `test_predictions_metadata.csv` produced by the
training scripts for the format).