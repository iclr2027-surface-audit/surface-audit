# v3 dataset scale-up progress

Target n: 100    Max batches: 5..14 (10 batches)    Min yield: 15%

## Per-batch results

| Batch | Generated | Mech | Polarity | Judge | Drift | New (after dedup) | Yield % | Cumulative n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 20 | 17 | 17 | 15 | 10 | 10 | 50.0% | 41 |
| 6 | 20 | 7 | 7 | 6 | 6 | 6 | 30.0% | 47 |
| 7 | 20 | 17 | 17 | 17 | 12 | 12 | 60.0% | 59 |
| 8 | 20 | 9 | 9 | 9 | 6 | 6 | 30.0% | 65 |
| 9 | 20 | 16 | 16 | 15 | 7 | 7 | 35.0% | 72 |
| 10 | 20 | 14 | 14 | 10 | 4 | 4 | 20.0% | 76 |
| 11 | 20 | 18 | 18 | 16 | 12 | 12 | 60.0% | 88 |
| 12 | 20 | 17 | 17 | 17 | 9 | 9 | 45.0% | 97 |
| 13 | 20 | 15 | 15 | 12 | 4 | 4 | 20.0% | 101 |

Latest cumulative n = **101** (target 100)

## Loop terminated

- target reached: cum_n=101 >= 100