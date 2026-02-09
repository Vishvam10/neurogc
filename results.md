# Benchmark Results

_Generated on February 09, 2026 at 09:40_


## Architecture: `classical`

### Absolute Metrics

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-09-03 | 39.140 | 54.444 | 30.174 | 3469.857 | 4686.919 | 35.000 |
| 09-02-2026-09-39 | 34.496 | 54.606 | 41.522 | 2949.438 | 3765.277 | 43.000 |

### Run-over-Run Changes

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-09-39 | 🟢 -11.9% | 🔴 +0.3% | 🟢 +37.6% | 🟢 -15.0% | 🟢 -19.7% | 🔴 +22.9% |

## Architecture: `feedforward`

### Absolute Metrics

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-09-06 | 41.400 | 55.296 | 29.550 | 3997.397 | 5321.253 | 32.000 |
| 09-02-2026-09-23 | 40.898 | 54.974 | 35.892 | 4008.872 | 5644.187 | 38.000 |

### Run-over-Run Changes

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-09-23 | 🟢 -1.2% | 🟢 -0.6% | 🟢 +21.5% | 🔴 +0.3% | 🔴 +6.1% | 🔴 +18.8% |

## Architecture: `lstm`

### Absolute Metrics

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 31-01-2026-20-06 | 33.540 | 50.123 | 30.948 | 3574.343 | 4649.743 | 36.000 |
| 31-01-2026-20-15 | 41.697 | 53.453 | 29.033 | 4361.017 | 9013.018 | 30.000 |
| 09-02-2026-08-09 | 37.189 | 46.111 | 28.810 | 3738.008 | 4823.195 | 33.000 |
| 09-02-2026-08-48 | 5.250 | 25.800 | 8.891 | 34.480 | 34.480 | 0.000 |

### Run-over-Run Changes

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 31-01-2026-20-15 | 🔴 +24.3% | 🔴 +6.6% | 🔴 -6.2% | 🔴 +22.0% | 🔴 +93.8% | 🟢 -16.7% |
| 09-02-2026-08-09 | 🟢 -10.8% | 🟢 -13.7% | 🔴 -0.8% | 🟢 -14.3% | 🟢 -46.5% | 🔴 +10.0% |
| 09-02-2026-08-48 | 🟢 -85.9% | 🟢 -44.0% | 🔴 -69.1% | 🟢 -99.1% | 🟢 -99.3% | 🟢 -100.0% |

## Architecture: `transformer`

### Absolute Metrics

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-09-12 | 35.606 | 55.138 | 30.205 | 3749.713 | 4870.470 | 35.000 |
| 09-02-2026-09-33 | 38.888 | 54.983 | 31.617 | 3702.234 | 4730.179 | 36.000 |

### Run-over-Run Changes

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-09-33 | 🔴 +9.2% | 🟢 -0.3% | 🟢 +4.7% | 🟢 -1.3% | 🟢 -2.9% | 🔴 +2.9% |

## Architecture: `transformers`

### Absolute Metrics

| Run | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| 09-02-2026-08-30 | 39.148 | 50.729 | 30.934 | 3767.105 | 5165.904 | 33.000 |

## Cross-Architecture Comparison (Latest Runs)

| Architecture | Avg CPU (%) | Avg Memory (%) | Avg RPS | P95 Latency (ms) | P99 Latency (ms) | GC Events |
| --- | --- | --- | --- | --- | --- | --- |
| classical | 34.496 | 54.606 | 41.522 | 2949.438 | 3765.277 | 43.000 |
| feedforward | 40.898 | 54.974 | 35.892 | 4008.872 | 5644.187 | 38.000 |
| lstm | 5.250 | 25.800 | 8.891 | 34.480 | 34.480 | 0.000 |
| transformer | 38.888 | 54.983 | 31.617 | 3702.234 | 4730.179 | 36.000 |
| transformers | 39.148 | 50.729 | 30.934 | 3767.105 | 5165.904 | 33.000 |

### 🏆 Winners

- **Avg CPU (%)** → `lstm`
- **Avg Memory (%)** → `lstm`
- **P95 Latency (ms)** → `lstm`
- **P99 Latency (ms)** → `lstm`
- **GC Events** → `lstm`
- **Avg RPS** → `1      classical
1    feedforward
1    transformer
Name: arch, dtype: str`