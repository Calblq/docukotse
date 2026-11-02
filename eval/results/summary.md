| Configuration | Hit@5 | Answer accuracy | Taglish accuracy | Photo/scan accuracy | Abstention accuracy | False abstain | Faithfulness | Confidently wrong | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| Vector only | 100.0% | 96.6% | 85.7% | 85.7% | 37.5% | 3.4% | 97.0% | 13.5% | 7.78s / 8.84s |
| Hybrid (vector + keyword, RRF) | 100.0% | 96.6% | 85.7% | 85.7% | 37.5% | 3.4% | 93.9% | 13.5% | 7.63s / 9.5s |
| Hybrid + rerank + abstain | 100.0% | 96.6% | 85.7% | 85.7% | 100.0% | 3.4% | 100.0% | 0.0% | 7.79s / 9.34s |
| + verification | 100.0% | 96.6% | 85.7% | 85.7% | 100.0% | 3.4% | 100.0% | 0.0% | 7.59s / 8.87s |
| + vehicle scoping + Taglish translation (full system) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 17.85s / 28.89s |
| llama3.2:3b, everything except verification | 100.0% | 96.6% | 100.0% | 100.0% | 100.0% | 0.0% | 72.4% | 2.7% | 6.09s / 9.11s |
| llama3.2:3b, full system | 100.0% | 86.2% | 71.4% | 100.0% | 87.5% | 13.8% | 88.5% | 2.7% | 6.1s / 12.01s |
