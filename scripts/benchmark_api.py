import statistics
import time
import urllib.request


URL = "http://127.0.0.1:8000/recommend/1?k=10"

NUM_REQUESTS = 100


def main():

    latencies = []

    for _ in range(NUM_REQUESTS):

        start = time.perf_counter()

        with urllib.request.urlopen(URL) as response:
            response.read()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies.append(latency_ms)

    latencies.sort()

    mean = statistics.mean(latencies)

    p50 = latencies[
        int(0.50 * len(latencies))
    ]

    p95 = latencies[
        int(0.95 * len(latencies))
    ]

    p99 = latencies[
        int(0.99 * len(latencies))
    ]

    total_time = sum(latencies) / 1000

    throughput = (
        NUM_REQUESTS / total_time
    )

    print()
    print("========================================")
    print("       API BENCHMARK")
    print("========================================")
    print()

    print(f"Requests:       {NUM_REQUESTS}")
    print(f"Mean latency:   {mean:.3f} ms")
    print(f"P50 latency:    {p50:.3f} ms")
    print(f"P95 latency:    {p95:.3f} ms")
    print(f"P99 latency:    {p99:.3f} ms")
    print(f"Throughput:     {throughput:.2f} req/s")
    print()


if __name__ == "__main__":
    main()
