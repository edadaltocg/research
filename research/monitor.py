from collections import defaultdict


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0, "max": float("-inf"), "min": float("inf")})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
        metric["max"] = max(metric["max"], val)
        metric["min"] = min(metric["min"], val)

    def __str__(self):
        return " | ".join([
            "{metric_name}: avg={avg:.{float_precision}f}, max={max:.{float_precision}f}, min={min:.{float_precision}f}".format(
                metric_name=metric_name,
                avg=metric["avg"],
                max=metric["max"],
                min=metric["min"],
                float_precision=self.float_precision,
            )
            for (metric_name, metric) in self.metrics.items()
        ])


def _example():
    monitor = MetricMonitor(float_precision=2)

    # Step 2: Update metrics with new values
    monitor.update("accuracy", 0.8)
    monitor.update("accuracy", 0.85)
    monitor.update("loss", 0.5)
    monitor.update("loss", 0.45)

    # Step 3: Retrieve and print the current averages
    print(monitor)  # Output: accuracy: 0.83 | loss: 0.48

    # You can continue updating metrics
    monitor.update("accuracy", 0.9)
    monitor.update("loss", 0.4)

    # Print the updated averages
    print(monitor)  # Output: accuracy: 0.85 | loss: 0.45

    # Reset the monitor if needed
    monitor.reset()

    # After reset, the metrics will be empty
    print(monitor)  # Output: (an empty string, as no metrics have been updated yet)


if __name__ == "__main__":
    _example()
