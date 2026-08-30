import time
from typing import Dict, Any

class TelemetryLogger:
    def __init__(self):
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_cost = 0.0

    def add_metric(self, tokens: int, cost: float):
        self.total_tokens += tokens
        self.total_cost += cost

    def get_telemetry(self) -> Dict[str, Any]:
        end_time = time.time()
        return {
            "latency_ms": int((end_time - self.start_time) * 1000),
            "tokens": int(self.total_tokens),
            "cost_usd": self.total_cost
        }
