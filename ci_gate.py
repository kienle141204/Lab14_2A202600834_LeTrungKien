"""
CI/CD evaluation quality gate.

From the lecture: "Agent không pass eval = không được deploy, giống unit test."
This script runs the golden-dataset benchmark and BLOCKS (exit 1) when a metric
falls below its threshold — wire it into CI after the unit tests.

Thresholds are overridable via env vars so different domains can tune them:
    FAITHFULNESS_GATE (default 0.40)   # raise to 0.70 in a high-stakes domain
    RELEVANCE_GATE    (default 0.20)   # heuristic relevance is harsh; keep loose
    COMPLETENESS_GATE (default 0.50)

Run:
    python ci_gate.py        # exit 0 = pass gate, exit 1 = block deploy
"""

from __future__ import annotations

import os
import sys

from benchmark import GOLDEN, agent_fn
from solution.solution import BenchmarkRunner, RAGASEvaluator

GATES = {
    "avg_faithfulness": float(os.getenv("FAITHFULNESS_GATE", "0.40")),
    "avg_relevance": float(os.getenv("RELEVANCE_GATE", "0.20")),
    "avg_completeness": float(os.getenv("COMPLETENESS_GATE", "0.50")),
}


def main() -> int:
    runner = BenchmarkRunner()
    results = runner.run(GOLDEN, agent_fn, RAGASEvaluator())
    report = runner.generate_report(results)

    print(f"Benchmark: {report['passed']}/{report['total']} passed "
          f"(pass_rate={report['pass_rate']:.2%})")

    failed_gates: list[str] = []
    for metric, threshold in GATES.items():
        value = report[metric]
        ok = value >= threshold
        print(f"  {'OK ' if ok else 'FAIL'} {metric}={value:.3f} (gate >= {threshold})")
        if not ok:
            failed_gates.append(metric)

    if failed_gates:
        print(f"\nQUALITY GATE FAILED on: {', '.join(failed_gates)} — blocking deploy.")
        return 1
    print("\nQUALITY GATE PASSED — safe to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
