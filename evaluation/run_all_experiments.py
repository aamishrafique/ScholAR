"""
Run all four evaluation experiments in sequence.

Requires built arXiv indexes for Experiment 4; BEIR datasets for 1–3.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "evaluation")

EXPERIMENTS = [
    ("Experiment 1 — Retrieval comparison", "run_beir.py"),
    ("Experiment 2 — Feedback gain", "run_feedback_experiment.py"),
    ("Experiment 3 — Clustering quality", "run_clustering_experiment.py"),
    ("Experiment 4 — Latency", "run_latency_experiment.py"),
]


def main():
    for label, script in EXPERIMENTS:
        path = os.path.join(EVAL, script)
        print(f"\n{'#'*60}\n  {label}\n{'#'*60}\n")
        result = subprocess.run([sys.executable, path], cwd=ROOT)
        if result.returncode != 0:
            print(f"Failed: {script} (exit {result.returncode})")
            sys.exit(result.returncode)
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
