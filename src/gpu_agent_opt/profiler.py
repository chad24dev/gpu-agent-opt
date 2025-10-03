# gpu_agent_opt/profiler.py
import subprocess
import json
import tempfile
import os

class NsightProfiler:
    """
    Wraps Nsight Compute CLI to collect GPU kernel metrics.
    """
    def __init__(self, metrics=None, ncu_path="/usr/local/cuda/bin/ncu"):
        self.metrics = metrics or ["sm_efficiency", "dram_utilization"]
        self.ncu_path = ncu_path

    def evaluate(self, config, runtime=None, output=None):
        # In real Nsight, you’d parse profiler logs
        # Here: just return inverse runtime as "score"
        if runtime is not None:
            return 1.0 / runtime
        else:
            return 0.0
