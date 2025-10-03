# gpu_agent_opt/executor.py
import time
import torch
from torch.cuda import nvtx


class Executor:
    """
    Runs a kernel/config with given input and returns result + runtime.
    Stage A: uses CUDA events for accurate GPU timing if available.
    """

    def __init__(self):
        pass

    def run(self, func, variant, *args, **kwargs):
        # NVTX marker for Nsight Systems timeline
        nvtx.range_push(f"Config {variant}")

        # Stage A timing: use CUDA events if on GPU, else fallback to wall clock
        if torch.cuda.is_available():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()

            output = func(*args, **kwargs)

            ender.record()
            torch.cuda.synchronize()
            runtime = starter.elapsed_time(ender) / 1000.0  # ms → sec
        else:
            start = time.time()
            output = func(*args, **kwargs)
            runtime = time.time() - start

        nvtx.range_pop()
        return output, runtime
