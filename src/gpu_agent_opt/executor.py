# gpu_agent_opt/executor.py
import time
import nvtx

class Executor:
    """
    Runs a kernel/config with given input and returns result.
    """

    def __init__(self):
        pass

    def run(self, func, variant, *args, **kwargs):
        start = time.time()
        nvtx.range_push(f"Config {variant}")
        output = func(*args, **kwargs)  # run kernel ONCE
        nvtx.range_pop()
        runtime = time.time() - start
        return output, runtime
