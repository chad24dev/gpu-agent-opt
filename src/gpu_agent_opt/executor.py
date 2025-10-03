# gpu_agent_opt/executor.py
import time


class Executor:
    """
    Runs a kernel/config with given input and returns result.
    """

    def __init__(self):
        pass

    def run(self, func, variant, *args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)  # run kernel ONCE
        runtime = time.time() - start
        return output, runtime
