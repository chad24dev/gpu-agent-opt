# gpu_agent_opt/executor.py
class Executor:
    """
    Runs a kernel/config with given input and returns result.
    """
    def __init__(self):
        pass

    def run(self, func, config, *args, **kwargs):
        return func(*args, **config, **kwargs)
