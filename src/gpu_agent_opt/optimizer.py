# gpu_agent_opt/optimizer.py
import itertools

class GridSearchOptimizer:
    """
    Simple optimizer that exhaustively tries all configs.
    """
    def __init__(self, search_space):
        self.search_space = search_space

    def iter_configs(self):
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))


class BayesianOptimizer:
    """
    Stub for Bayesian optimization. Right now falls back to grid search.
    """
    def __init__(self, search_space):
        self.gs = GridSearchOptimizer(search_space)

    def iter_configs(self):
        yield from self.gs.iter_configs()
