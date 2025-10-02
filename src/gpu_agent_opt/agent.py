# gpu_agent_opt/agent.py
from .profiler import NsightProfiler
from .executor import Executor
from .generator import KernelGenerator
from .optimizer import BayesianOptimizer
from .knowledge import KnowledgeBase

class KernelAgent:
    """
    Main agent class to orchestrate GPU kernel autotuning.
    Loop: Generate → Execute → Profile → Learn.
    """
    def __init__(self, kernel_func, metrics=None, gpu="default"):
        self.kernel_func = kernel_func
        self.metrics = metrics or ["sm_efficiency"]
        self.gpu = gpu

        # Core components
        self.profiler = NsightProfiler(self.metrics)
        self.executor = Executor()
        self.generator = KernelGenerator()
        self.kb = KnowledgeBase(gpu_name=gpu)

    def autotune(self, search_space, strategy="bayesian", *args, **kwargs):
        """
        Autotune kernel function with given search space.
        search_space: dict, e.g. {"block_x":[32,64], "batch_size":[128,256]}
        strategy: "bayesian" (default), could extend to "grid" etc.
        """
        if strategy == "bayesian":
            optimizer = BayesianOptimizer(search_space)
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented")

        best_cfg, best_score, best_output = None, -1, None

        for config in optimizer.iter_configs():
            print(f"[Agent] Trying config: {config}")

            # Generate variant (stubbed for now)
            variant = self.generator.generate(**config)

            # Run kernel
            try:
                output = self.executor.run(self.kernel_func, variant, *args, **kwargs)
            except Exception as e:
                print(f"[Agent] Config {config} failed with error: {e}")
                continue

            # Profile score
            score = self.profiler.evaluate(self.kernel_func, config, *args, **kwargs)
            print(f"[Agent] Config {config} → Score {score:.4f}")

            # Store in knowledge base
            self.kb.store(config, score)

            # Update best
            if score > best_score:
                best_cfg, best_score, best_output = config, score, output
                print(f"[Agent] ✅ New best config {best_cfg} with score {best_score:.4f}")

        return best_cfg, best_output

    def get_best_from_kb(self):
        """
        Retrieve best known config from knowledge base.
        """
        return self.kb.get_best()
