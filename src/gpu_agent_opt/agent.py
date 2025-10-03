import inspect
from .profiler import NsightProfiler
from .executor import Executor
from .generator import KernelGenerator
from .knowledge import KnowledgeBase
from tqdm import tqdm
from tabulate import tabulate
import time
import csv
import os


class KernelAgent:
    """
    Main agent class to orchestrate GPU kernel autotuning.
    Loop: Discover → Execute → Profile → Learn.
    """

    def __init__(self, kernel_func, metrics=None, gpu="default"):
        self.kernel_func = kernel_func
        self.metrics = metrics or ["sm_efficiency"]
        self.gpu = gpu

        # Core components
        self.profiler = NsightProfiler(metrics=self.metrics, advanced=False)
        self.executor = Executor()
        self.generator = KernelGenerator()
        self.kb = KnowledgeBase(gpu_name=gpu)

    # -----------------------
    # Auto Parameter Discovery
    # -----------------------
    def discover_params(self):
        sig = inspect.signature(self.kernel_func)
        discovered = {}
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                continue
            if isinstance(param.default, (int, float)):
                discovered[name] = param.default
        return discovered

    def build_search_space(self, discovered):
        search_space = {}
        for k, v in discovered.items():
            if "batch" in k.lower():
                search_space[k] = [max(8, v // 2), v, v * 2, v * 4]
            elif isinstance(v, float):
                search_space[k] = [max(0.1, v / 2), v, v * 2]
            else:
                search_space[k] = [v]
        return search_space

    # -----------------------
    # Autotune Entry
    # -----------------------
    def autotune(self, search_space=None, strategy="grid", *args, **kwargs):
        best_cfg, best_score = self.kb.get_best()
        if best_cfg:
            print(f"[Agent] ⚡ Using best known config {best_cfg} (score={best_score:.4f})")
            return best_cfg, best_score

        if search_space is None:
            discovered = self.discover_params()
            print(f"[Agent] 🔍 Discovered tunable params: {discovered}")
            search_space = self.build_search_space(discovered)
            print(f"[Agent] 🔧 Built search space: {search_space}")

        if strategy == "grid":
            best_cfg, best_score = self._grid_search(search_space, *args, **kwargs)
        elif strategy == "bayesian":
            best_cfg, best_score = self._bayes_opt(search_space, *args, **kwargs)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        # Ask user if they want advanced Nsight profiling
        resp = input("[Agent] Do you want to run advanced Nsight profiling on best config? (y/n): ").strip().lower()
        if resp == "y":
            self.profiler.advanced = True
            self.profiler.passes = 1  # keep minimal passes first
            scores = self.profiler.evaluate(best_cfg, func=self.kernel_func, *args, **kwargs)
            print(f"[Agent] Advanced Nsight results: {scores}")

        return best_cfg, best_score

    # -----------------------
    # Grid Search
    # -----------------------
    def _grid_search(self, search_space, *args, **kwargs):
        from itertools import product
        keys = list(search_space.keys())
        values = list(search_space.values())
        configs = [dict(zip(keys, v)) for v in product(*values)]

        best_cfg, best_score, trial_results = None, -1, []

        with tqdm(total=len(configs), desc="[Agent][Grid] Autotuning", unit="config") as pbar:
            for config in configs:
                start = time.time()
                print(f"\n[Agent][Grid] Trying {config}")

                try:
                    variant = self.generator.generate(**config)
                    output, runtime = self.executor.run(self.kernel_func, variant, *args, **{**kwargs, **config})
                    score = 1.0 / runtime  # simple throughput-like score
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[Agent][Grid] 🚨 OOM at {config}, skipping further growth")
                        continue
                    print(f"[Agent][Grid] Config {config} failed: {e}")
                    continue

                elapsed = time.time() - start
                trial_results.append((config, score, elapsed))
                self.kb.store(config, score)

                if score > best_score:
                    best_cfg, best_score = config, score
                    print(f"[Agent][Grid] ✅ New best {best_cfg} score={best_score:.4f}")

                pbar.set_postfix({"Last Time (s)": f"{elapsed:.2f}"})
                pbar.update(1)

        # Summary
        table = [(str(c), f"{s:.4f}", f"{t:.2f}s") for c, s, t in trial_results]
        print("\n[Agent][Grid] Summary:\n")
        print(tabulate(table, headers=["Config", "Score", "Time"], tablefmt="github"))
        self._save_results_csv(trial_results, strategy="grid")
        return best_cfg, best_score

    # -----------------------
    # Bayesian Optimization
    # -----------------------
    def _bayes_opt(self, search_space, *args, **kwargs):
        from skopt import gp_minimize
        from skopt.space import Categorical
        keys = list(search_space.keys())
        space = [Categorical(search_space[k], name=k) for k in keys]
        trial_results = []

        def objective(params):
            config = dict(zip(keys, params))
            print(f"\n[Agent][BO] Trying {config}")
            start = time.time()
            try:
                variant = self.generator.generate(**config)
                output, runtime = self.executor.run(self.kernel_func, variant, *args, **{**kwargs, **config})
                score = 1.0 / runtime
                self.kb.store(config, score)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    return 1e6
                print(f"[Agent][BO] Config {config} failed: {e}")
                return 1e6
            elapsed = time.time() - start
            trial_results.append((config, score, elapsed))
            print(f"[Agent][BO] Config {config} → Score {score:.4f}, Time {elapsed:.2f}s")
            return -score

        res = gp_minimize(objective, space, n_calls=10, n_initial_points=3, random_state=42)
        best_cfg = dict(zip(keys, res.x))
        best_score = -res.fun
        print(f"[Agent][BO] ✅ Best {best_cfg} with score={best_score:.4f}")

        table = [(str(c), f"{s:.4f}", f"{t:.2f}s") for c, s, t in trial_results]
        print("\n[Agent][BO] Summary:\n")
        print(tabulate(table, headers=["Config", "Score", "Time"], tablefmt="github"))
        self._save_results_csv(trial_results, strategy="bayesian")
        return best_cfg, best_score

    def _save_results_csv(self, trial_results, strategy="grid"):
        os.makedirs("runs", exist_ok=True)
        filename = os.path.join("runs", f"autotune_results_{strategy}.csv")
        with open(filename, mode="w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["Config", "Score", "Time (s)"])
            for config, score, elapsed in trial_results:
                writer.writerow([str(config), f"{score:.4f}", f"{elapsed:.2f}"])
        print(f"[Agent] Results saved to {filename}")

    def get_best_from_kb(self):
        return self.kb.get_best()
