from .profiler import NsightProfiler
from .executor import Executor
from .generator import KernelGenerator
from .knowledge import KnowledgeBase
from tqdm import tqdm
from tabulate import tabulate
import time
from skopt import gp_minimize
from skopt.space import Categorical
import csv
import os


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

    def autotune(self, search_space, strategy="grid", *args, **kwargs):
        if strategy == "grid":
            return self._grid_search(search_space, *args, **kwargs)
        elif strategy == "bayesian":
            return self._bayes_opt(search_space, *args, **kwargs)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    # -----------------------
    # Grid Search (brute force)
    # -----------------------
    def _grid_search(self, search_space, *args, **kwargs):
        from itertools import product

        keys = list(search_space.keys())
        values = list(search_space.values())
        configs = [dict(zip(keys, v)) for v in product(*values)]

        best_cfg, best_score, best_output = None, -1, None
        trial_results = []

        with tqdm(total=len(configs), desc="[Agent][Grid] Autotuning", unit="config") as pbar:
            for config in configs:
                start = time.time()
                print(f"\n[Agent][Grid] Trying {config}")

                try:
                    variant = self.generator.generate(**config)
                    output, runtime = self.executor.run(self.kernel_func, variant, *args, **kwargs)
                    score = self.profiler.evaluate(config, runtime=runtime, output=output)
                except Exception as e:
                    print(f"[Agent][Grid] Config {config} failed: {e}")
                    pbar.update(1)
                    continue

                elapsed = time.time() - start
                trial_results.append((config, score, elapsed))
                self.kb.store(config, score)

                if score > best_score:
                    best_cfg, best_score, best_output = config, score, output
                    print(f"[Agent][Grid] ✅ New best {best_cfg} score={best_score:.4f}")

                pbar.set_postfix({"Last Time (s)": f"{elapsed:.2f}"})
                pbar.update(1)

        # ---- Print summary ----
        table = [(str(c), f"{s:.4f}", f"{t:.2f}s") for c, s, t in trial_results]
        print("\n[Agent][Grid] Summary:\n")
        print(tabulate(table, headers=["Config", "Score", "Time"], tablefmt="github"))
        self._save_results_csv(trial_results, strategy="grid")
        return best_cfg, best_output

    # -----------------------
    # Bayesian Optimization
    # -----------------------
    def _bayes_opt(self, search_space, *args, **kwargs):
        keys = list(search_space.keys())
        space = [Categorical(search_space[k], name=k) for k in keys]

        trial_results = []

        def objective(params):
            config = dict(zip(keys, params))
            print(f"\n[Agent][BO] Trying {config}")
            start = time.time()

            try:
                variant = self.generator.generate(**config)
                output = self.executor.run(self.kernel_func, variant, *args, **kwargs)
                score = self.profiler.evaluate(self.kernel_func, config, *args, **kwargs)
                self.kb.store(config, score)
            except Exception as e:
                print(f"[Agent][BO] Config {config} failed: {e}")
                return 1e6  # big penalty

            elapsed = time.time() - start
            trial_results.append((config, score, elapsed))
            print(f"[Agent][BO] Config {config} → Score {score:.4f}, Time {elapsed:.2f}s")
            return -score  # minimize negative

        n_calls = 10
        with tqdm(total=n_calls, desc="[Agent][BO] Autotuning", unit="trial") as pbar:
            def callback(res):
                pbar.update(1)

            res = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=n_calls,
                n_initial_points=3,
                random_state=42,
                callback=callback
            )
            pbar.close()

        best_cfg = dict(zip(keys, res.x))
        best_score = -res.fun
        print(f"[Agent][BO] ✅ Best {best_cfg} with score={best_score:.4f}")

        # ---- Print summary ----
        table = [(str(c), f"{s:.4f}", f"{t:.2f}s") for c, s, t in trial_results]
        print("\n[Agent][BO] Summary:\n")
        print(tabulate(table, headers=["Config", "Score", "Time"], tablefmt="github"))
        self._save_results_csv(trial_results, strategy="bayesian")
        return best_cfg, best_score

    def _save_results_csv(self, trial_results, strategy="grid"):
        os.makedirs("runs", exist_ok=True)
        filename = os.path.join("runs", f"autotune_results_{strategy}.csv")

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Config", "Score", "Time (s)"])
            for config, score, elapsed in trial_results:
                writer.writerow([str(config), f"{score:.4f}", f"{elapsed:.2f}"])

        print(f"[Agent] Results saved to {filename}")

    def get_best_from_kb(self):
        """Retrieve best known config from knowledge base."""
        return self.kb.get_best()
