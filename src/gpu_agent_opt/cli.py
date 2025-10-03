# gpu_agent_opt/cli.py
import argparse
import importlib.util
import sys
import cloudpickle
from .agent import KernelAgent


def load_function(script_path, func_name):
    """Dynamically load a Python script and extract a function by name."""
    spec = importlib.util.spec_from_file_location("user_module", script_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)
    if not hasattr(user_module, func_name):
        raise ValueError(f"Function {func_name} not found in {script_path}")
    return getattr(user_module, func_name)


def parse_args(arg_str):
    """Parse key=val string into kwargs dict."""
    kwargs = {}
    if arg_str:
        for kv in arg_str.split(","):
            if "=" in kv:
                k, v = kv.split("=")
                k, v = k.strip(), v.strip()
                # Try to eval numbers/booleans, else keep as string
                try:
                    v = eval(v)
                except Exception:
                    pass
                kwargs[k] = v
    return kwargs


def main():
    ap = argparse.ArgumentParser("GPU Agent CLI")
    ap.add_argument("mode", choices=["profile", "autotune"], help="Run mode")
    ap.add_argument("script", help="Path to Python script containing target function")
    ap.add_argument("--func", required=True, help="Function name to run/profile")
    ap.add_argument("--args", default="", help="Comma-separated key=val args for function")
    ap.add_argument("--strategy", default="grid", choices=["grid", "bayesian"], help="Search strategy")
    ap.add_argument("--advanced", action="store_true", help="Enable advanced Nsight profiling after autotune")
    ap.add_argument("--passes", type=int, default=1, help="Number of Nsight passes (default=1 minimal)")
    args = ap.parse_args()

    # Load function
    func = load_function(args.script, args.func)
    kwargs = parse_args(args.args)

    # Create KernelAgent
    agent = KernelAgent(kernel_func=func, metrics=["sm_efficiency", "dram_utilization"])

    if args.mode == "autotune":
        best_cfg, best_score = agent.autotune(strategy=args.strategy, **kwargs)
        print("✅ Best Config:", best_cfg, "Score:", best_score)

        if args.advanced:
            agent.profiler.advanced = True
            agent.profiler.passes = args.passes
            scores = agent.profiler.evaluate(best_cfg, func=func, **kwargs)
            print("🔬 Advanced Nsight Metrics:", scores)

    elif args.mode == "profile":
        # Just profile once
        agent.profiler.advanced = True
        agent.profiler.passes = args.passes
        scores = agent.profiler.evaluate({}, func=func, **kwargs)
        print("🔬 Advanced Nsight Metrics:", scores)


if __name__ == "__main__":
    main()
