import tempfile
import subprocess
import os
import cloudpickle


class NsightProfiler:
    """
    Nsight Compute based profiler.
    Stage A: cheap (executor timing, CUDA events) → always runs in Agent.
    Stage B: optional advanced profiling via Nsight Compute (ncu).
    """

    def __init__(self, metrics=None, passes=1, advanced=False):
        """
        Args:
            metrics: list of Nsight Compute metrics to collect.
            passes: number of passes (1=minimal, >1 for deeper).
            advanced: whether advanced profiling is enabled.
        """
        self.metrics = metrics or [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        ]
        self.passes = passes
        self.advanced = advanced

    def evaluate(self, config, func=None, *args, **kwargs):
        """
        Run Nsight Compute profiling on a Python function if advanced mode is on.
        If advanced=False, return {} and let Agent rely on Executor timing.
        """
        if not self.advanced:
            print("[Profiler] ⚡ Skipping advanced profiling (using Executor timing only).")
            return {}

        if func is None:
            raise ValueError("Profiler requires a function to run in advanced mode.")

        # --- Step 1: Save function + args to pickle
        payload_file = tempfile.mktemp(suffix=".pkl")
        with open(payload_file, "wb") as f:
            cloudpickle.dump((func, args, kwargs), f)

        # --- Step 2: Create runner script
        runner_file = tempfile.mktemp(suffix=".py")
        with open(runner_file, "w") as f:
            f.write(
                "import cloudpickle\n"
                "func, args, kwargs = cloudpickle.load(open(r'{}','rb'))\n"
                "func(*args, **kwargs)\n".format(payload_file.replace("\\", "/"))
            )

        # --- Step 3: Build Nsight Compute command
        metrics_str = ",".join(self.metrics)
        report_file = tempfile.mktemp(suffix=".ncu-rep")

        # Configurable passes: use minimal first, then allow >1 if user wants deeper
        ncu_cmd = [
            "ncu",
            f"--metrics={metrics_str}",
            "--target-processes", "all",
            "--force-overwrite",
            "--set", "minimal" if self.passes == 1 else "full",
            "-o", report_file,
            "python", runner_file
        ]

        # If >1 passes requested
        if self.passes > 1:
            ncu_cmd += ["--launch-skip", "0", "--launch-count", str(self.passes)]

        print(f"[Profiler] Running: {' '.join(ncu_cmd)}")
        subprocess.run(ncu_cmd, check=True)

        # --- Step 4: Extract metrics
        extract_cmd = ["ncu", "--import", report_file, "--csv"]
        result = subprocess.run(extract_cmd, check=True, capture_output=True, text=True)

        scores = {}
        headers = None
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if headers is None:
                headers = parts
            else:
                if parts[0] in self.metrics:
                    try:
                        idx = headers.index("Metric Value")
                        scores[parts[0]] = float(parts[idx])
                    except Exception:
                        pass

        print("[Profiler] Extracted metrics:", scores)

        # --- Step 5: Cleanup
        try:
            os.remove(payload_file)
            os.remove(runner_file)
        except Exception:
            pass

        return scores
