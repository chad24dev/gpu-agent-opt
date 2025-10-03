class NsightProfiler:
    def __init__(self, metrics=None):
        self.metrics = metrics or [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        ]

    def evaluate_once(self):
        import tempfile, subprocess

        metrics_str = ",".join(self.metrics)
        report_file = tempfile.mktemp(suffix=".ncu-rep")

        cmd = [
            "ncu",
            f"--metrics={metrics_str}",
            "--target-processes", "all",
            "--force-overwrite",
            "--set", "full",
            "-o", report_file,
            "python", "kernel_entry.py"
        ]

        print(f"[Profiler] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

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
                    # Nsight CSV usually: Metric Name, Metric Value, ...
                    try:
                        idx = headers.index("Metric Value")
                        scores[parts[0]] = float(parts[idx])
                    except Exception:
                        pass

        print("[Profiler] Extracted metrics:", scores)
        return scores
