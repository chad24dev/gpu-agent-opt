# gpu_agent_opt/examples/run_with_auto_annotate.py

from src.gpu_agent_opt import KernelAgent
from auto_annotate_dino_nvtx import extract_dino_features_all
import rasterio
import numpy as np


def main(image_path):
    with rasterio.open(image_path) as src:
        img = np.stack([(src.read(b).astype(np.float32) - src.read(b).min()) /
                        (src.read(b).ptp() + 1e-6) * 255
                        for b in [3, 2, 1]], -1).astype(np.uint8)

    agent = KernelAgent(
        kernel_func=extract_dino_features_all,
        metrics=["sm_efficiency", "dram_utilization"],
        gpu="RTX3060"
    )

    best_cfg, results = agent.autotune(
        search_space={
            "stride_frac": [0.25, 0.5, 1.0],
            "batch_size": [128, 256, 512]
        },
        img=img
    )

    print("✅ Best Config:", best_cfg)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = ap.parse_args()
    main(args.image)
