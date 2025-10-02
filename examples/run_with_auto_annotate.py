# gpu_agent_opt/examples/run_with_auto_annotate.py

from gpu_agent_opt import KernelAgent
from auto_annotate_dino_nvtx import extract_dino_features_all
import rasterio
import torch


def main(image_path):
    # ---- Load 3 bands with rasterio ----
    with rasterio.open(image_path) as src:
        bands = [src.read(b).astype("float32") for b in [3, 2, 1]]

    # ---- Convert to Torch tensor (H, W, 3) ----
    img = torch.from_numpy(
        __import__("numpy").stack(bands, axis=-1)  # stacked H,W,3
    )

    # ---- Normalize each channel (per band min/max) ----
    min_vals = img.amin(dim=(0, 1), keepdim=True)
    max_vals = img.amax(dim=(0, 1), keepdim=True)
    img = (img - min_vals) / (max_vals - min_vals + 1e-6) * 255.0

    # ---- Convert to uint8 for downstream processing ----
    img = img.to(torch.uint8)

    # If you want everything on GPU already:
    # img = img.cuda()

    # ---- Run KernelAgent autotuning ----
    agent = KernelAgent(
        kernel_func=extract_dino_features_all,
        metrics=["sm_efficiency", "dram_utilization"],
        gpu="RTX3060"
    )

    best_cfg, results = agent.autotune(
        search_space={
            "stride_frac": [0.25, 0.5, 1.0],
            "batch_size": [16, 32, 64]  # safer for DINOv2
        },
        img=img.numpy()  # convert back to numpy if your kernel expects numpy
    )

    print("✅ Best Config:", best_cfg)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = ap.parse_args()
    main(args.image)
