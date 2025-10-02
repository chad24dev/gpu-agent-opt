# gpu-agent-opt

**AI Agent Framework for GPU Kernel Autotuning & Optimization**

`gpu-agent-opt` is a Python package that brings **AI agents** into the loop for **CUDA kernel exploration, profiling, and optimization**.  
It automates the search for efficient GPU configurations (e.g. batch sizes, block dimensions, memory usage) using a feedback loop:

👉 **Sense → Think → Act → Learn**

---

## ✨ Features

- 🔍 **Autotune GPU Kernels**: Automatically test different kernel configurations (blockDim, gridDim, stride, etc.).  
- ⚡ **Profiler Integration**: Wraps around Nsight Compute / PyTorch profiler for real metrics (SM efficiency, DRAM utilization).  
- 🤖 **AI Agent Optimizer**: Bayesian search / RL loop for efficient exploration.  
- 📚 **Knowledge Base**: Stores past experiments and reuses best configs.  
- 🛰 **Use Cases**:
  - Geospatial AI pipelines (auto annotation, DINO/SAM embeddings).
  - Deep learning training & inference optimization.
  - HPC workloads (climate, genomics, simulations).
  - Edge AI (Jetson/embedded GPU batch-size tuning).

---

## 📦 Installation

From PyPI (coming soon):

```bash
pip install gpu-agent-opt
```

From source (development mode):

```bash
git clone https://github.com/yourusername/gpu_agent_opt.git
cd gpu_agent_project
pip install -e ./gpu_agent_opt
```

---

## 🚀 Quickstart

```python
from gpu_agent_opt import KernelAgent
from auto_annotate_dino_nvtx import extract_dino_features_all
import rasterio, numpy as np

# Load example image
with rasterio.open("satellite.tif") as src:
    img = np.stack([
        (src.read(b).astype(np.float32) - src.read(b).min()) /
        (src.read(b).ptp() + 1e-6) * 255
        for b in [3,2,1]
    ], -1).astype(np.uint8)

# Create agent
agent = KernelAgent(kernel_func=extract_dino_features_all)

# Autotune stride_frac and batch_size
best_cfg, results = agent.autotune(
    search_space={
        "stride_frac": [0.25, 0.5, 1.0],
        "batch_size": [128, 256, 512]
    },
    img=img
)

print("✅ Best Config Found:", best_cfg)
```

---

## 🆓 Free vs 💎 Premium

| Feature                                | Free (PyPI) | Premium (Enterprise) |
|----------------------------------------|-------------|-----------------------|
| Kernel tuning (grid/block search)      | ✅ Yes      | ✅ Yes                |
| Nsight Compute CLI integration         | ❌ No       | ✅ Yes                |
| Bayesian optimization                  | Basic       | Advanced w/ RL        |
| Knowledge base persistence (JSON)      | Local only  | Shared DB + dashboard |
| Supported platforms                    | RTX/desktop | + Jetson, A100, H100  |
| Commercial support & consulting        | ❌ No       | ✅ Available          |

---

## 📖 Documentation

Coming soon at [https://aifusion.in](https://aifusion.in) 🚀

---

## 🤝 Contributing

Contributions are welcome! Please open issues and PRs on  
[GitHub Repository](https://github.com/yourusername/gpu_agent_opt).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.