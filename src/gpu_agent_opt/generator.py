# gpu_agent_opt/generator.py
class KernelGenerator:
    """
    Generates kernel variants (different configs).
    For now, just holds configs. In future, expand to generate CUDA/Triton code.
    """
    def __init__(self, template=None):
        self.template = template

    def generate(self, **kwargs):
        return kwargs

    def compile(self, variant):
        # Stub: return variant unchanged
        return variant
