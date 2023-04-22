import torch


class TransformerWarmup:
    """follow https://arxiv.org/abs/1706.03762"""

    def __init__(self, scalar: float=1., d_model: int=768, warmup_steps: int=4000):
        self._scalar = torch.tensor(scalar)
        self._d_model = torch.tensor(d_model)
        self._warmup_steps = torch.tensor(warmup_steps)

    def __call__(self, step: int):
        step = torch.tensor(step)
        arg1 = torch.rsqrt(step)
        arg2 = step * (self._warmup_steps ** -1.5)
        return self._scalar * torch.rsqrt(self._d_model) * torch.minimum(arg1, arg2)