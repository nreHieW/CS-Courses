from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float], weight_decay: float, eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data

                m = self.betas[0] * m + (1 - self.betas[0]) * grad
                v = self.betas[1] * v + (1 - self.betas[1]) * grad**2
                alpha_t = lr * ((math.sqrt(1 - self.betas[1] ** t)) / (1 - self.betas[0] ** t))

                p.data -= alpha_t * (m / (torch.sqrt(v) + self.eps))

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

                p.data -= p.data * self.weight_decay * lr

        return loss
