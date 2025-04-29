import torch
import math
import numpy as np
from typing import Iterable


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor):
    max_val = torch.max(inputs, dim=-1, keepdim=True)[0]
    inputs_exp = torch.exp(inputs - max_val)
    log_probs = (inputs - max_val) - torch.log(inputs_exp.sum(-1, keepdim=True))
    return -torch.gather(log_probs, -1, targets.unsqueeze(-1)).mean()


def get_lr(it: int, max_learning_rate: int, min_learning_rate: int, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = sum([(param.grad**2).sum() for param in parameters if param.grad is not None])
    total_norm = total_norm**0.5

    if total_norm >= max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad *= max_l2_norm / (total_norm + 1e-6)


def get_batch(x: np.array, batch_size: int, context_length: int, device: str):
    start = np.random.randint(low=0, high=x.shape[0] - context_length, size=(batch_size,)).reshape(-1, 1)
    idx = start + np.arange(context_length + 1).reshape(1, -1)
    examples = x[idx].reshape(batch_size, -1)
    examples = torch.from_numpy(examples.astype(np.int32)).to(device)
    examples = examples.long()
    examples = examples.view(batch_size, -1)
    return examples[:, :-1], examples[:, 1:]


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_states": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    saved = torch.load(src)
    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer_states"])
    return saved["iteration"]
