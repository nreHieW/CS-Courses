import math
import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.tokenizer import Tokenizer


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight = torch.randn(out_features, in_features, device=device, dtype=dtype)  # y = Wx
        sigma = (2 / (in_features + out_features)) ** 0.5
        weight = torch.nn.init.trunc_normal_(weight, 0, sigma**2, -3 * sigma, 3 * sigma)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight = torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        weight = torch.nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((1 / self.d_model * x.pow(2).sum(-1, keepdim=True) + self.eps))
        return ((x / rms) * self.weight).to(in_dtype)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swish(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swish(self.w1(x)))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # https://arxiv.org/pdf/2104.09864
        thetas = theta ** -((2 * torch.arange(0, d_k // 2)) / d_k)
        pos_ids = torch.arange(0, max_seq_len)  # m

        angles = einsum(thetas, pos_ids, "half_dim,  ctx_len -> ctx_len half_dim")
        self.register_buffer("sin_cos", torch.stack([torch.sin(angles), torch.cos(angles)]), persistent=False)  # [2, ctx-len, d/2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin_cos = self.sin_cos[:, token_positions, :]
        sin, cos = sin_cos[0], sin_cos[1]

        # eq 34 of https://arxiv.org/pdf/2104.09864 but 0 indexed
        x_reshaped = rearrange(x, "... (half_d even_odd) -> even_odd ... half_d", even_odd=2)  # this is equivalent to first reshape to (... half_d, even_odd) and then permute
        x_even_orig = x_reshaped[0]
        x_odd_orig = x_reshaped[1]

        x_even = x_even_orig * cos - x_odd_orig * sin
        x_odd = x_odd_orig * cos + x_even_orig * sin
        return rearrange([x_even, x_odd], "n ... d_half -> ... (d_half n)")  # equivalent to a permute and flatten


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    largest = torch.max(x)
    x_exp = torch.exp(x - largest)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None):
    attention_matrix = einsum(queries, keys, "... q_len dim, ... k_len dim -> ... q_len k_len") / math.sqrt(queries.shape[-1])
    if mask is not None:
        attention_matrix = torch.where(mask, attention_matrix, -torch.inf)

    attention_scores = softmax(attention_matrix, -1)
    return einsum(attention_scores, values, "... q_len k_len, ... k_len dim -> ... q_len dim")


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = rearrange(self.q_proj(x), "... seq_len (n_heads dim) -> ... n_heads seq_len dim", n_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq_len (n_heads dim) -> ... n_heads seq_len dim", n_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq_len (n_heads dim) -> ... n_heads seq_len dim", n_heads=self.num_heads)

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) - 1).bool()  # diagonal = 1 keeps the diagonal

        out = scaled_dot_product_attention(q, k, v, mask.to(x.device))
        out = rearrange(out, "... n_heads seq_len dim -> ... seq_len (n_heads dim)")
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, rope: RoPE | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = Attention(d_model, num_heads, rope, device, dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads, self.rope) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self, prompt: str, tokenizer: Tokenizer, temperature: float, top_p: float, max_new_tokens: int, return_prompt: bool = False) -> torch.Tensor:
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.token_embeddings.weight.device)
        prompt_len = tokens.shape[1]
        for _ in range(max_new_tokens):
            logits = self(tokens)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                masked_probs = torch.where(cumulative > top_p, 0, sorted_probs)
                masked_probs /= masked_probs.sum()  # softmax does not preserve 0
                next_token_idx = torch.multinomial(masked_probs, 1)
                next_token = torch.gather(sorted_indices, -1, next_token_idx)
            tokens = torch.cat([tokens, next_token], dim=-1)
        if return_prompt:
            return tokens
        return tokens[:, prompt_len:]
