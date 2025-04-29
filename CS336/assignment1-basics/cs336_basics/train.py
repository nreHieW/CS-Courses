import argparse
import torch
import numpy as np
import os
import time
import wandb
from dataclasses import dataclass
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import clip_gradients, cross_entropy_loss, get_batch, get_lr, load_checkpoint, save_checkpoint
from cs336_basics.optimizer import AdamW
from cs336_basics.layers import TransformerLM
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


@dataclass
class TrainingArguments:
    tokenizer_merges_path: str
    tokenizer_vocab_path: str
    train_data_path: str
    val_data_path: str

    batch_size: int = 1
    context_length: int = 12
    max_iters: int | None = None  # Changed to be optional
    target_tokens: int = 327_680_000  # ~327M tokens as default target

    vocab_size: int = 10000
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

    grad_clip: float = 1.0
    warmup_iters: int = 200
    cosine_cycle_iters: int = 10000
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    val_every: int = 1000
    val_batches: int = 10

    project_name: str = "cs336"


def parse_args() -> TrainingArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_merges_path", type=str, required=True)
    parser.add_argument("--tokenizer_vocab_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
    )
    parser.add_argument("--max_learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batches", type=int, default=10)
    args = parser.parse_args()
    return TrainingArguments(**vars(args))


def tokenize_data(tokenizer: Tokenizer, train_data_path: str, val_data_path: str) -> tuple[str, str]:
    print("Tokenizing data...")
    with open(train_data_path, "r") as f:
        train_data = f.read()
        filesize = os.path.getsize(train_data_path)
    with open(val_data_path, "r") as f:
        val_data = f.read()
    start = time.time()
    train_tokens = tokenizer.encode(train_data)
    end = time.time()
    val_tokens = tokenizer.encode(val_data)
    print(f"Tokenization took {end - start:.2f} seconds")
    print(f"Bytes / second: {filesize / (end - start):.2f}")

    train_tokens = np.array(train_tokens, dtype=np.uint16)
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    print(f"Saving tokenized data to {train_data_path.replace('.txt', '_tokens.npy')} and {val_data_path.replace('.txt', '_tokens.npy')}")
    np.save(train_data_path.replace(".txt", "_tokens.npy"), train_tokens)
    np.save(val_data_path.replace(".txt", "_tokens.npy"), val_tokens)
    return train_data_path.replace(".txt", "_tokens.npy"), val_data_path.replace(".txt", "_tokens.npy")


def main():
    args = parse_args()
    print("Arguments:", args)

    wandb.init(project=args.project_name, config=vars(args))

    tokenizer: Tokenizer = Tokenizer.from_files(args.tokenizer_vocab_path, args.tokenizer_merges_path, special_tokens="<|endoftext|>")
    train_tokens_path = args.train_data_path.replace(".txt", "_tokens.npy")
    val_tokens_path = args.val_data_path.replace(".txt", "_tokens.npy")
    if not (os.path.exists(train_tokens_path) or os.path.exists(val_tokens_path)):
        train_tokens_path, val_tokens_path = tokenize_data(tokenizer, args.train_data_path, args.val_data_path)
    train_data = np.load(train_tokens_path, mmap_mode="r")
    val_data = np.load(val_tokens_path, mmap_mode="r")

    # Calculate max_iters if not provided
    if args.max_iters is None:
        tokens_per_iter = args.batch_size * args.context_length
        args.max_iters = args.target_tokens // tokens_per_iter
        print(f"Automatically calculated max_iters={args.max_iters} to reach target {args.target_tokens:,} tokens")

    total_tokens = args.batch_size * args.context_length * args.max_iters
    print(f"Total Tokens to be Processed: {total_tokens:,}")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(DEVICE)
    if DEVICE.type == "cuda":
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.min_learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    best_val_loss = float("inf")
    if os.path.exists("checkpoint.pth"):
        iteration = load_checkpoint("checkpoint.pth", model, optimizer)
    else:
        iteration = 1

    for i in tqdm(range(iteration, args.max_iters + 1)):
        x, y = get_batch(train_data, args.batch_size, args.context_length, DEVICE)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        lr = get_lr(i, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad()

        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()

        wandb.log({"train_loss": loss.item(), "learning_rate": lr, "iteration": i})

        if i % args.val_every == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for _ in range(args.val_batches):
                    val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, DEVICE)
                    val_logits = model(val_x)
                    val_loss = cross_entropy_loss(val_logits, val_y)
                    total_val_loss += val_loss.item()
            val_loss = total_val_loss / args.val_batches
            model.train()

            print(f"Iteration {i}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {lr:.6f}")
            wandb.log({"val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint("best_model.pth", model, optimizer, i)

            save_checkpoint(f"checkpoint_{i}.pth", model, optimizer, i)

    test_gen = model.generate("Once upon a time,", tokenizer, temperature=0.8, top_p=0.9, max_new_tokens=256)
    generated_text = tokenizer.decode(test_gen[0].tolist())
    print("Generated text:", generated_text)
    wandb.log({"final_generation": generated_text})
    wandb.finish()


if __name__ == "__main__":
    main()
