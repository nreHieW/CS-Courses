import os
import time
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)
NUM_PROCESSES = os.cpu_count()


def pretokenize(path, start, end, special_tokens) -> Counter:
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    splitted_text = re.split("|".join(re.escape(t) for t in special_tokens), text)
    counts = Counter()
    for text in splitted_text:
        for token_match in PAT.finditer(text):
            token = token_match.group(0)
            byte_list = [x.to_bytes() for x in list(token.encode("utf-8"))]
            counts[tuple(byte_list)] += 1
    return counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, "<|endoftext|>".encode("utf-8"))

    start_ends = list(zip(boundaries[:-1], boundaries[1:]))
    with Pool(processes=NUM_PROCESSES) as pool:
        pretokenize_fn = partial(pretokenize, input_path, special_tokens=special_tokens)
        results: list[dict[tuple[bytes], int]] = pool.starmap(pretokenize_fn, [(start, end) for start, end in start_ends])

    combined_pretokens = sum(results, Counter())
    pair_counts: dict[tuple[bytes], int] = defaultdict(int)
    for pretoken_bytes, v in combined_pretokens.items():
        for byte1, byte2 in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
            pair_counts[(byte1, byte2)] += v

    while len(vocab) < vocab_size:
        pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        a, b = pair
        new_token = a + b
        merges.append(pair)
        vocab[len(vocab)] = new_token
        pair_counts.pop(pair)

        new_pretokens = defaultdict(int)
        for pretoken_bytes, v in combined_pretokens.items():
            new_pretoken = []
            i = 0
            while i < len(pretoken_bytes):
                if i + 1 < len(pretoken_bytes) and pretoken_bytes[i] == a and pretoken_bytes[i + 1] == b:
                    new_pretoken.append(new_token)

                    if i > 0:
                        before = pretoken_bytes[i - 1]
                        pair_counts[(before, a)] -= v
                        pair_counts[(before, new_token)] += v

                    if i < len(pretoken_bytes) - 2:
                        after = pretoken_bytes[i + 2]
                        pair_counts[(b, after)] -= v
                        pair_counts[(new_token, after)] += v
                    i += 2

                else:
                    new_pretoken.append(pretoken_bytes[i])
                    i += 1
            new_pretokens[tuple(new_pretoken)] += v
        combined_pretokens = new_pretokens

    return vocab, merges


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input_path", type=str, help="Path to the input file")
    parser.add_argument("--vocab_size", type=int, help="Size of the vocabulary")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    start = time.time()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, ["<|endoftext|>"])
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")
    pickle.dump(vocab, open(os.path.join(args.output_dir, "vocab.pkl"), "wb"))
    pickle.dump(merges, open(os.path.join(args.output_dir, "merges.pkl"), "wb"))

# !uv run cs336_basics/train_bpe.py --vocab_size 32000 --input_path /content/CS-Courses/CS336/assignment1-basics/data/owt_train.txt --output_path /content/CS-Courses/CS336/assignment1-basics/data/owt_bpe.pkl
