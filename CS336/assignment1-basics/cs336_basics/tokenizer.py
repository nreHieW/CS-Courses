import pickle
import regex as re
from typing import Iterable


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        # sort to capture the longer special tokens (for eg. ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        out = []
        if self.special_tokens:
            splitted_text = re.split("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")", text)  # () captures rather than discards the delimiters
        else:
            splitted_text = [text]
        for text in splitted_text:
            if self.special_tokens and (text in self.special_tokens):
                out.append(self.vocab_reverse[text.encode("utf-8")])
            elif text:
                pretokens = re.findall(PAT, text)
                for pretoken in pretokens:
                    pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    while True:
                        best_rank, best_index = float("inf"), None
                        for i in range(len(pretoken_bytes) - 1):
                            curr_pair = (pretoken_bytes[i], pretoken_bytes[i + 1])
                            curr_rank = self.merges.get(curr_pair, float("inf"))

                            if curr_rank < best_rank:
                                best_rank = curr_rank
                                best_index = i

                        if best_index is None:
                            break

                        pretoken_bytes = pretoken_bytes[:best_index] + (pretoken_bytes[best_index] + pretoken_bytes[best_index + 1],) + pretoken_bytes[best_index + 2 :]

                    out.extend([self.vocab_reverse[x] for x in pretoken_bytes])
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        def generator():
            for item in iterable:
                for token in self.encode(item):
                    yield token

        return generator()

    def decode(self, ids: list[int]) -> str:
        text = [self.vocab[x] for x in ids]
        text = b"".join(text)
        return text.decode("utf-8", errors="ignore")
