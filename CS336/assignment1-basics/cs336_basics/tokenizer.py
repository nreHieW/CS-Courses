import pickle
import regex as re
from typing import Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        # sort to capture the longer special tokens (for eg. ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, "r") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "r") as f:
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
                pretoken_bytes = [tuple([x.to_bytes() for x in token.encode("utf-8")]) for token in pretokens]
                for pretoken_idx in range(len(pretoken_bytes)):
                    pretoken = pretoken_bytes[pretoken_idx]
                    while len(pretoken) > 1:
                        change = False
                        for merge_a, merge_b in self.merges:
                            i = 0
                            while i < len(pretoken):
                                if i + 1 < len(pretoken) and pretoken[i] == merge_a and pretoken[i + 1] == merge_b:
                                    pretoken = pretoken[:i] + tuple([b"".join((merge_a, merge_b))]) + pretoken[i + 2 :]
                                    change = True
                                i += 1
                        if not change:
                            break

                    for item in pretoken:
                        out.append(self.vocab_reverse[item])
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
