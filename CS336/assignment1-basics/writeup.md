### Unicode1
a) `\x00` The escape character. 
b) When printed, nothing is shown. `__repr__` gives `'\\x00'`
c) It is basically ignored. `this is a teststring`

### Unicode2
a) They require more bytes per character, leading to longer sequence length.
b) Non ascii characters use more than 1 byte per character. "こんにちは!"
c) `b'\xC3\x28'` This indicates a 2 byte character but the second byte does not start with `10`

### Train_bpe_tinystories
a) On a M1 Mac, it took around 7 minutes with max memory usage of around 4GB
b) The merging logic took the most time. Command ran `uv run scalene cs336_basics/train_bpe.py --vocab_size 10000 --input_path data/TinyStoriesV2-GPT4-valid.txt --output_path "tinystories_bpe.pkl" --html`

### Train_bpe_expts_owt

b) The longest token in the TinyStories tokenizer is `' accomplishment'`


### Learning_rate_tuning
It diverges at 1e3 and fails to change at 1e2



