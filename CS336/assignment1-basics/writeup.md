### Unicode1
a) `\x00` The escape character. 
b) When printed, nothing is shown. `__repr__` gives `'\\x00'`
c) It is basically ignored. `this is a teststring`

### Unicode2
a) They require more bytes per character, leading to longer sequence length.
b) Non ascii characters use more than 1 byte per character. "こんにちは!"
c) `b'\xC3\x28'` This indicates a 2 byte character but the second byte does not start with `10`

### Train_bpe_tinystories