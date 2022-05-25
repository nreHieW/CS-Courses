import re

fname = input("Filename: ")
fhandle = open(fname)
num = list()
for line in fhandle:
    line = line.rstrip()
    num = num + re.findall('[0-9]+',line)
num = [int(i) for i in num]
print(sum(num))
