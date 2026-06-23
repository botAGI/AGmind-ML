import os
p="/home/gamer/ru-splitter/llama.cpp/conversion/base.py"
H="e9b7dbd66e0308c6e89983d5b6e1ca047106d862879a0fd33a12c8491b91ec5c"
s=open(p).read()
anchor='        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":'
block='        if chkhsh == "'+H+'":\n            res = "qwen2"  # T-lite-it-2.1 (Qwen3 + ext RU vocab)\n'
if H in s:
    print("already patched")
elif anchor in s:
    open(p,"w").write(s.replace(anchor, block+anchor, 1))
    print("PATCHED OK" if H in open(p).read() else "FAIL")
else:
    print("ANCHOR NOT FOUND")
