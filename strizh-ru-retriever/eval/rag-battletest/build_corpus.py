#!/usr/bin/env python3
"""Build a dogfood RAG corpus from a source repository.

Chunks Markdown docs, service/Ansible YAML (config + comments) and Python source into
~500-token passages, tagged with provenance and a coarse stratum (RU prose / RU+config /
code). We ran it over the AGmind repo itself — a leak-free corpus (no overlap with the
model's MIRACL/FRIDA training) that is genuinely mixed Russian prose + code + English
identifiers, i.e. exactly the "developer asks the RAG about our stack" workload.

Usage:  python build_corpus.py [REPO_ROOT]   (default: current dir)
Output: $DATA_DIR/chunks.jsonl                (default: ./data)

The directory layout below (docs/, templates/services/, ansible/roles/, a Python
package) is tailored to the AGmind repo; adapt `collect()` to your own tree.
"""
import os, re, json, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
DATA = os.environ.get("DATA_DIR", "./data")
OUT = os.path.join(DATA, "chunks.jsonl")

CYR = re.compile(r"[а-яА-ЯёЁ]")
def cyr_frac(t):
    letters = [c for c in t if c.isalpha()]
    if not letters: return 0.0
    return sum(1 for c in letters if CYR.match(c)) / len(letters)

def approx_tok(t):  # RuModernBERT ~3.5 chars/token
    return int(len(t) / 3.5)

def recursive_chunks(text, target_chars, overlap):
    """Recursive split: on blank lines first, accumulate to target, char overlap."""
    paras = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for p in paras:
        p = p.rstrip()
        if not p: continue
        if len(buf) + len(p) + 2 <= target_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf: chunks.append(buf)
            if len(p) > target_chars:
                lines, lb = p.split("\n"), ""
                for ln in lines:
                    if len(lb) + len(ln) + 1 <= target_chars:
                        lb = (lb + "\n" + ln) if lb else ln
                    else:
                        if lb: chunks.append(lb)
                        lb = ln
                buf = lb
            else:
                buf = p
    if buf: chunks.append(buf)
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            out.append(chunks[i-1][-overlap:] + "\n" + chunks[i])
        chunks = out
    return chunks

def collect():
    files = []
    for f in ["README.ru.md", "README.md"]:
        p = os.path.join(REPO, f)
        if os.path.exists(p): files.append((p, "readme", "ru_prose", 2200, 200))
    for root, _, fs in os.walk(os.path.join(REPO, "docs")):
        for f in fs:
            if f.endswith(".md"): files.append((os.path.join(root, f), "docs", "ru_prose", 2200, 200))
    svc = os.path.join(REPO, "templates/services")
    if os.path.isdir(svc):
        for f in sorted(os.listdir(svc)):
            if f.endswith(".yaml"): files.append((os.path.join(svc, f), "svc_yaml", "ru_code", 2400, 150))
    for pkg in ("agmind",):
        for root, _, fs in os.walk(os.path.join(REPO, pkg)):
            if "__pycache__" in root or "/test" in root: continue
            for f in fs:
                if f.endswith(".py") and "test" not in f:
                    files.append((os.path.join(root, f), "py_code", "code", 2600, 150))
    for root, _, fs in os.walk(os.path.join(REPO, "ansible")):
        if any(x in root for x in ("/molecule", "/.git")): continue
        if os.path.basename(root) in ("tasks", "defaults", "vars", "handlers", "templates"):
            for f in fs:
                if f.endswith((".yml", ".yaml")):
                    files.append((os.path.join(root, f), "ansible", "ru_code", 2400, 150))
    return files

def main():
    os.makedirs(DATA, exist_ok=True)
    rows, cid, seen = [], 0, set()
    for path, src, stratum, tgt, ov in collect():
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue
        rel = os.path.relpath(path, REPO)
        for ch in recursive_chunks(text, tgt, ov):
            ch = ch.strip()
            if len(ch) < 120: continue
            if stratum == "ru_prose" and cyr_frac(ch) < 0.15: continue
            key = ch[:200]
            if key in seen: continue
            seen.add(key)
            toks = approx_tok(ch)
            rows.append({"id": f"c{cid}", "text": ch, "source": rel, "src_type": src,
                         "stratum_hint": stratum, "approx_tok": toks,
                         "is_long": toks > 400, "cyr_frac": round(cyr_frac(ch), 2)})
            cid += 1
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"CHUNKS: {len(rows)} -> {OUT}")
    print("  by src_type:", dict(Counter(r["src_type"] for r in rows)))
    print("  by stratum :", dict(Counter(r["stratum_hint"] for r in rows)))

if __name__ == "__main__":
    main()
