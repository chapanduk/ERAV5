"""Download one FineWeb-Edu shard and tokenize a ~51.5M-token slice with GPT-2 BPE.

Output (uint16 token ids, documents separated by <|endoftext|>):
  data/val.bin    first  VAL_TOKENS  tokens   (held out)
  data/train.bin  next   TRAIN_TOKENS tokens  (each token is trained on exactly once)

Usage:  python prepare_data.py [--out data] [--train 50500000] [--val 1000000]
"""
import argparse
import os
import urllib.request

import json

import numpy as np
import tiktoken

URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"


def download(url, path):
    if os.path.exists(path):
        return
    tmp = path + ".part"
    print(f"downloading {url}")

    def hook(n, bs, total):
        if n % 200 == 0:
            print(f"  {n*bs/1e9:.2f}/{total/1e9:.2f} GB", flush=True)

    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=50_500_000)
    ap.add_argument("--val", type=int, default=1_000_000)
    ap.add_argument("--from-jsonl", default=None,
                    help="read documents from a JSONL text export instead of the parquet shard "
                         "(for environments without pyarrow; see tools/export_text.py)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.from_jsonl:
        def batches():
            with open(args.from_jsonl) as f:
                chunk = []
                for line in f:
                    chunk.append(json.loads(line)["text"])
                    if len(chunk) == 2048:
                        yield chunk; chunk = []
                if chunk:
                    yield chunk
    else:
        import pyarrow.parquet as pq
        shard = os.path.join(args.out, "fineweb_edu_10BT_000_00000.parquet")
        download(URL, shard)

        def batches():
            for rb in pq.ParquetFile(shard).iter_batches(batch_size=2048, columns=["text"]):
                yield rb.column(0).to_pylist()

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    need = args.val + args.train
    buf = np.empty(need, dtype=np.uint16)
    n, docs = 0, 0
    for texts in batches():
        for ids in enc.encode_ordinary_batch(texts, num_threads=8):
            ids.append(eot)
            take = min(len(ids), need - n)
            buf[n : n + take] = ids[:take]
            n += take
            docs += 1
            if n >= need:
                break
        print(f"  tokenized {n/1e6:.1f}M tokens from {docs} docs", flush=True)
        if n >= need:
            break
    assert n == need, f"shard too small: {n} < {need}"
    buf[: args.val].tofile(os.path.join(args.out, "val.bin"))
    buf[args.val :].tofile(os.path.join(args.out, "train.bin"))
    print(f"wrote {args.val} val + {args.train} train tokens from {docs} documents to {args.out}/")


if __name__ == "__main__":
    main()
