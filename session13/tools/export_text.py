"""Export the first N characters' worth of documents from the FineWeb-Edu parquet
shard to JSONL (one {"text": ...} per line). Used locally when the Python env that
has tiktoken lacks pyarrow; see prepare_data.py --from-jsonl."""
import json, sys
import pyarrow.parquet as pq
src, dst, max_chars = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = docs = 0
with open(dst, "w") as f:
    for rb in pq.ParquetFile(src).iter_batches(batch_size=4096, columns=["text"]):
        for t in rb.column(0).to_pylist():
            f.write(json.dumps({"text": t}) + "\n"); n += len(t); docs += 1
        if n >= max_chars: break
print(f"exported {docs} docs, {n/1e6:.0f}M chars")
