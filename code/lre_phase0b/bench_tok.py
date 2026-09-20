import time, glob, sys
import pyarrow.parquet as pq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

COLS = ["task_prompt_text","prefix_action_text","prefix_feedback_text",
        "last_action_text","last_feedback_text","prefix_thought_text","last_thought_text"]
parts = sorted(glob.glob("work/lre_phase0b/prefix_table/prefix_table.part-*.parquet"))
pf = pq.ParquetFile(parts[0])
tbl = pf.read(columns=COLS)
print("sample rows", tbl.num_rows)
for c in COLS:
    vals = ["" if v is None else v for v in tbl.column(c).to_pylist()]
    la = np.array([len(v) for v in vals], dtype=np.float64)
    print(f"{c:26s} mean={la.mean():10.1f} total_MB={la.sum()/1e6:9.2f}")
print()
for c in COLS:
    vals = ["" if v is None else v for v in tbl.column(c).to_pylist()]
    vec = TfidfVectorizer(ngram_range=(1,2), dtype=np.float32)
    an = vec.build_analyzer()
    t0=time.time()
    toks=[len(an(t)) for t in vals]
    dt=time.time()-t0
    print(f"{c:26s} tok_time_per_1k_rows={dt/len(vals)*1000:8.3f}s  mean_tokens={np.mean(toks):8.0f}")
