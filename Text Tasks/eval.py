import json
from pathlib import Path
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RESULTS_FILE = "llm_text_role_results_ollama.json"
# ──────────────────────────────────────────────────────────────────────────────



with open(RESULTS_FILE) as f:
    data = json.load(f)



classes = sorted({r["actual"] for r in data})



tp = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)



for r in data:
    a = r["actual"]
    p = r["predicted"]
    if p == a:
        tp[a] += 1
    else:
        fp[p] += 1
        fn[a] += 1



print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
print("-" * 70)




for c in classes:
    denom_p = tp[c] + fp[c]
    denom_r = tp[c] + fn[c]
    p = tp[c] / denom_p if denom_p else 0.0
    r = tp[c] / denom_r if denom_r else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"{c:<30} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {tp[c] + fn[c]:>8}")

# Macro averages
macro_p = sum(tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0 for c in classes) / len(classes)
macro_r = sum(tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0 for c in classes) / len(classes)
macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0

# Micro averages = global accuracy
total_correct = sum(tp.values())
total = len(data)
micro_p = micro_r = micro_f1 = total_correct / total if total else 0.0

print("-" * 70)
print(f"{'Macro avg':<30} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total:>8}")
print(f"{'Micro avg / Accuracy':<30} {micro_p:>10.4f} {micro_r:>10.4f} {micro_f1:>10.4f} {total:>8}")
