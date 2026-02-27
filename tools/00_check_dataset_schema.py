#!/usr/bin/env python3
import argparse, json, os
from collections import Counter, defaultdict

LABEL_CANDIDATES = [
    "label", "y", "target", "malware", "is_malware", "isMalware",
    "malicious", "is_malicious", "class", "family", "gt"
]
ID_CANDIDATES = ["sha256", "sha1", "md5", "hash", "id", "sample_id", "name"]

FEATURE_GROUP_HINTS = [
    "histogram", "byteentropy", "strings", "general", "header", "section",
    "imports", "exports", "datadirectories", "richheader", "signature",
    "warnings", "features", "raw_features"
]

def safe_type(v):
    if v is None: return "null"
    return type(v).__name__

def norm_label(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        if v in (0, 1):
            return int(v)
        if v in (-1, 1):
            return 1 if v == 1 else 0
        if v in (1, 2):
            return 1 if v == 2 else 0
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("0", "benign", "good", "clean", "false"):
            return 0
        if s in ("1", "malware", "malicious", "true"):
            return 1
    return None

def iter_jsonl(path, max_lines=None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                yield i, {"__JSON_DECODE_ERROR__": True}

def analyze_file(path, max_lines):
    key_count = Counter()
    type_by_key = defaultdict(Counter)
    label_presence = Counter()
    id_presence = Counter()
    feature_presence = Counter()

    label_values_norm = Counter()
    labels_norm_list = []

    lengths = defaultdict(Counter)
    has_bytez = 0
    has_raw_bytes_like = 0

    decode_errors = 0
    lines = 0

    for _, obj in iter_jsonl(path, max_lines=max_lines):
        lines += 1
        if "__JSON_DECODE_ERROR__" in obj:
            decode_errors += 1
            continue

        for k, v in obj.items():
            key_count[k] += 1
            type_by_key[k][safe_type(v)] += 1

        for lk in LABEL_CANDIDATES:
            if lk in obj:
                label_presence[lk] += 1
                nl = norm_label(obj[lk])
                if nl is not None:
                    label_values_norm[str(nl)] += 1
                    labels_norm_list.append(nl)

        for ik in ID_CANDIDATES:
            if ik in obj:
                id_presence[ik] += 1

        for fk in FEATURE_GROUP_HINTS:
            if fk in obj:
                feature_presence[fk] += 1

        for k in ("histogram", "byteentropy"):
            if k in obj and isinstance(obj[k], (list, tuple)):
                lengths[k][len(obj[k])] += 1

        if "bytez" in obj:
            has_bytez += 1
            v = obj["bytez"]
            if isinstance(v, str) and len(v) > 1000:
                has_raw_bytes_like += 1
            if isinstance(v, (bytes, bytearray)):
                has_raw_bytes_like += 1

    chosen_label_key = max(label_presence.items(), key=lambda x: x[1])[0] if label_presence else None
    chosen_id_key = max(id_presence.items(), key=lambda x: x[1])[0] if id_presence else None

    nlab = len(labels_norm_list)
    pos_rate = (sum(labels_norm_list) / nlab) if nlab else None

    return {
        "file": path,
        "lines_read": lines,
        "decode_errors": decode_errors,
        "top_level_keys_top25": key_count.most_common(25),
        "key_types_top25": {k: type_by_key[k].most_common(5) for k, _ in key_count.most_common(25)},
        "label_candidates_found": label_presence.most_common(),
        "chosen_label_key": chosen_label_key,
        "label_norm_counts": label_values_norm.most_common(10),
        "label_pos_rate": pos_rate,
        "id_candidates_found": id_presence.most_common(),
        "chosen_id_key": chosen_id_key,
        "feature_group_hints_found": feature_presence.most_common(),
        "lengths": {k: lengths[k].most_common(10) for k in lengths},
        "bytez_key_count": has_bytez,
        "raw_bytes_like_count": has_raw_bytes_like,
        "inference": {
            "has_raw_bytes": bool(has_raw_bytes_like > 0),
            "has_histogram": bool(key_count.get("histogram", 0) > 0),
            "has_byteentropy": bool(key_count.get("byteentropy", 0) > 0),
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--dataset_tag", default="EMBER2024_CORE_PE")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_lines", type=int, default=50000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_rep = analyze_file(args.train, args.max_lines)
    test_rep  = analyze_file(args.test, args.max_lines)

    out_train = os.path.join(args.outdir, f"{args.dataset_tag}_TRAIN_schema_report.json")
    out_test  = os.path.join(args.outdir, f"{args.dataset_tag}_TEST_schema_report.json")

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_rep, f, indent=2)
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test_rep, f, indent=2)

    print("=== TRAIN SUMMARY ===")
    print("chosen_label_key:", train_rep["chosen_label_key"])
    print("label_pos_rate:", train_rep["label_pos_rate"])
    print("chosen_id_key:", train_rep["chosen_id_key"])
    print("has_raw_bytes:", train_rep["inference"]["has_raw_bytes"])
    print("feature_hints_top10:", train_rep["feature_group_hints_found"][:10])
    print("lengths:", train_rep["lengths"])
    print("\n=== TEST SUMMARY ===")
    print("chosen_label_key:", test_rep["chosen_label_key"])
    print("label_pos_rate:", test_rep["label_pos_rate"])
    print("chosen_id_key:", test_rep["chosen_id_key"])
    print("has_raw_bytes:", test_rep["inference"]["has_raw_bytes"])
    print("feature_hints_top10:", test_rep["feature_group_hints_found"][:10])
    print("lengths:", test_rep["lengths"])
    print("\nWrote reports:")
    print(out_train)
    print(out_test)

if __name__ == "__main__":
    main()
