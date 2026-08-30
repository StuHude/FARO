"""Audit text and path shortcuts in balanced existence schemas."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def label(row: dict) -> int:
    if "answer" in row:
        return int("no target" not in str(row["answer"]).lower())
    return int(not bool((row.get("meta") or {}).get("no_target", False)))


def features(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [f"w:{word}" for word in words] + [f"b:{a}_{b}" for a, b in zip(words, words[1:])]


def fit_nb(rows: list[dict]) -> tuple[dict[int, Counter], Counter, set[str]]:
    counts = {0: Counter(), 1: Counter()}
    classes = Counter()
    vocabulary: set[str] = set()
    for row in rows:
        y = label(row)
        token_counts = Counter(features(str(row.get("query", ""))))
        counts[y].update(token_counts)
        classes[y] += 1
        vocabulary.update(token_counts)
    return counts, classes, vocabulary


def predict_nb(model: tuple[dict[int, Counter], Counter, set[str]], row: dict) -> int:
    counts, classes, vocabulary = model
    total_rows = sum(classes.values())
    row_counts = Counter(features(str(row.get("query", ""))))
    scores = {}
    for y in (0, 1):
        denominator = sum(counts[y].values()) + len(vocabulary)
        score = math.log((classes[y] + 1) / (total_rows + 2))
        for token, frequency in row_counts.items():
            score += frequency * math.log((counts[y][token] + 1) / max(denominator, 1))
        scores[y] = score
    return max(scores, key=scores.get)


def grouped_majority_accuracy(train: list[dict], test: list[dict], key_fn) -> float:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        groups[key_fn(row)][label(row)] += 1
    global_label = Counter(label(row) for row in train).most_common(1)[0][0]
    predictions = [groups[key_fn(row)].most_common(1)[0][0] if groups[key_fn(row)] else global_label for row in test]
    return sum(pred == label(row) for pred, row in zip(predictions, test)) / max(len(test), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    train, test = load(args.train), load(args.test)
    nb = fit_nb(train)
    text_accuracy = sum(predict_nb(nb, row) == label(row) for row in test) / max(len(test), 1)
    parent_accuracy = grouped_majority_accuracy(train, test, lambda row: str(Path(row["image_path"]).parent))
    suffix_accuracy = grouped_majority_accuracy(train, test, lambda row: str("in this image" in str(row.get("query", "")).lower()))
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "query_nb_accuracy": text_accuracy,
        "parent_path_accuracy": parent_accuracy,
        "in_this_image_rule_accuracy": suffix_accuracy,
        "train_balance": sum(label(row) for row in train) / max(len(train), 1),
        "test_balance": sum(label(row) for row in test) / max(len(test), 1),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
