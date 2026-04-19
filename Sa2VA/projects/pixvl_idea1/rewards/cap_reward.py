from __future__ import annotations

import re

from .text_similarity import SentenceSimilarityScorer, rouge_l_f1


SPECIAL_PATTERNS = [
    r"<\|im_end\|>",
    r"<\|end\|>",
]


def clean_caption_text(text: str) -> str:
    cleaned = text
    for pattern in SPECIAL_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    return " ".join(cleaned.strip().split())


def compute_cap_reward(
    prediction: str,
    reference: str,
    similarity_scorer: SentenceSimilarityScorer | None = None,
    semantic_weight: float = 0.6,
    rouge_weight: float = 0.4,
) -> float:
    pred = clean_caption_text(prediction)
    ref = clean_caption_text(reference)
    rouge = rouge_l_f1(pred, ref)
    if similarity_scorer is None:
        return rouge
    semantic = similarity_scorer.similarity(pred, ref)
    return semantic_weight * semantic + rouge_weight * rouge


def is_cap_failure(reward: float, threshold: float = 0.65) -> bool:
    return reward < threshold

