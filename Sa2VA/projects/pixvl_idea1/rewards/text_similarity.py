from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from rouge_score import rouge_scorer


@lru_cache(maxsize=4)
def _build_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_f1(prediction: str, reference: str) -> float:
    scorer = _build_scorer()
    return float(scorer.score(reference, prediction)["rougeL"].fmeasure)


class SentenceSimilarityScorer:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or os.environ.get("PIXVL_TEXT_SIM_DEVICE", "cpu")
        self.local_files_only = os.environ.get("PIXVL_TEXT_SIM_LOCAL_ONLY", "1") != "0"
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        return self._model

    def similarity(self, prediction: str, reference: str) -> float:
        try:
            model = self._ensure_model()
        except Exception:
            return 0.0
        emb = model.encode([prediction, reference], normalize_embeddings=True)
        return float(np.clip((emb[0] @ emb[1] + 1.0) / 2.0, 0.0, 1.0))
