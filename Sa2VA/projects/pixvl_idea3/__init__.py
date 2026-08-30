"""Idea3 exports.

Keep imports lazy so lightweight routing diagnostics do not require the full
training stack (Torch, pycocotools, and sentence-transformers).
"""

__all__ = [
    "annotate_record",
    "build_relation_confuser_map",
    "compute_geometry_reward",
    "compute_relation_caption_reward",
    "compute_relation_reward",
    "compute_semantic_caption_reward",
    "derive_slice_tags",
    "infer_failure_route",
]


def __getattr__(name: str):
    if name in __all__:
        from . import routing

        return getattr(routing, name)
    raise AttributeError(name)
