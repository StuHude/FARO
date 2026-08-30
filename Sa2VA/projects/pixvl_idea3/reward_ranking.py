"""Rolling per-component empirical ranks for matched reward controls."""

from __future__ import annotations

from collections import deque
from typing import Mapping


class RunningComponentRanker:
    """Map reward components to FIFO empirical CDF scores in ``[0, 1]``."""

    def __init__(self, capacity: int, components: tuple[str, ...]) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not components:
            raise ValueError("components must be non-empty")
        self.capacity = int(capacity)
        self.components = tuple(components)
        self.history = {
            name: deque(maxlen=self.capacity) for name in self.components
        }

    def rank(self, values: Mapping[str, float]) -> dict[str, float]:
        ranks: dict[str, float] = {}
        for name in self.components:
            value = float(values[name])
            prior = self.history[name]
            if not prior:
                ranks[name] = 0.5
            else:
                less = sum(item < value for item in prior)
                equal = sum(item == value for item in prior)
                ranks[name] = (less + 0.5 * equal) / len(prior)
        return ranks

    def update(self, values: Mapping[str, float]) -> None:
        for name in self.components:
            self.history[name].append(float(values[name]))

    def score(self, values: Mapping[str, float], *, update: bool = True) -> float:
        ranks = self.rank(values)
        if update:
            self.update(values)
        return sum(ranks.values()) / len(ranks)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "components": list(self.components),
            "history": {
                name: list(self.history[name]) for name in self.components
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("ranker capacity mismatch")
        if tuple(state.get("components", ())) != self.components:
            raise ValueError("ranker component mismatch")
        history = state.get("history", {})
        if not isinstance(history, Mapping):
            raise TypeError("ranker history must be a mapping")
        for name in self.components:
            values = history.get(name, ())
            self.history[name].clear()
            self.history[name].extend(float(value) for value in values)
