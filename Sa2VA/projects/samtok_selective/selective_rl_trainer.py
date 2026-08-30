from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V7Decision:
    accepted: bool
    null_risk: float
    null_budget: float
    anchor_kl: float
    kl_budget: float


def accept_v7_step(
    *, null_risk: float, null_budget: float, anchor_kl: float, kl_budget: float
) -> V7Decision:
    """Registered reject-only V7 feasibility decision used after SFT reproduction."""
    values = (null_risk, null_budget, anchor_kl, kl_budget)
    if any(value < 0.0 for value in values):
        raise ValueError("V7 risk and KL values must be nonnegative")
    return V7Decision(
        accepted=null_risk <= null_budget and anchor_kl <= kl_budget,
        null_risk=float(null_risk),
        null_budget=float(null_budget),
        anchor_kl=float(anchor_kl),
        kl_budget=float(kl_budget),
    )


def main() -> None:
    raise RuntimeError(
        "Formal V7 is gated on standalone first-stage and continued-SFT reproduction; "
        "this module intentionally exposes only the preregistered feasibility decision."
    )


if __name__ == "__main__":
    main()
