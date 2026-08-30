from projects.pixvl_idea3.reward_ranking import RunningComponentRanker


def test_component_ranks_are_scale_independent():
    ranker = RunningComponentRanker(4, ("ciou", "boundary"))
    for ciou, boundary in ((0.1, 10.0), (0.2, 20.0), (0.3, 30.0)):
        ranker.update({"ciou": ciou, "boundary": boundary})

    ranks = ranker.rank({"ciou": 0.25, "boundary": 25.0})
    assert ranks == {"ciou": 2 / 3, "boundary": 2 / 3}


def test_rank_uses_midrank_for_ties_and_fifo_capacity():
    ranker = RunningComponentRanker(2, ("reward",))
    ranker.update({"reward": 1.0})
    ranker.update({"reward": 1.0})
    assert ranker.score({"reward": 1.0}, update=False) == 0.5
    ranker.update({"reward": 2.0})
    assert list(ranker.history["reward"]) == [1.0, 2.0]


def test_ranker_state_round_trip():
    ranker = RunningComponentRanker(3, ("a", "b"))
    ranker.update({"a": 0.2, "b": 0.8})
    restored = RunningComponentRanker(3, ("a", "b"))
    restored.load_state_dict(ranker.state_dict())
    assert restored.rank({"a": 0.3, "b": 0.7}) == {"a": 1.0, "b": 0.0}
