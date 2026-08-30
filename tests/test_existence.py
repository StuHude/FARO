from projects.pixvl_idea3.existence import predicts_target_exists


def test_explicit_no_target_variants_are_abstentions():
    for text in ("No.", "No target.", "The target is absent.", "It does not exist.", "Not present"):
        assert not predicts_target_exists(text)


def test_positive_answers_are_not_abstentions():
    for text in ("Target exists.", "Yes.", "The person is present."):
        assert predicts_target_exists(text)
