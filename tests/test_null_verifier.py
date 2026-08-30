from projects.pixvl_idea3.eval.eval_null_verifier import target_description


def test_target_description_removes_label_leaking_instructions():
    prompt = "Please segment elephant. If no described target exists, answer No target."
    assert target_description(prompt) == "elephant"


def test_target_description_keeps_content():
    assert target_description("Please segment a red cup on the left.") == "a red cup on the left"
