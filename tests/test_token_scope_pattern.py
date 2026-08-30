import re


def test_samtok_mask_token_pattern_matches_context_codes():
    pattern = re.compile(r"<\|mt_\d{4}\|>")
    assert pattern.fullmatch("<|mt_0001|>")
    assert pattern.fullmatch("<|mt_0256|>")
    assert not pattern.fullmatch("<|mt_start|>")
    assert not pattern.fullmatch("<|mt_001|>")
