from tools.analyze_official_grefcoco_pair import decode_compressed_counts, rle_pair_stats


def rle(counts: str) -> dict:
    return {"size": [2, 2], "counts": counts}


def test_decode_compressed_counts_handles_coco_delta_runs():
    assert decode_compressed_counts("1110") == [1, 1, 1, 1]


def test_rle_pair_stats_empty_full_and_alternating_masks():
    assert rle_pair_stats(rle("4"), rle("04")) == (0, 4, 0, 4)
    assert rle_pair_stats(rle("1110"), rle("1110")) == (2, 2, 2, 2)
    assert rle_pair_stats(rle("1110"), rle("022")) == (2, 2, 1, 3)
