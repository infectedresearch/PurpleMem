from benchmarks.metrics import dcg, ndcg_score


def test_dcg():
    assert dcg([1, 1], 2) > 0


def test_ndcg_perfect():
    assert ndcg_score(["a", "b"], {"a", "b"}, 2) == 1.0
