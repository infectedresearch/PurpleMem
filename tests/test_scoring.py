from purplemem.scoring import SearchConfig, lexical_match_bonus


def test_default_search_config():
    cfg = SearchConfig()
    assert cfg.lexical_mode == "multiplicative"
    assert cfg.topic_boost == 0.0


def test_lexical_match_bonus_exact():
    assert lexical_match_bonus("alex likes sushi", ["likes sushi"]) == 0.25
