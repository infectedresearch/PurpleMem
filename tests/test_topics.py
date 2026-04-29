from purplemem.topics import detect_topic


def test_detect_food():
    assert detect_topic("alex likes sushi and ramen") == "food"


def test_detect_gaming():
    assert detect_topic("plays valorant every weekend") == "gaming"


def test_detect_general():
    assert detect_topic("hello there") == "general"
