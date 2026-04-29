from purplemem.temporal import parse_temporal_reference


def test_recently():
    assert parse_temporal_reference("what happened recently?") == (3, 7)


def test_no_temporal():
    assert parse_temporal_reference("what does alex like?") is None
