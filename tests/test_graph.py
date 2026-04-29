from purplemem.graph import (
    EntityGraph,
    classify_entity,
    entity_id,
    extract_entities,
    extract_relationships,
)
from purplemem.storage.sqlite import connect


def test_entity_id():
    assert entity_id("Alex Smith") == "alex_smith"


def test_classify_company():
    assert classify_entity("google") == "company"


def test_extract_entities():
    ents = extract_entities("Alex plays Valorant", username="alex")
    names = {e["name"] for e in ents}
    assert "alex" in names
    assert "valorant" in names


def test_extract_relationships():
    rels = extract_relationships("alex works at google", username="alex")
    assert any(r["predicate"] == "works_at" for r in rels)


def test_entity_graph_roundtrip(tmp_path):
    conn = connect(str(tmp_path / "graph.sqlite"))
    graph = EntityGraph(conn)
    graph.process_text("alex works at google", username="alex", source_memory_id="m1")
    result = graph.query_entity("alex")
    assert result is not None
    assert result["name"] == "alex"
    assert any(item["predicate"] == "works_at" for item in result["outgoing"])
