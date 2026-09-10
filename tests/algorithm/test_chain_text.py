from server.algorithm.models import Chain, EdgeRecord, format_chain_text
from server.tools.source_registry import SourceRegistry
from server.tools.subgraph_search import _format_accepted_chains


def test_new_chain_preserves_raw_relation_and_evidence():
    quote = 'COMPOSED_OF\nUNIT 9\nA —PRODUCES→ B'
    edge = EdgeRecord('e', 'e', 'COMPOSED_OF', 'a', 'b', 'A', 'B', evidence=quote)
    text = Chain('a1', ['e'], 1, edges=[edge]).format_unit()
    assert text.startswith('Chain a1\nA —composed of→ B')
    assert quote in text
    assert edge.to_dict_edge()['type'] == 'COMPOSED_OF'


def test_legacy_text_only_changes_headers_and_relations():
    quote = '"COMPOSED_OF\nUNIT 9\nA —PRODUCES→ B\nend"  (paper.pdf; conf=1.00)'
    old = f'UNIT U1:\nUNIT a1\n@A  B —COMPOSED_OF→ A\n  {quote}'
    expected = f'Chain [7]\n@A  B —composed of→ A\n  {quote}'
    assert format_chain_text(old, '[7]') == expected
    assert format_chain_text(expected, '[7]') == expected


def test_batches_renumber_and_remap_sources_without_duplicate_headers():
    registry = SourceRegistry()
    item = {'unit_no': 3, 'text': 'UNIT a1\nA —PRODUCES→ B\n  "data"  (paper.pdf; conf=1.00)', 'edges': [{'source_file': 'paper.pdf'}]}
    first = _format_accepted_chains([item], registry)
    assert 'UNIT' not in first
    assert '\nChain [3]\nA —produces→ B' in first
    assert '(source:1; conf=1.00)' in first
    assert 'this batch is Chain [3]–[3]' in first
    second = _format_accepted_chains([{**item, 'unit_no': 4}], registry)
    assert '\nChain [4]\n' in second
    assert '(source:1; conf=1.00)' in second
    assert item['text'].startswith('UNIT a1')
