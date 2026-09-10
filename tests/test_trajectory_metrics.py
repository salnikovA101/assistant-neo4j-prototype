from tests.trajectory_metrics import paired_bootstrap_ci, prefix_consistent, trajectory_metrics


def unit(key: str):
    return {"edge_keys": [key]}


def test_trajectory_gain_duplicates_and_prefix():
    metrics = trajectory_metrics([[unit("a"), unit("b")], [unit("b"), unit("c")]])
    assert metrics["totalUniqueUnits"] == 3
    assert metrics["iterations"][1]["uniqueEvidenceGain"] == 1
    assert metrics["iterations"][1]["duplicateRate"] == 0.5
    assert prefix_consistent([unit("a"), unit("b")], [unit("a"), unit("b"), unit("c")])
    assert not prefix_consistent([unit("b")], [unit("a")])


def test_paired_bootstrap_reports_candidate_gain():
    result = paired_bootstrap_ci([(0.2, 0.4), (0.3, 0.5), (0.4, 0.6)], samples=300)
    assert round(result["delta"], 6) == 0.2
    assert result["low"] > 0
