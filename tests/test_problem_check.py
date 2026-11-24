from src.evaluation.problem_check import bootstrap, error_kind, numbers, pair_scores, prf, summary, variant


def test_numbers_reads_labels():
    assert numbers("1 4") == {1, 4}
    assert numbers("2, 1") == {1, 2}
    assert numbers("1.0") == {1}
    assert numbers("0") == set()
    assert numbers("") is None
    assert numbers(float("nan")) is None


def test_prf_counts_problems_not_complaints():
    pairs = [({1}, {1}), ({1, 2}, {1}), (set(), {3}), ({2}, {2, 4})]
    precision, recall, f1 = prf(pairs)
    assert precision == 3 / 5
    assert recall == 3 / 4
    assert round(f1, 3) == round(2 * 0.6 * 0.75 / 1.35, 3)


def test_error_kinds():
    assert error_kind({1}, {1}) == "same"
    assert error_kind(set(), {1}) == "model_problem_where_none"
    assert error_kind({1}, set()) == "model_none"
    assert error_kind({1}, {1, 2}) == "model_extra_problem"
    assert error_kind({1, 2}, {1}) == "model_missed_problem"
    assert error_kind({1}, {2}) == "model_other_problem"
    assert error_kind({1, 2}, {2, 3}) == "model_partly_other"


def test_summary_and_bootstrap_are_stable():
    pairs = [({1}, {1})] * 8 + [({1}, {2})] * 2
    result = summary(pairs, [1, 1, 0, 1])
    assert result["exact_match"] == 0.8
    assert result["details_supported"] == 0.75
    assert result["errors"] == {"same": 8, "model_other_problem": 2}
    assert bootstrap(pairs, pair_scores) == bootstrap(pairs, pair_scores)
    low, high = result["ci95"]["precision"]
    assert low <= result["precision"] <= high


def test_variant_names_problems():
    titles = {1: "Підтримка", 2: "Затримки"}
    assert variant({2, 1}, titles) == "1 — Підтримка\n2 — Затримки"
    assert variant(set(), titles).startswith("0")