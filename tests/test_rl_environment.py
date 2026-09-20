from copy import deepcopy
import json

import pytest

from story_copilot.rl_environment import EvidenceEpisode, demonstrations, make_case, make_suite


def run_expert(case):
    episode = EvidenceEpisode(case)
    for decision in case["expert"]:
        episode.step(json.dumps(decision))
    return episode


def test_oracle_solutions_use_the_real_tools_across_all_scenario_families():
    for split in ["train", "validation", "test", "transfer"]:
        cases = make_suite(8, split=split)
        for case in cases:
            before = deepcopy(case["context"])
            episode = run_expert(case)
            assert episode.assessment()["success"], case["family"]
            assert case["context"] == before == episode.context


def test_success_requires_seen_correct_sources_and_strict_conclusion_types():
    case = make_case(17, "activation")
    episode = EvidenceEpisode(case)
    episode.step(json.dumps(case["expected"]))
    assert not episode.assessment()["success"]  # Even an oracle answer cannot cite unseen evidence.
    episode = EvidenceEpisode(case)
    for decision in case["expert"][:-1]:
        episode.step(json.dumps(decision))
    wrong = deepcopy(case["expected"])
    wrong["allowed"] = "true"
    episode.step(json.dumps(wrong))
    assert episode.assessment()["invalid_actions"] == 1
    assert not episode.assessment()["success"]


def test_stale_count_citation_dump_and_always_clarify_do_not_win():
    case = make_case(11, "correction")
    episode = EvidenceEpisode(case)
    episode.step(json.dumps(case["expert"][0]))
    wrong = deepcopy(case["expected"])
    wrong["value"] = next(iter(case["context"]["frozen_snapshot"]["characters"][0]["sheet"]["resources"].values()))
    episode.step(json.dumps(wrong))
    assert not episode.assessment()["success"]
    for family in ["activation", "visible", "sheet"]:
        case = make_case(11, family)
        episode = EvidenceEpisode(case)
        episode.step(json.dumps({"action": "clarify", "question": "What is the balance?", "missing": ["balance"]}))
        assert not episode.assessment()["success"]
    case = make_case(12, "correction")
    episode = EvidenceEpisode(case)
    episode.step(json.dumps(case["expert"][0]))
    wrong = {**case["expected"], "sources": list(episode.seen)}
    episode.step(json.dumps(wrong))
    assert not episode.assessment()["success"]


def test_missing_information_has_a_grounded_clarification_and_answerable_variants():
    for family, missing in [("missing_balance", "balance"), ("missing_rule", "rule")]:
        case = make_case(3, family)
        episode = run_expert(case)
        assert episode.assessment()["success"]
        assert episode.final["missing"] == [missing]
        assert episode.final["value"] is None
        assert episode.final["allowed"] is None


def test_equivalent_sources_and_recovery_do_not_require_useless_tool_calls():
    case = make_case(3, "missing_balance")
    for alternative in case["source_alternatives"]:
        episode = EvidenceEpisode(case)
        if alternative == case["expected"]["sources"]:
            episode.step(json.dumps(case["expert"][0]))
        else:
            name = case["context"]["frozen_snapshot"]["characters"][0]["name"]
            episode.step(json.dumps({"action": "recall", "query": name}))
        episode.step(json.dumps({**case["expected"], "sources": alternative}))
        assert episode.assessment()["success"] and episode.calls == 1
    case = make_case(4, "visible")
    episode = EvidenceEpisode(case)
    episode.step('{"action":"wrong"}')
    episode.step('```json\n' + json.dumps(case["expected"]) + '\n```')
    assert episode.assessment()["success"]
    assert episode.assessment()["invalid_actions"] == 1


def test_tool_budget_repeated_requests_and_episode_reset():
    case = make_case(4, "correction")
    a, b = EvidenceEpisode(case), EvidenceEpisode(case)
    request = json.dumps(case["expert"][0])
    a.step(request)
    a.step(request)
    a.step('{"action":"publish","query":"alter state"}')
    a.step('{"action":"recall","query":"unknown"}')
    assert a.done and a.invalid == 2
    assert not b.trace and not b.seen and b.calls == 0
    with pytest.raises(ValueError, match="ended"):
        a.step(request)


def test_partition_identity_demonstration_boundaries_and_no_oracle_in_prompt():
    groups = {s: make_suite(3, split=s) for s in ["train", "validation", "test", "transfer"]}
    ids = [c["id"] for values in groups.values() for c in values]
    assert len(ids) == len(set(ids))
    for values in groups.values():
        for case in values:
            public = json.loads(case["prompt"][1]["content"])
            assert set(public) == {"episode", "task", "characters", "inventory_status", "visible_evidence", "available_tools"}
            assert "expected" not in public and "family" not in public
    rows = demonstrations(groups["train"])
    assert rows and all(json.loads(row["completion"])["action"] for row in rows)
    with pytest.raises(ValueError, match="training partition"):
        demonstrations(groups["test"])
