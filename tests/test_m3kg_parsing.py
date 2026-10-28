from mmrag.m3kg.agents import (clean_concept, parse_index, parse_kept_indices, parse_score,
                               parse_triples)
from mmrag.m3kg.kb import KnowledgeBase, parse_wikipedia, parse_wiktionary
from mmrag.m3kg.vllm_stages import build_text, parse_judge, parse_winrate


def test_parse_triples_handles_commas_bullets_and_duplicates():
    txt = """(a dog, barks at, a cat)
- (man, speaks with, crowd, loudly)
1. (engine, idles, )
(a dog, barks at, a cat)
not a triple
(water, flows over, rocks)."""
    assert parse_triples(txt) == [("a dog", "barks at", "a cat"), ("man", "speaks with, crowd", "loudly"),
                                  ("water", "flows over", "rocks")]


def test_clean_concept():
    assert clean_concept('"Jack Russell Terrier".\nextra', "x") == "Jack Russell Terrier"
    assert clean_concept("Concept: dog", "x") == "dog"
    assert clean_concept("", "fallback") == "fallback"


def test_parse_index_score_and_kept():
    assert parse_index("2", 3) == 1 and parse_index("Candidate 5", 3) is None and parse_index("", 3) is None
    assert parse_score("8") == 8 and parse_score("Score: 10") == 10 and parse_score("none") == 0
    assert parse_kept_indices("[0, 2, 9]", 4) == [0, 2]
    assert parse_kept_indices("Kept: []", 3) == []
    assert parse_kept_indices("no list here", 3) == [0, 1, 2]       # conservative: keep all


def test_wikipedia_and_wiktionary_parsers():
    wiki = {"query": {"pages": [
        {"index": 2, "title": "Bark (sound)", "extract": "A bark is a sound made by dogs. More text. Third."},
        {"index": 1, "title": "Dog", "extract": "The dog is a domesticated canine. It is a pet."},
        {"index": 3, "title": "Bark", "extract": "Bark may refer to: tree bark, dog bark"}]}}
    assert parse_wikipedia(wiki) == ["The dog is a domesticated canine. It is a pet.",
                                     "A bark is a sound made by dogs. More text."]
    wikt = {"en": [{"partOfSpeech": "Noun", "definitions": [{"definition": "The <b>sound</b> made by a dog."},
                                                            {"definition": "x"}]}]}
    assert parse_wiktionary(wikt) == ["The sound made by a dog."]


def test_knowledge_base_uses_cache_and_fake_http(tmp_path):
    from mmrag.common.cache import KVCache

    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        if "wikipedia" in url:
            return {"query": {"pages": [{"index": 1, "extract": f"About {params['gsrsearch']}."}]}}
        return {}

    cache = KVCache(tmp_path / "c.sqlite")
    kb = KnowledgeBase(cache, threads=2, http_get=fake_get)
    res = kb.lookup_many(["dog", "siren", "dog"])
    assert res["dog"]["wikipedia"] == ["About dog."]
    n = len(calls)
    kb.lookup_many(["dog"])
    assert len(calls) == n  # served from cache


def test_build_text_and_judge_parsing():
    assert build_text("What is heard?", []) == "What is heard?"
    t = build_text("Q?", [{"h": "dog", "r": "barks", "t": "loudly", "hd": "-", "td": "-"}])
    assert "[0] head=dog | relation=barks | tail=loudly" in t and "Q?" in t
    assert parse_judge("Explanation: ... Rating: 4") == (4.0, "audiobench")
    assert parse_judge("Rating: 3\nThanks!") == (3.0, "regex")
    assert parse_judge("garbage") == (0.0, "failed")
    assert parse_winrate('{"Comprehensiveness": 2, "Diversity": "Answer 1", "Empowerment": 2, "Overall": 2}') == \
        {"Comprehensiveness": 2, "Diversity": 1, "Empowerment": 2, "Overall": 2}
    assert parse_winrate("nope") is None


def test_audiobench_template_loads_from_clone():
    import pytest
    from conftest import REPO
    from mmrag.m3kg.vllm_stages import audiobench_template

    if not (REPO / "third_party" / "AudioBench").exists():
        pytest.skip("AudioBench not cloned")
    tpl, _ = audiobench_template({"judge": {"audiobench_dir": str(REPO / "third_party" / "AudioBench")}})
    filled = tpl.format(question="q", reference="r", prediction="p")
    assert "[Reference Answer]" in filled and "Rating: (int)" in filled and "{" not in filled
