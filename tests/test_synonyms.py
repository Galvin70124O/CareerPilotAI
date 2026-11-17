"""Synonyms (skill_synonyms.json) and the context rules for short or ambiguous names."""
import json
import os

import pytest

from features.skills import SYNONYMS, has_skill

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_synonym_file_keeps_each_source_apart():
    with open(os.path.join(PROJECT_DIR, "skill_synonyms.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert set(data["sources"]) == {"esco", "onet", "india"}
    for sources in data["synonyms"].values():
        assert set(sources) <= {"esco", "onet", "india"}


@pytest.mark.parametrize("skill, text", [
    ("excel", "Tools: MS Excel, Word"),
    ("excel", "Microsoft Excel (pivot tables)"),
    ("javascript", "HTML, CSS, JS"),
    ("data structures and algorithms", "Solved 300+ DSA problems on LeetCode"),
    ("internship", "6 week in-plant training at BHEL"),
    ("nptel", "NPTEL certificate in Python for Data Science"),
    ("sql", "Structured Query Language queries"),
])
def test_synonyms_are_found(skill, text):
    assert has_skill(skill, text)


@pytest.mark.parametrize("text", ["B.E. Mechanical, 2025", "B.Tech (CSE)", "BTech in ECE",
                                  "Bachelor of Engineering", "BE (Civil), 7.8 CGPA"])
def test_engineering_degree_is_found(text):
    assert has_skill("b.e./b.tech", text)


@pytest.mark.parametrize("text", ["I want to be a good engineer", "B.Ed, 2019", "Will be in Pune"])
def test_be_as_a_word_or_bed_is_not_an_engineering_degree(text):
    assert not has_skill("b.e./b.tech", text)


@pytest.mark.parametrize("text", ["Golang microservices", "Languages: Java, Go", "Go programming",
                                  "Python / Go / Rust"])
def test_go_language_is_found(text):
    assert has_skill("go", text)


@pytest.mark.parametrize("text", ["Ready to go the extra mile", "I like to go hiking", "go-getter",
                                  "Python and go to meetups", "Google Cloud"])
def test_go_as_a_word_is_not_the_language(text):
    assert not has_skill("go", text)


def test_js_is_whole_word_only():
    assert not has_skill("javascript", "Worked at JSW Steel")


def test_no_synonym_is_shorter_than_three_letters_except_js():
    short = {word for words in SYNONYMS.values() for word in words if len(word) < 3}
    assert short <= {"js"}
