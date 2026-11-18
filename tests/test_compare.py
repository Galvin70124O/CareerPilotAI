"""features/compare.py and POST /api/compare ("Compare with another version")."""
import io

import app
from features import rate_limit
from features.compare import SAME, compare_versions, number_change, snapshot, tag_change
from features.rate_limit import LIMIT

with open("tests/fixtures/good_resume.txt", encoding="utf-8") as f:
    GOOD = f.read()
with open("tests/fixtures/weak_resume.txt", encoding="utf-8") as f:
    WEAK = f.read()


def compare(before_text, after_text):
    top_before, top_after = app.predict_top_careers(before_text), app.predict_top_careers(after_text)
    career = top_before[0]["career"]
    return compare_versions(snapshot(app.model, before_text, top_before, career),
                            snapshot(app.model, after_text, top_after, career))


def rows_of(result):
    return {(section["title"], r["label"]): r for section in result["sections"] for r in section["rows"]}


# --- the diff ----------------------------------------------------------------------------

def test_number_and_tag_changes_are_worded_with_a_sign():
    assert (number_change(5, 8), number_change(8, 5), number_change(4, 4)) == ("+3", "−3", SAME)
    assert [tag_change("claimed", "proven"), tag_change("proven", "coursework"), tag_change(None, "claimed"),
            tag_change("claimed", None), tag_change("proven", "proven")] == ["stronger", "weaker", "new", "gone", SAME]


def test_two_identical_versions_have_no_changes():
    result = compare(GOOD, GOOD)
    assert result["unchanged"] and not result["top_changed"] and result["note"] == ""
    assert result["finding"] == "No changes: both versions give the same results."
    assert all(r["change"] == SAME for r in rows_of(result).values())


def test_the_finding_names_the_score_and_readiness_change_and_the_career_it_is_measured_against():
    result = compare(GOOD, WEAK)
    total = rows_of(result)[("Resume Score", "Total")]
    assert result["finding"] == (f"The new version scores {total['after']} ({total['change']}). "
                                 'Readiness moves down from "Ready to apply" to "Not ready yet".')
    assert result["top_changed"] and "still measured against Data Science" in result["note"]


def test_every_section_is_there_with_each_score_line():
    result = compare(GOOD, GOOD)
    assert [s["title"] for s in result["sections"]] == [
        "Resume Score", "Career match", "Skill gap for Data Science", "Evidence", "Readiness"]
    score_labels = [r["label"] for r in result["sections"][0]["rows"]]
    assert score_labels[0] == "Total" and len(score_labels) == 7          # total + the 6 breakdown lines
    assert [r["label"] for r in result["sections"][1]["rows"]][1:] == ["1st match", "2nd match", "3rd match"]


def test_a_much_worse_version_shows_drops_against_the_first_versions_career():
    result = compare(GOOD, WEAK)
    rows = rows_of(result)
    assert not result["unchanged"]
    assert rows[("Resume Score", "Total")]["change"].startswith("−")
    assert rows[("Skill gap for Data Science", "Key skills in your resume")]["change"].startswith("−")
    assert "lost" in rows[("Skill gap for Data Science", "Missing key skills")]["change"]
    assert rows[("Readiness", "Verdict")]["before"] == "Ready to apply"
    assert rows[("Readiness", "Verdict")]["change"] == "down"
    # every row is about Data Science (version 1's top career), even if the weak version ranks it lower
    assert ("Career match", "Match with Data Science") in rows


def test_better_evidence_is_stronger():
    # version 1 only lists SQL; version 2 uses it in a project
    before = GOOD.replace("- Designed a SQL database", "- Designed a database").replace("optimized 6 slow SQL queries", "optimized 6 slow queries")
    rows = rows_of(compare(before, GOOD))
    assert rows[("Evidence", "SQL")]["before"] == "claimed" and rows[("Evidence", "SQL")]["after"] == "proven"
    assert rows[("Evidence", "SQL")]["change"] == "stronger"


# --- the route ---------------------------------------------------------------------------

def post(text=GOOD, file=None, name="v2.txt"):
    data = {"text": text}
    if file is not None:
        data["resume_file"] = (io.BytesIO(file), name)
    return app.app.test_client().post("/api/compare", data=data, content_type="multipart/form-data")


def test_api_compares_and_stores_nothing():
    # headings added: the bare weak resume would get the not-a-resume message (tests/test_resume_check.py)
    response = post(file=("SKILLS\nComputers\nEXPERIENCE\n" + WEAK).encode())
    assert response.status_code == 200
    assert response.get_json()["career"] == "Data Science"


def test_api_checks_the_second_file_like_the_first():
    assert "Choose the other version" in post().get_json()["error"]
    for body, name in [(b"MZ...", "v2.exe"), (b"a" * (app.MAX_FILE_BYTES + 1), "big.txt")]:
        response = post(file=body, name=name)
        assert response.status_code == 400 and response.get_json()["error"] == app.UPLOAD_RULE
    assert "Could not open this file" in post(file=b"not really a pdf", name="fake.pdf").get_json()["error"]
    assert "Could not read any text" in post(file=b"   ", name="empty.txt").get_json()["error"]
    assert post(text="", file=WEAK.encode()).status_code == 400


def test_a_request_over_the_size_cap_gets_a_json_answer():
    response = post(file=b"a" * (app.MAX_CONTENT_LENGTH + 1), name="huge.txt")
    assert response.status_code == 413 and response.get_json()["error"] == app.UPLOAD_RULE


def test_api_is_rate_limited():
    for _ in range(LIMIT):
        rate_limit.allow("127.0.0.1")
    assert post(file=WEAK.encode()).status_code == 429


def test_the_home_upload_uses_the_same_checks():
    client = app.app.test_client()
    html = client.post("/", data={"resume_file": (io.BytesIO(b"not really a pdf"), "fake.pdf")},
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "Could not open this file" in html and "Traceback" not in html


def test_each_change_has_a_direction_for_the_colours():
    worse = rows_of(compare(GOOD, WEAK))
    assert worse[("Resume Score", "Total")]["direction"] == "worse"
    assert worse[("Readiness", "Verdict")]["direction"] == "worse"
    assert worse[("Skill gap for Data Science", "Missing key skills")]["direction"] == "worse"   # lost only
    better = rows_of(compare(WEAK, GOOD))
    assert better[("Resume Score", "Total")]["direction"] == "better"
    # a different 2nd or 3rd match is neither better nor worse
    assert all(r["direction"] is None for (section, label), r in better.items() if label in ("2nd match", "3rd match"))
    same = rows_of(compare(GOOD, GOOD))
    assert all(r["direction"] is None for r in same.values())


def test_direction_and_mixed_skill_changes():
    from features.compare import direction, gap_rows
    assert (direction(1, 2), direction(2, 1), direction(2, 2)) == ("better", "worse", None)
    before = {"gap": {"have": ["sql"], "missing": ["git"], "total": 2}}
    after = {"gap": {"have": ["git"], "missing": ["sql"], "total": 2}}
    assert gap_rows(before, after)[1]["direction"] is None     # found one, lost one
