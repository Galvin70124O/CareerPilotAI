import app
from features.explain import top_contributing_terms


def test_terms_come_from_the_text_and_push_the_career():
    text = "Java developer: Spring, Hibernate, J2EE and Maven. Also likes cooking."
    terms = top_contributing_terms(app.model, text, "Java Developer")
    assert 3 <= len(terms) <= 4
    assert "java" in terms
    assert set(terms) <= {"java", "developer", "spring", "hibernate", "j2ee", "maven", "likes", "cooking"}
    assert "cooking" not in terms[:2]                                    # unrelated words don't lead


def test_different_careers_get_different_explanations():
    text = "python pandas sql machine learning tableau react javascript css"
    science = top_contributing_terms(app.model, text, "Data Science")
    react = top_contributing_terms(app.model, text, "React Developer")
    assert science != react
    assert "react" in react or "javascript" in react


def test_text_with_no_known_words_gives_no_terms():
    assert top_contributing_terms(app.model, "zzqx blorf", "Data Science") == []


def test_page_shows_why_this_match(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    html = app.app.test_client().post("/", data={"input_text": "java spring hibernate j2ee"}).get_data(as_text=True)
    careers = html.split('data-card="careers"')[1].split("</article>")[0]
    assert "Why this match" in careers and "data-why-match" in careers
    assert '<li class="chip chip-accent">Java</li>' in careers            # shown with display names
