"""Tests for the redesigned page: structure, accessibility hooks and design-system rules."""
import io
import re

import pytest

import app

SAMPLE_JOBS = [
    {"title": "Data Scientist", "company": "Acme", "url": "https://example.com/1", "location": "EU", "related": False},
    {"title": "Office Assistant", "company": "Beta", "url": "https://example.com/2", "location": "EU", "related": True},
]


def page(monkeypatch, data=None, jobs=SAMPLE_JOBS):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: jobs)
    client = app.app.test_client()
    if data is None:
        return client.get("/").get_data(as_text=True)
    return client.post("/", data=data, content_type="multipart/form-data").get_data(as_text=True)


def resume_page(monkeypatch, jobs=SAMPLE_JOBS):
    with open("tests/fixtures/good_resume.txt", "rb") as f:
        return page(monkeypatch, {"resume_file": (io.BytesIO(f.read()), "cv.txt")}, jobs)


def card_order(html):
    return re.findall(r'data-card="(\w+)"', html)


# --- layout ------------------------------------------------------------------------

def test_results_are_in_the_mobile_reading_order(monkeypatch):
    assert card_order(resume_page(monkeypatch)) == [
        "summary", "careers", "score", "gap", "roadmap", "courses", "whatif", "jobs"]


def test_typed_skills_have_no_score_card_but_keep_the_order(monkeypatch):
    html = page(monkeypatch, {"input_text": "linux, aws, a little git"})
    assert card_order(html) == ["summary", "careers", "gap", "roadmap", "courses", "whatif", "jobs"]
    assert "Upload a resume to get a score" in html


def test_summary_strip_has_the_four_numbers(monkeypatch):
    html = resume_page(monkeypatch)
    summary = html.split('data-card="summary"')[1].split('class="dashboard"')[0]
    for label in ["Recommended career path", "Confidence", "Resume score", "Key skills covered"]:
        assert label in summary
    assert 'aria-label="Resume score' in summary            # the small ring


def test_home_page_shows_the_preview_not_results(monkeypatch):
    html = page(monkeypatch)
    assert 'class="preview"' in html
    assert 'id="results"' not in html


# --- top bar, input, footer --------------------------------------------------------

def test_top_bar_has_brand_dark_mode_toggle_and_github(monkeypatch):
    html = page(monkeypatch)
    assert 'id="theme-toggle"' in html and 'aria-label="Toggle dark mode"' in html
    assert 'href="https://github.com/Galvin70124O/CareerPilotAI"' in html
    assert "CareerPilot <span>AI</span>" in html


def test_theme_is_set_before_paint_from_saved_choice_or_system(monkeypatch):
    head = page(monkeypatch).split("</head>")[0]
    assert 'localStorage.getItem("theme")' in head
    assert "prefers-color-scheme: dark" in head


def test_upload_zone(monkeypatch):
    html = page(monkeypatch)
    assert 'class="dropzone" for="resume_file" data-dropzone' in html
    file_input = re.search(r'<input[^>]*id="resume_file"[^>]*>', html, re.S).group(0)
    assert 'name="resume_file"' in file_input and 'type="file"' in file_input
    for extension in [".pdf", ".docx", ".txt"]:
        assert extension in file_input
    assert "data-file-name" in html and "data-file-size" in html


def test_tabs_are_accessible(monkeypatch):
    html = page(monkeypatch)
    assert 'role="tablist"' in html
    assert html.count('role="tab"') == 2 and html.count('role="tabpanel"') == 2
    assert 'aria-controls="panel-upload"' in html and 'aria-controls="panel-text"' in html


def test_text_tab_is_open_after_typed_skills(monkeypatch):
    html = page(monkeypatch, {"input_text": "python, sql"})
    assert re.search(r'id="tab-text"[^>]*aria-selected="true"', html, re.S)
    assert re.search(r'id="panel-upload"[^>]*hidden', html, re.S)


def test_form_fields_have_labels(monkeypatch):
    html = page(monkeypatch)
    assert '<label class="field-label" for="input_text">' in html
    assert 'id="mic-button"' in html and 'aria-pressed="false"' in html


def test_loading_state_is_ready_but_hidden(monkeypatch):
    html = page(monkeypatch)
    assert re.search(r'<section class="results" id="loading" hidden', html)
    assert 'role="status" data-loading-status' in html


def test_results_are_announced_and_can_be_restarted(monkeypatch):
    html = resume_page(monkeypatch)
    assert re.search(r'id="results"[^>]*aria-live="polite"', html)
    assert "data-analyze-another" in html and "Analyze another resume" in html


def test_footer_credits(monkeypatch):
    footer = page(monkeypatch).split('<footer class="footer">')[1]
    assert "ResumeAtlas" in footer and "MIT" in footer and "Remotive" in footer and "GitHub" in footer


# --- states ------------------------------------------------------------------------

def test_unreadable_file_shows_a_designed_alert(monkeypatch):
    html = page(monkeypatch, {"resume_file": (io.BytesIO(b"x"), "photo.png")})
    assert '<div class="alert" role="alert">' in html
    assert "We couldn't analyze that" in html


def test_jobs_unavailable_is_an_empty_state(monkeypatch):
    html = resume_page(monkeypatch, jobs=None)
    jobs = html.split('data-card="jobs"')[1]
    assert 'class="empty"' in jobs and "Live jobs are unavailable right now" in jobs


def test_job_cards_show_related_badge_and_credit(monkeypatch):
    html = resume_page(monkeypatch)
    assert html.count('class="job"') == 2
    assert html.count("data-related") == 1
    assert "Jobs from <a href=\"https://remotive.com\"" in html


# --- design-system rules -----------------------------------------------------------

@pytest.mark.parametrize("data", [
    None,
    {"input_text": "python, sql"},
    {"resume_file": (io.BytesIO(b"x"), "photo.png")},
])
def test_no_inline_styles_no_bootstrap_no_inline_handlers(monkeypatch, data):
    html = page(monkeypatch, data)
    assert not re.search(r'\sstyle="', html)
    assert "bootstrap" not in html.lower()
    assert not re.search(r'\son(click|change|load)=', html)


def test_resume_page_has_no_inline_styles_or_emoji(monkeypatch):
    html = resume_page(monkeypatch)
    assert not re.search(r'\sstyle="', html)
    assert not re.search("[\U0001F300-\U0001FAFF☀-⛿]", html)   # icons are SVG, not emoji


def test_css_and_js_are_linked_and_served(monkeypatch):
    html = page(monkeypatch)
    assert 'href="/static/css/app.css"' in html and 'src="/static/js/app.js"' in html
    client = app.app.test_client()
    css = client.get("/static/css/app.css")
    assert css.status_code == 200
    text = css.get_data(as_text=True)
    assert ':root[data-theme="dark"]' in text and "prefers-color-scheme: dark" in text
    assert "--accent:" in text and "--radius-lg:" in text and "--space-4:" in text
    assert client.get("/static/js/app.js").status_code == 200


def test_bars_are_progress_elements(monkeypatch):
    html = resume_page(monkeypatch)
    assert html.count('<progress class="bar') >= 3 + 6 + 1   # top 3 + score breakdown + skill gap
