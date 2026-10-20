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
    summary = html.split('data-card="summary"')[1].split('class="dashboard ')[0]
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
    assert "data-analyze-another" in html and "Analyze another" in html


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


# --- polish round: compact bar, grid areas, skill names, bar colours, courses -------

def test_results_page_has_compact_bar_instead_of_hero(monkeypatch):
    html = resume_page(monkeypatch)
    bar = html.split("data-compact-bar")[1].split('id="analyze-panel"')[0]
    assert "Analyzed resume" in bar and "cv.txt" in bar
    assert 'id="edit-skills" aria-expanded="false" aria-controls="analyze-panel"' in bar
    assert "data-analyze-another" in bar and "Analyze another" in bar
    assert 'class="hero"' not in html and 'id="hero-title"' not in html   # (the tagline is still in the og: meta tags)
    assert re.search(r'id="analyze-panel" hidden', html)                  # editor closed until asked
    assert html.index("data-compact-bar") < html.index('data-card="summary"')


def test_compact_bar_for_typed_skills(monkeypatch):
    html = page(monkeypatch, {"input_text": "linux, aws"})
    bar = html.split("data-compact-bar")[1].split('id="analyze-panel"')[0]
    assert "Typed skills" in bar and "linux, aws" in bar


def test_edit_skills_panel_is_prefilled_with_found_skills(monkeypatch):
    html = resume_page(monkeypatch)
    textarea = re.search(r'<textarea[^>]*id="input_text"[^>]*>(.*?)</textarea>', html, re.S).group(1)
    assert "python" in textarea and "sql" in textarea


def test_full_hero_only_without_results(monkeypatch):
    html = page(monkeypatch)
    assert 'class="hero"' in html and "data-compact-bar" not in html


def test_dashboard_is_one_grid_in_mobile_order(monkeypatch):
    html = resume_page(monkeypatch)
    assert 'class="dashboard dashboard--full"' in html
    assert "dashboard-col" not in html                                    # no column wrappers / DOM reordering
    grid = html.split('class="dashboard dashboard--full"')[1]
    assert re.findall(r'data-card="(\w+)"', grid) == ["careers", "score", "gap", "roadmap", "courses", "whatif", "jobs"]


def test_missing_cards_use_the_auto_layout(monkeypatch):
    html = page(monkeypatch, {"input_text": "linux, aws"})               # typed: no resume score
    assert 'class="dashboard dashboard--auto"' in html


def test_desktop_grid_areas_in_css():
    css = app.app.test_client().get("/static/css/app.css").get_data(as_text=True)
    desktop = css.split("@media (min-width: 1024px)")[1]
    areas = re.sub(r"\s+", " ", desktop.split("grid-template-areas:")[1].split(";")[0])
    assert areas.strip() == '"careers gap" "score roadmap" "courses whatif" "jobs jobs"'
    for card in ["careers", "gap", "score", "roadmap", "courses", "whatif", "jobs"]:
        assert f'.dashboard--full > [data-card="{card}"] {{ grid-area: {card}; }}' in desktop


def test_skill_names_are_shown_with_display_names(monkeypatch):
    html = resume_page(monkeypatch)
    gap = html.split('data-card="gap"')[1].split('data-card="roadmap"')[0]
    assert '<li class="chip chip-missing">R</li>' in gap                    # not "r"
    assert ">Python</li>" in gap and ">SQL</li>" in gap
    whatif = html.split('data-card="whatif"')[1]
    assert '<input type="checkbox" value="r">' in whatif                   # the value stays the internal name
    assert "</svg>R</span>" in whatif


def test_breakdown_bars_are_coloured_by_ratio(monkeypatch):
    html = resume_page(monkeypatch)
    # good_resume.txt: Sections 20/20, Length 10/15 (67%), Skill match 22/25 (88%)
    assert re.search(r'breakdown-good">\s*<div class="breakdown-row">\s*<span>Sections</span>', html)
    assert re.search(r'breakdown-ok">\s*<div class="breakdown-row">\s*<span>Length &amp; readability</span>', html)
    assert re.search(r'breakdown-ok">\s*<div class="breakdown-row">\s*<span>Skill match</span>', html)


def test_breakdown_red_for_a_weak_resume(monkeypatch):
    with open("tests/fixtures/weak_resume.txt", "rb") as f:
        html = page(monkeypatch, {"resume_file": (io.BytesIO(f.read()), "weak.txt")})
    assert html.count("breakdown-low") >= 4


def test_bar_colours_use_the_ring_tokens():
    css = app.app.test_client().get("/static/css/app.css").get_data(as_text=True)
    assert ".breakdown-good .bar::-webkit-progress-value { background: var(--success); }" in css
    assert ".breakdown-ok .bar::-webkit-progress-value { background: var(--warning); }" in css
    assert ".breakdown-low .bar::-webkit-progress-value { background: var(--danger); }" in css
    assert ".ring-good .ring-value { stroke: var(--success); }" in css


def test_single_letter_chips_have_a_min_width():
    css = app.app.test_client().get("/static/css/app.css").get_data(as_text=True)
    chip = css.split(".chip {")[1].split("}")[0]
    assert "min-width:" in chip


def test_courses_one_row_per_skill_and_good_news_note(monkeypatch):
    html = resume_page(monkeypatch)                                       # only "r" is missing
    courses = html.split('data-card="courses"')[1].split('data-card="whatif"')[0]
    assert courses.count('class="course-row"') == 1
    assert "Good news: only 1 skill to learn." in courses
    assert '<span class="chip chip-missing course-skill">R</span>' in courses


def test_courses_note_for_several_skills(monkeypatch):
    html = page(monkeypatch, {"input_text": "linux, aws"})
    courses = html.split('data-card="courses"')[1].split('data-card="whatif"')[0]
    rows = courses.count('class="course-row"')
    assert rows > 1 and f"{rows} skills to learn" in courses


def test_readme_screenshot_captions_say_sample_job_data():
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()
    screenshots = readme.split("## Screenshots")[1].split("## Features")[0]
    assert screenshots.count("sample job data") >= 2


def test_auto_layout_has_no_empty_cell_before_jobs():
    css = app.app.test_client().get("/static/css/app.css").get_data(as_text=True)
    assert ".dashboard--auto > :nth-child(odd):nth-last-child(2) { grid-column: 1 / -1; }" in css
