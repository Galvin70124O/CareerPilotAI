"""Logo, favicons, link previews, sample resume, loading steps, GitHub tooltip."""
import io
import os
import re

from PIL import Image

import app

STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "img")
SITE = "https://careerpilotai-1.onrender.com"


def home(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    return app.app.test_client().get("/").get_data(as_text=True)


def sample(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    return app.app.test_client().post("/", data={"sample": "1"}).get_data(as_text=True)


# --- logo + favicons -----------------------------------------------------------------

def test_logo_files_exist_and_are_svg():
    for name in ["logo.svg", "logo-dark.svg", "favicon.svg"]:
        with open(os.path.join(STATIC, name), encoding="utf-8") as f:
            svg = f.read()
        assert svg.startswith("<svg") and 'viewBox="0 0 32 32"' in svg, name
        assert "<text" not in svg, name                        # no text inside the icon
    with open(os.path.join(STATIC, "favicon.svg"), encoding="utf-8") as f:
        assert "prefers-color-scheme:dark" in f.read()          # the favicon follows the browser theme


def test_png_icons_have_the_right_sizes():
    assert Image.open(os.path.join(STATIC, "favicon-32.png")).size == (32, 32)
    assert Image.open(os.path.join(STATIC, "apple-touch-icon.png")).size == (180, 180)
    assert Image.open(os.path.join(STATIC, "og-image.png")).size == (1200, 630)


def test_logo_in_top_bar_and_footer(monkeypatch):
    html = home(monkeypatch)
    topbar = html.split('<header class="topbar">')[1].split("</header>")[0]
    footer = html.split('<footer class="footer">')[1]
    assert 'class="logo-mark"' in topbar and 'class="logo-mark"' in footer


def test_favicon_links(monkeypatch):
    head = home(monkeypatch).split("</head>")[0]
    assert '<link rel="icon" href="/static/img/favicon.svg" type="image/svg+xml">' in head
    assert 'href="/static/img/favicon-32.png" type="image/png" sizes="32x32"' in head
    assert '<link rel="apple-touch-icon" href="/static/img/apple-touch-icon.png" sizes="180x180">' in head
    client = app.app.test_client()
    for path in ["favicon.svg", "favicon-32.png", "apple-touch-icon.png", "og-image.png"]:
        assert client.get(f"/static/img/{path}").status_code == 200, path


# --- link previews -----------------------------------------------------------------

def test_open_graph_and_twitter_tags(monkeypatch):
    head = home(monkeypatch).split("</head>")[0]
    assert f'<meta property="og:url" content="{SITE}">' in head
    assert f'<meta property="og:image" content="{SITE}/static/img/og-image.png">' in head
    assert '<meta property="og:title" content="CareerPilot AI · Find the career your resume is built for">' in head
    assert 'property="og:description"' in head and 'property="og:type" content="website"' in head
    assert '<meta name="twitter:card" content="summary_large_image">' in head
    assert f'<meta name="twitter:image" content="{SITE}/static/img/og-image.png">' in head
    assert head.count('name="theme-color"') == 2                # light + dark


# --- GitHub tooltip ----------------------------------------------------------------

def test_github_link_has_tooltip(monkeypatch):
    topbar = home(monkeypatch).split('<header class="topbar">')[1].split("</header>")[0]
    link = re.search(r'<a[^>]*github\.com/Galvin70124O/CareerPilotAI[^>]*>', topbar, re.S).group(0)
    assert 'data-tooltip="View source on GitHub"' in link and "has-tooltip" in link
    css = app.app.test_client().get("/static/css/app.css").get_data(as_text=True)
    assert ".has-tooltip:hover::after, .has-tooltip:focus-visible::after" in css   # hover AND keyboard focus


# --- sample resume -----------------------------------------------------------------

def test_sample_resume_file_exists():
    with open(app.SAMPLE_RESUME_PATH, encoding="utf-8") as f:
        text = f.read()
    assert "Experience" in text and "@example.com" in text      # fictional contact details


def test_sample_button_under_the_upload_zone(monkeypatch):
    html = home(monkeypatch)
    upload_panel = html.split('id="panel-upload"')[1].split('id="panel-text"')[0]
    assert re.search(r'<button type="submit"[^>]*name="sample" value="1"', upload_panel)
    assert "Try with a sample resume" in upload_panel


def test_sample_button_analyzes_the_sample(monkeypatch):
    html = sample(monkeypatch)
    assert "Sample resume (Alex Morgan)" in html.split("data-compact-bar")[1]
    assert 'aria-label="Resume score' in html                    # treated like an uploaded resume
    assert "data-summary-career>Data Science</p>" in html


# --- loading steps -----------------------------------------------------------------

def test_loading_shows_the_four_steps_in_order(monkeypatch):
    loading = home(monkeypatch).split('id="loading"')[1].split("</section>")[0]
    labels = re.findall(r'<span class="loading-step-label">([^<]+)</span>', loading)
    assert labels == ["Reading your resume", "Matching 42 careers", "Scoring your resume", "Building your roadmap"]
    assert loading.count('data-step="pending"') == 4
    assert 'role="status" data-loading-status' in loading
    assert "skeleton" not in loading


def test_loading_respects_reduced_motion():
    css = app.app.test_client().get("/static/css/app.css").get_data(as_text=True)
    reduced = css.split("@media (prefers-reduced-motion: reduce)")[1].split("\n}")[0]
    assert ".spinner { animation: none !important; }" in reduced
    assert ".loading-progress span { animation: none !important;" in reduced
