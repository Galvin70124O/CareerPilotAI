"""Download report (PDF)."""
import io
import re

from pypdf import PdfReader

import app
from features.report import latin1


def post_report(data):
    return app.app.test_client().post("/report", data=data)


def pdf_text(response):
    reader = PdfReader(io.BytesIO(response.data))
    return len(reader.pages), "\n".join(page.extract_text() for page in reader.pages)


def sample_text():
    with open(app.SAMPLE_RESUME_PATH, encoding="utf-8") as f:
        return f.read()


def test_resume_report_is_a_one_or_two_page_pdf_with_every_section():
    response = post_report({"text": sample_text(), "is_resume": "1", "name": "cv.txt"})
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.headers["Content-Disposition"] == 'attachment; filename="careerpilot-report.pdf"'
    assert response.data.startswith(b"%PDF")
    pages, text = pdf_text(response)
    assert 1 <= pages <= 2
    for section in ["CareerPilot AI report", "Resume: cv.txt", "Career matches", "Data Science", "Resume score",
                    "Tips to improve", "Skill gap for Data Science", "Roadmap to Data Science",
                    "Free resources for your missing skills"]:
        assert section in text, section


def test_typed_skills_report_has_no_resume_score():
    pages, text = pdf_text(post_report({"text": "java, spring, hibernate", "is_resume": "0"}))
    assert "Typed skills" in text and "Java Developer" in text
    assert "Resume score" not in text


def test_report_links_to_courses():
    response = post_report({"text": "python, sql", "is_resume": "0"})
    assert b"/URI" in response.data                               # clickable course links


def test_report_rejects_bad_input():
    assert post_report({"text": ""}).status_code == 400
    assert post_report({"text": "x" * 100_001}).status_code == 400


def test_non_latin_characters_do_not_break_the_pdf():
    assert latin1("“Built…” → • ok") == '"Built..." -> - ok'
    response = post_report({"text": "• Built “tools” → Python, SQL… ✓ 日本語", "is_resume": "1", "name": "résumé.pdf"})
    assert response.status_code == 200 and response.data.startswith(b"%PDF")


def test_download_button_in_the_results_bar(monkeypatch):
    monkeypatch.setattr(app, "fetch_job_listings", lambda title, limit=5: [])
    html = app.app.test_client().post("/", data={"sample": "1"}).get_data(as_text=True)
    bar = html.split("data-compact-bar")[1].split('id="analyze-panel"')[0]
    form = re.search(r'<form method="POST" action="/report".*?</form>', bar, re.S).group(0)
    assert 'name="text"' in form and '<input type="hidden" name="is_resume" value="1">' in form
    assert "Download report (PDF)" in form


def test_no_section_heading_ends_a_page_alone():
    response = post_report({"text": sample_text(), "is_resume": "1", "name": "cv.txt"})
    reader = PdfReader(io.BytesIO(response.data))
    headings = ["Career matches", "Resume score", "Tips to improve", "Free resources for your missing skills"]
    for page in reader.pages:
        lines = [line for line in page.extract_text().splitlines() if line.strip() and "Page " not in line]
        assert not any(lines[-1].startswith(h) for h in headings), lines[-1]
