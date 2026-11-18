import io

import app
from features.resume_check import NOT_A_RESUME, looks_like_resume, resume_sections

INVOICE = """INVOICE #4471
Sharma Traders, 12 MG Road, Pune
Bill to: Kumar Electricals
Date: 03-11-2026
Item                 Qty   Rate     Amount
Copper wire 2.5 mm   10    450.00   4500.00
Switch board         4     120.00   480.00
Subtotal 4980.00
GST 18% 896.40
Total 5876.40
Payment due within 15 days. Thank you for your business."""

with open("tests/fixtures/good_resume.txt", encoding="utf-8") as f:
    GOOD = f.read()


def test_an_invoice_is_not_a_resume():
    assert resume_sections(INVOICE) == set()
    assert not looks_like_resume(INVOICE)


def test_a_real_looking_resume_passes():
    assert len(resume_sections(GOOD)) >= 2
    assert looks_like_resume(GOOD)


def test_a_short_cv_with_two_sections_passes():
    assert looks_like_resume("SKILLS\nPython, SQL\nEDUCATION\nB.Tech CSE, 2026")


HEADINGLESS = ("Priya Nair, Kochi. B.Tech in Computer Science from CUSAT, 2026, CGPA 8.1. "
               "I built a library management web app in Python and Flask with a MySQL database, "
               "and a sales dashboard in Power BI that cleaned three years of shop data with pandas. "
               "During a two-month internship at a Kochi startup I wrote SQL reports and fixed bugs "
               "in a Django admin panel. I use Git every day, know HTML, CSS and JavaScript, and "
               "solved 200 problems on LeetCode. Looking for a fresher software or data role.")


def test_a_resume_without_headings_passes_on_skills_and_length():
    assert resume_sections(HEADINGLESS) == set()
    assert looks_like_resume(HEADINGLESS)


def test_blocked_only_when_skills_and_length_both_fail():
    assert looks_like_resume("Python, SQL and Excel, used in two college projects.")     # 2+ skills, short
    assert looks_like_resume("I enjoy building useful things for people. " * 12)          # 80+ words, no skill
    assert not looks_like_resume("Python only. A short note.")                            # 1 skill, short
    with open("tests/fixtures/weak_resume.txt", encoding="utf-8") as f:
        assert not looks_like_resume(f.read())                                          # no skill, 77 words


def test_the_home_upload_shows_the_message_instead_of_a_report():
    client = app.app.test_client()
    html = client.post("/", data={"resume_file": (io.BytesIO(INVOICE.encode()), "invoice.txt")},
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "This doesn&#39;t look like a resume" in html
    assert "Career report" not in html


def test_compare_rejects_a_second_file_that_is_not_a_resume():
    response = app.app.test_client().post(
        "/api/compare", data={"text": GOOD, "resume_file": (io.BytesIO(INVOICE.encode()), "invoice.txt")},
        content_type="multipart/form-data")
    assert response.status_code == 400 and response.get_json()["error"] == NOT_A_RESUME
