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


def test_one_section_needs_skills_and_enough_words():
    skills_line = "Skills: Python, SQL, Excel"
    assert not looks_like_resume(skills_line)                                   # too short
    long_text = skills_line + "\n" + "I enjoy building useful things for people. " * 12
    assert looks_like_resume(long_text)                                          # skills + 80 words
    assert not looks_like_resume("Skills:\n" + "nothing technical here at all " * 20)   # no known skill


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
