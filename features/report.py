"""
"Download report (PDF)": a clean one-to-two page PDF of the results, made with fpdf2.

build_report_pdf(report) takes a dict with the same pieces the page shows
(see report_data in app.py) and returns the PDF as bytes.

fpdf2's built-in Helvetica font only knows Latin-1 characters, so text goes through
latin1() first: curly quotes, "…", "→" and bullets become plain characters.
"""
import datetime
import os

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from features.careers import display_name
from features.skills import display_skill

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(PROJECT_DIR, "static", "img", "logo-512.png")

# Colours (RGB), the same as the light theme in static/css/app.css
INDIGO = (79, 70, 229)
TEXT = (24, 24, 27)
MUTED = (82, 82, 91)
SUBTLE = (113, 113, 122)
BORDER = (228, 228, 231)
TRACK = (228, 228, 231)
GREEN = (21, 128, 61)
AMBER = (180, 83, 9)
RED = (185, 28, 28)

REPLACEMENTS = {"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "→": "->",
                "•": "-", "–": "-", "—": "-", "−": "-", "✓": "v", " ": " "}


def latin1(text):
    text = "".join(REPLACEMENTS.get(ch, ch) for ch in str(text))
    return text.encode("latin-1", "replace").decode("latin-1")


def level_color(ratio):
    # Same rule as the page: >= 90% green, 60-89% amber, < 60% red
    return GREEN if ratio >= 0.9 else AMBER if ratio >= 0.6 else RED


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*SUBTLE)
        self.cell(0, 5, latin1(f"CareerPilot AI  ·  Career data: ResumeAtlas (MIT)  ·  Page {self.page_no()}"), align="C")

    # --- small building blocks -------------------------------------------------------
    def ensure_space(self, height):
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def heading(self, text, keep_with=12):
        # Room for the heading AND the start of its section, so a heading never ends a page alone
        self.ensure_space(16 + keep_with)
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*INDIGO)
        self.cell(0, 7, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BORDER)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)

    def paragraph(self, text, size=9.5, color=MUTED, style=""):
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.multi_cell(0, 5, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bar_row(self, label, value_text, ratio, color, label_width=62):
        """One labelled progress bar, like the bars on the page."""
        self.ensure_space(8)
        y = self.get_y()
        self.set_font("Helvetica", size=9.5)
        self.set_text_color(*TEXT)
        self.cell(label_width, 6, latin1(label))
        bar_x, bar_w = self.l_margin + label_width, self.w - self.l_margin - self.r_margin - label_width - 22
        self.set_fill_color(*TRACK)
        self.rect(bar_x, y + 2, bar_w, 2.2, style="F")
        self.set_fill_color(*color)
        self.rect(bar_x, y + 2, max(0.6, bar_w * max(0.0, min(ratio, 1.0))), 2.2, style="F")
        self.set_xy(bar_x + bar_w, y)
        self.set_text_color(*MUTED)
        self.cell(22, 6, latin1(value_text), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def labelled_list(self, label, items):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*TEXT)
        self.cell(32, 5.5, latin1(label))
        self.set_font("Helvetica", size=9.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 5.5, latin1(", ".join(items) if items else "-"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def mark(self, done):
        """A small filled (done) or empty (to do) circle in front of a roadmap step."""
        x, y = self.get_x(), self.get_y()
        if done:
            self.set_fill_color(*GREEN)
            self.ellipse(x + 1, y + 1.4, 2.6, 2.6, style="F")
        else:
            self.set_draw_color(*SUBTLE)
            self.ellipse(x + 1, y + 1.4, 2.6, 2.6, style="D")
        self.set_x(x + 6)


def build_report_pdf(report):
    pdf = ReportPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_title("CareerPilot AI report")
    pdf.set_author("CareerPilot AI")
    pdf.add_page()

    # Header: logo, title, what was analyzed
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, x=18, y=15, w=11)
    pdf.set_xy(32, 15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*TEXT)
    pdf.cell(0, 6.5, "CareerPilot AI report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(32)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*SUBTLE)
    today = datetime.date.today().strftime("%d %B %Y")
    pdf.cell(0, 5, latin1(f"{report['source_label']}  ·  {today}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Career matches
    careers = report["top_careers"]
    best = careers[0]
    pdf.heading("Career matches")
    pdf.paragraph(f"Recommended career path: {best['label']} ({best['percent']}% confidence)", size=10.5, color=TEXT, style="B")
    pdf.ln(1)
    for i, career in enumerate(careers):
        pdf.bar_row(f"{i + 1}. {career['label']}", f"{career['percent']}%", career["percent"] / 100,
                    INDIGO if i == 0 else (165, 180, 252))
    if report.get("why_terms"):
        pdf.ln(1)
        pdf.paragraph("Why this match - words that pushed it most: "
                      + ", ".join(display_skill(t) for t in report["why_terms"]), size=9)

    # Resume score (uploaded resumes only)
    score = report.get("resume_score")
    if score:
        verdict = "Strong" if score["total"] >= 75 else "Needs work" if score["total"] >= 50 else "Weak"
        pdf.heading("Resume score")
        pdf.paragraph(f"{score['total']} / 100  ({verdict})", size=13, color=level_color(score["total"] / 100), style="B")
        pdf.ln(1)
        for c in score["categories"]:
            pdf.bar_row(c["label"], f"{c['score']} / {c['max']}", c["score"] / c["max"], level_color(c["score"] / c["max"]))
        pdf.ln(2)
        pdf.paragraph("Tips to improve", size=10, color=TEXT, style="B")
        for i, tip in enumerate(score["tips"], 1):
            pdf.ensure_space(10)
            pdf.paragraph(f"{i}.  {tip}", size=9.5)
            pdf.ln(0.5)

    # Skill gap
    gap = report.get("skill_gap")
    if gap and gap["total"]:
        pdf.heading(f"Skill gap for {display_name(gap['career'])}  ({len(gap['have'])} / {gap['total']} key skills)")
        pdf.labelled_list("You have:", [display_skill(s) for s in gap["have"]])
        pdf.labelled_list("To learn next:", [display_skill(s) for s in gap["missing"]])

    # Roadmap
    roadmap = report.get("roadmap")
    if roadmap:
        pdf.heading(f"Roadmap to {display_name(roadmap['career'])}  ({roadmap['done']} / {roadmap['total']} steps done)", keep_with=22)
        for stage in roadmap["stages"]:
            pdf.ensure_space(22)
            pdf.paragraph(stage["name"], size=10, color=TEXT, style="B")
            for step in stage["steps"]:
                pdf.mark(step["done"])
                pdf.set_font("Helvetica", size=9.5)
                pdf.set_text_color(*(SUBTLE if step["done"] else TEXT))
                pdf.multi_cell(0, 5, latin1(step["step"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

    # Courses
    courses = report.get("courses")
    if courses:
        pdf.heading("Free resources for your missing skills", keep_with=7 + 5 * len(courses[0]["resources"]))
        for item in courses:
            pdf.ensure_space(7 + 5 * len(item["resources"]))   # keep a skill and its links together
            pdf.paragraph(display_skill(item["skill"]), size=9.5, color=TEXT, style="B")
            for resource in item["resources"]:
                pdf.set_font("Helvetica", "U", 9)
                pdf.set_text_color(*INDIGO)
                pdf.cell(0, 5, latin1(resource["title"]), link=resource["url"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(0.5)

    return bytes(pdf.output())
