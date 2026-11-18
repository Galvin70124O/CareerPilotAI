"""
Interview practice: 10 questions per career from interview.json, each with the key skill it tests
and a one-line tip. The page and the PDF show them for the top career.
"""
import json
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PROJECT_DIR, "interview.json"), encoding="utf-8") as f:
    QUESTIONS = json.load(f)["careers"]   # {career: [{"question", "skill", "tip"}, ...]}


def interview_questions(career):
    """The career's questions, or [] for a career without any (every profile career has 10)."""
    return QUESTIONS.get(career, [])
