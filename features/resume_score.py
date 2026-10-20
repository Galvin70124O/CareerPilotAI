"""
ATS-style Resume Score (0-100).

score_resume(text, career) looks at the RAW resume text (with punctuation, line breaks
and bullets) and returns the total score, a per-category breakdown, a checklist and tips.

Change SCORE_WEIGHTS to make a category count more or less. They should add up to 100.
"""
import re

from features.careers import display_name
from features.skills import analyze_skill_gap, display_skill

SCORE_WEIGHTS = {
    "sections": 20,       # education, skills, projects, experience, contact, links
    "length": 15,         # word count, bullet points, not too dense
    "action_verbs": 15,   # built, developed, led...
    "impact": 15,         # numbers, %, results
    "skill_match": 25,    # key skills of the predicted career (skills.json)
    "red_flags": 10,      # missing email/phone, very long lines, too much "I"
}

CATEGORY_LABELS = {
    "sections": "Sections",
    "length": "Length & readability",
    "action_verbs": "Action verbs",
    "impact": "Measurable impact",
    "skill_match": "Skill match",
    "red_flags": "Red flags",
}

# Section headings: a line that STARTS with one of these words
SECTION_HEADINGS = {
    "Education": r"education|academic|qualifications?",
    "Skills": r"(technical\s+)?skills|core\s+competencies|technologies",
    "Projects": r"(academic\s+|personal\s+|key\s+)?projects?",
    "Experience": r"(work\s+|professional\s+)?experience|employment|work\s+history|internships?",
}

ACTION_VERBS = [
    "achieved", "analyzed", "automated", "built", "collaborated", "created", "delivered", "designed",
    "developed", "directed", "engineered", "established", "executed", "founded", "generated", "implemented",
    "improved", "increased", "initiated", "launched", "led", "managed", "mentored", "optimized", "organized",
    "planned", "presented", "published", "reduced", "researched", "resolved", "saved", "streamlined",
    "supervised", "tested", "trained", "won",
]
ACTION_VERBS_FOR_FULL_SCORE = 8

# Bullets that start weakly: suggest a strong verb instead
WEAK_STARTS = r"(responsible\s+for|worked\s+on|helped|assisted\s+(with|in)|involved\s+in|duties\s+included)"

EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE = r"(\+?\d[\d\s().-]{8,}\d)"
LINKS = r"(github\.com|linkedin\.com)"
BULLET = r"^\s*([-*•●▪■◦‣–]|\d+[.)])\s+"
# A "result": a percentage, money, or a number next to a result word
METRIC = (r"\d+(\.\d+)?\s?%|[$₹€£]\s?\d|\b\d+(\.\d+)?\s?(k|m|x)\b|"
          r"\b\d[\d,]*\+?\s+(users|customers|clients|students|people|members|downloads|projects|"
          r"hours|days|weeks|requests|transactions|sales|leads|teams?)\b")

IDEAL_WORDS = (300, 900)
LONG_LINE_CHARS = 200
MAX_I_PER_100_WORDS = 2


def lines_of(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def short_quote(line, max_chars=70):
    line = re.sub(BULLET, "", line).strip()
    return line if len(line) <= max_chars else line[:max_chars].rstrip() + "…"


def has_phone(text):
    # A phone number has 10-15 digits; this stops date ranges like "2021 - 2025" from counting
    for match in re.finditer(PHONE, text):
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 15:
            return True
    return False


def check_sections(text):
    checks = {}
    for name, pattern in SECTION_HEADINGS.items():
        # ^ with re.M = start of any line; the heading line must be short (a real heading)
        checks[name] = re.search(r"^\W*(" + pattern + r")\b.{0,30}$", text, re.I | re.M) is not None
    checks["Contact info"] = bool(re.search(EMAIL, text) or has_phone(text))
    checks["GitHub/LinkedIn link"] = re.search(LINKS, text, re.I) is not None
    return checks


def score_resume(text, career):
    lines = lines_of(text)
    words = re.findall(r"[A-Za-z][\w'+#.-]*", text)
    word_count = len(words)
    fractions = {}     # category -> 0.0 .. 1.0
    checklist = []     # {"text": ..., "ok": True/False}
    tips = []          # (points it could win back, tip text)

    # 1) Sections
    sections = check_sections(text)
    fractions["sections"] = sum(sections.values()) / len(sections)
    for name, found in sections.items():
        checklist.append({"text": f"{name} section" if name in SECTION_HEADINGS else name, "ok": found})
    missing_sections = [name for name, found in sections.items() if not found]
    if missing_sections:
        points = SCORE_WEIGHTS["sections"] * len(missing_sections) / len(sections)
        tips.append((points, "Add: " + ", ".join(missing_sections) + ". ATS systems look for clear headings like these."))

    # 2) Length & readability
    bullets = [line for line in lines if re.match(BULLET, line)]
    long_paragraphs = [line for line in lines if len(line.split()) > 40]
    good_length = IDEAL_WORDS[0] <= word_count <= IDEAL_WORDS[1]
    okay_length = IDEAL_WORDS[0] / 2 <= word_count <= IDEAL_WORDS[1] * 1.3
    length_part = 1 if good_length else 0.5 if okay_length else 0
    bullet_part = 1 if len(bullets) >= 5 else 0.5 if len(bullets) >= 2 else 0
    dense_part = 1 if not long_paragraphs else 0
    fractions["length"] = (length_part + bullet_part + dense_part) / 3
    checklist.append({"text": f"Length {IDEAL_WORDS[0]}-{IDEAL_WORDS[1]} words (yours: {word_count})", "ok": good_length})
    checklist.append({"text": f"Uses bullet points (yours: {len(bullets)})", "ok": len(bullets) >= 5})
    checklist.append({"text": "No dense paragraphs over 40 words", "ok": not long_paragraphs})
    if long_paragraphs:
        tips.append((SCORE_WEIGHTS["length"] / 3,
                     f'Split this long paragraph into 2-3 bullet points: "{short_quote(long_paragraphs[0])}"'))
    elif len(bullets) < 5:
        tips.append((SCORE_WEIGHTS["length"] / 3 * (1 - bullet_part),
                     "Use bullet points for experience and projects; recruiters scan, they don't read."))
    if not good_length:
        advice = "too short: add projects and results" if word_count < IDEAL_WORDS[0] else "too long: cut older or less relevant details"
        tips.append((SCORE_WEIGHTS["length"] / 3 * (1 - length_part), f"Your resume has {word_count} words, {advice}."))

    # 3) Action verbs
    text_lower = text.lower()
    verbs_used = [verb for verb in ACTION_VERBS if re.search(r"\b" + verb + r"\b", text_lower)]
    fractions["action_verbs"] = min(len(verbs_used) / ACTION_VERBS_FOR_FULL_SCORE, 1)
    checklist.append({"text": f"Strong action verbs (yours: {len(verbs_used)}, aim for {ACTION_VERBS_FOR_FULL_SCORE}+)",
                      "ok": len(verbs_used) >= ACTION_VERBS_FOR_FULL_SCORE})
    weak_lines = [line for line in lines if re.match(r"^\W*" + WEAK_STARTS, line, re.I)]
    if weak_lines:
        tips.append((SCORE_WEIGHTS["action_verbs"] * (1 - fractions["action_verbs"]) + 2,
                     f'Start with a strong verb instead of "{short_quote(weak_lines[0], 50)}", '
                     f'e.g. "Built…", "Developed…" or "Led…".'))
    elif fractions["action_verbs"] < 1:
        tips.append((SCORE_WEIGHTS["action_verbs"] * (1 - fractions["action_verbs"]),
                     "Begin bullets with action verbs like built, developed, led, improved, automated."))

    # 4) Measurable impact
    metric_lines = [line for line in lines if re.search(METRIC, line, re.I)]
    fractions["impact"] = min(len(metric_lines) / 5, 1)
    checklist.append({"text": f"Measurable results: numbers or % (yours: {len(metric_lines)}, aim for 5+)",
                      "ok": len(metric_lines) >= 5})
    if fractions["impact"] < 1:
        # Quote a bullet (or any longer line) that has no number yet
        candidates = bullets or [line for line in lines if len(line.split()) >= 4]
        no_number = [line for line in candidates if not re.search(r"\d", line)]
        example = f' For example: "{short_quote(no_number[0], 60)}": how many users, how much faster, what %?' if no_number else ""
        tips.append((SCORE_WEIGHTS["impact"] * (1 - fractions["impact"]),
                     "Add numbers that show results." + example))

    # 5) Skill match with the predicted career
    gap = analyze_skill_gap(career, text)
    fractions["skill_match"] = gap["percent"] / 100
    checklist.append({"text": f"Key skills for {display_name(career)}: {len(gap['have'])}/{gap['total']}", "ok": gap["percent"] >= 50})
    if gap["missing"]:
        tips.append((SCORE_WEIGHTS["skill_match"] * len(gap["missing"]) / max(gap["total"], 1),
                     f"If you know them, mention these {display_name(career)} skills: {', '.join(display_skill(s) for s in gap['missing'][:4])}."))

    # 6) Red flags
    has_email = re.search(EMAIL, text) is not None
    phone_found = has_phone(text)
    very_long_lines = [line for line in lines if len(line) > LONG_LINE_CHARS]
    i_count = len(re.findall(r"\b(I|me|my)\b", text))
    too_much_i = word_count > 0 and i_count / word_count * 100 > MAX_I_PER_100_WORDS
    red_flag_checks = [has_email, phone_found, not very_long_lines, not too_much_i]
    fractions["red_flags"] = sum(red_flag_checks) / len(red_flag_checks)
    checklist.append({"text": "Email address", "ok": has_email})
    checklist.append({"text": "Phone number", "ok": phone_found})
    checklist.append({"text": f"No very long lines (over {LONG_LINE_CHARS} characters)", "ok": not very_long_lines})
    checklist.append({"text": f'Not overusing "I/me/my" (yours: {i_count})', "ok": not too_much_i})
    flag_points = SCORE_WEIGHTS["red_flags"] / len(red_flag_checks)
    if not has_email or not phone_found:
        tips.append((flag_points * 2, "Put your email and phone number at the top; without them a recruiter can't contact you."))
    if too_much_i:
        i_line = next((line for line in lines if re.search(r"\bI\b", line)), "")
        tips.append((flag_points, f'Drop "I" and start with the verb: "{short_quote(i_line, 50)}" → "Built…"'))
    if very_long_lines and not long_paragraphs:
        tips.append((flag_points, f'Shorten this very long line: "{short_quote(very_long_lines[0], 50)}"'))

    # Total and breakdown
    categories = []
    for key, weight in SCORE_WEIGHTS.items():
        categories.append({"key": key, "label": CATEGORY_LABELS[key],
                           "score": round(fractions[key] * weight), "max": weight})
    total = round(sum(fractions[key] * weight for key, weight in SCORE_WEIGHTS.items()))

    # Biggest wins first, 3-5 tips
    tips = [tip for _, tip in sorted(tips, key=lambda pair: pair[0], reverse=True)][:5]
    for extra in ["Tailor your resume to each job: reuse words from the job description.",
                  "Keep the layout simple (one column, standard fonts) so ATS software can read it.",
                  "Put your strongest project or achievement near the top."]:
        if len(tips) >= 3:
            break
        tips.append(extra)

    return {"total": total, "categories": categories, "checklist": checklist, "tips": tips}
