"""
Job description match: how well this resume fits ONE job description the user pastes.

    match % = WORDS_WEIGHT x word overlap  +  SKILLS_WEIGHT x share of the JD's skills found in the resume

- word overlap: cosine similarity of the two TF-IDF vectors, made by the career model's own
  vectorizer. TF-IDF gives each word a weight (rare, specific words count more); the cosine is 1.0
  for the same words in the same mix and 0 when no word is shared.
- JD skills: every skill name the app knows (and its synonyms, so "ReactJS" counts as React).
  If the JD names no skill we know, the match is word overlap alone.

Then, for the JD skills the resume lacks or does not prove (Phase 1 evidence tags), ranked by how
many sentences of the JD mention them, it points at up to 3 Projects / Experience lines where the
skill could be named, IF the work really used it. Nothing is stored; everything is worked out per request.
"""
import re

from features.evidence import CLAIMED, COURSEWORK, PROVEN, evidence_tags, heading_of, line_tags
from features.ranking import PROFILES
from features.skills import CAREER_SKILLS, KNOWN_SKILLS, SYNONYMS, display_skill, has_skill

MAX_JD_LENGTH = 8000
WORDS_WEIGHT = 0.5    # design choice: words and named skills count the same
SKILLS_WEIGHT = 0.5
MAX_SUGGESTIONS = 3
MIN_LINE_WORDS = 4    # shorter lines ("Intern, Acme") are titles, not descriptions of the work
MISSING = "missing"
BULLETS = "-*•·–— \t"   # stripped from the start of a suggested line
# How each kind of gap is described, on the page and in the PDF
GAP_NOTES = {MISSING: "not in your resume", CLAIMED: "you list it but never prove it",
             COURSEWORK: "only in your coursework"}

# Skills in some career, but in a job ad they usually mean something else:
# "our organization", "state of the art", "reporting to the manager", "training is provided", "Benefits"
NOT_JD_SKILLS = {"organization", "art", "benefits", "responsive", "reporting", "training", "onboarding",
                 "quality", "operations", "portfolio", "content"}
JD_SKILLS = sorted((set(KNOWN_SKILLS)
                    | {skill for skills in CAREER_SKILLS.values() for skill in skills}
                    | {skill["name"] for profile in PROFILES.values() for skill in profile["key_skills"]}
                    | set(SYNONYMS)) - NOT_JD_SKILLS)


def sentences(text):
    """The JD's lines, split again at sentence ends ("Use Git. Write tests." is two)."""
    return [part for part in re.split(r"\n|(?<=[.!?;])\s+", text) if part.strip()]


def mentioned_in(skills, parts):
    """{skill: [indexes of the JD sentences that name it]}. Counting sentences (not words) keeps a
    skill repeated in one line from counting twice."""
    where = {skill: [] for skill in skills}
    for index, sentence in enumerate(parts):
        for skill in skills:              # sentence outside, skill inside: has_skill caches each sentence
            if has_skill(skill, sentence):
                where[skill].append(index)
    return where


def word_overlap(vectorizer, a, b):
    rows = vectorizer.transform([a, b])   # each row has length 1, so their dot product is the cosine
    return float(rows[0].multiply(rows[1]).sum())


def suggest_lines(vectorizer, resume, parts, where, gaps):
    """Up to MAX_SUGGESTIONS (gap, line) pairs: for each gap, the Projects / Experience line most like
    the JD sentences that name that skill. Each line is used once; lines sharing no word are skipped."""
    lines = [line for line, tag in line_tags(resume)
             if tag == PROVEN and heading_of(line) is None and len(line.split()) >= MIN_LINE_WORDS]
    if not lines:
        return []
    line_vectors = vectorizer.transform(lines)
    used, picks = set(), []
    for gap in gaps:
        context = " ".join(parts[i] for i in where[gap["skill"]]) or gap["skill"]
        scores = (line_vectors @ vectorizer.transform([context]).T).toarray().ravel()
        best = max((i for i in range(len(lines)) if i not in used and scores[i] > 0),
                   key=lambda i: scores[i], default=None)    # max keeps the first on a tie
        if best is not None:
            used.add(best)
            picks.append({"skill": gap["skill"], "label": gap["label"], "line": lines[best].lstrip(BULLETS)})
        if len(picks) == MAX_SUGGESTIONS:
            break
    return picks


def match_job(vectorizer, resume, jd, is_resume):
    """The match % with its two parts, the JD skills found / to work on, and line suggestions."""
    jd_skills = [skill for skill in JD_SKILLS if has_skill(skill, jd)]
    found = [skill for skill in jd_skills if has_skill(skill, resume)]
    # Typed skills have no sections, so they get no evidence tags: a found skill is simply found
    tags = evidence_tags(resume, found) if is_resume else {}
    parts = sentences(jd)
    where = mentioned_in(jd_skills, parts)

    gaps = []
    for skill in jd_skills:
        tag = tags.get(skill) if skill in found else MISSING
        if tag in (MISSING, CLAIMED, COURSEWORK):        # proven skills need no work
            # a skill matched only across a line break (rare) still counts as one mention
            gaps.append({"skill": skill, "label": display_skill(skill), "tag": tag, "note": GAP_NOTES[tag],
                         "mentions": len(where[skill]) or 1})
    first_mention = {skill: (where[skill] or [len(parts)])[0] for skill in jd_skills}
    gaps.sort(key=lambda gap: (-gap["mentions"], first_mention[gap["skill"]]))

    words = word_overlap(vectorizer, resume, jd)
    skills_share = len(found) / len(jd_skills) if jd_skills else None
    score = words if skills_share is None else WORDS_WEIGHT * words + SKILLS_WEIGHT * skills_share
    return {
        "percent": round(score * 100),
        "words_percent": round(words * 100),
        "skills_percent": None if skills_share is None else round(skills_share * 100),
        "skills_total": len(jd_skills),
        "found": [display_skill(skill) for skill in found],
        "gaps": gaps,
        "suggestions": suggest_lines(vectorizer, resume, parts, where, gaps) if is_resume else [],
        "is_resume": is_resume,
    }
