"""
Readiness verdict for the top career, and a warning when the resume can't separate the top two.

The verdict uses two things the page already shows: the match band of the top career
(strong / moderate / weak, from features/ranking.py) and the evidence tag of each of the career's
key skills (features/evidence.py). The thresholds below are DESIGN CHOICES, not measured from data.
"""
from features.evidence import PROVEN, STRENGTH, evidence_tags, heading_of
from features.resume_score import lines_of
from features.skills import CAREER_SKILLS, KNOWN_SKILLS, display_skill

# Design choice: about what one or two solid projects can show. Not measured.
SKILLS_NEEDED = 3
# Design choice: two matches closer than this many percentage points are "too close to call". Not measured.
CLOSE_GAP = 10
SECTIONS_THAT_HELP = ["Projects", "Experience", "Skills", "Education"]

READY, ONE_PROJECT, NOT_READY = "Ready to apply", "1 project away", "Not ready yet"
RULE = [
    (READY, f"A strong or moderate match, and at least {SKILLS_NEEDED} of the career's key skills are "
            "proven in a project, internship or job."),
    (ONE_PROJECT, f"A strong or moderate match, and at least {SKILLS_NEEDED} key skills appear, "
                  f"but fewer than {SKILLS_NEEDED} are proven."),
    (NOT_READY, f"A weak match, or fewer than {SKILLS_NEEDED} key skills appear in the resume."),
]


def join_words(words):
    """["A"] -> "A", ["A", "B"] -> "A and B", ["A", "B", "C"] -> "A, B and C"."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def next_step(level, career_label, have, proven, missing, tags):
    """The shortest way up one level, in one sentence (None when already ready)."""
    if level == READY:
        return None
    if level == ONE_PROJECT:
        # coursework skills first: they are closest to proven
        unproven = sorted((s for s in have if s not in proven), key=lambda s: -STRENGTH[tags[s]])
        picks = unproven[:SKILLS_NEEDED - len(proven)]
        names = join_words([f"{display_skill(s)} ({tags[s]})" for s in picks])
        return f"Use {names} in one project, so {'it is' if len(picks) == 1 else 'they are'} proven."
    if len(have) < SKILLS_NEEDED:
        picks = missing[:SKILLS_NEEDED - len(have)]
        return (f"Learn {join_words([display_skill(s) for s in picks])}, so the resume shows "
                f"{SKILLS_NEEDED} of the key skills for {career_label}.")
    # enough skills, but the match itself is weak
    return (f"Build one project that uses {join_words([display_skill(s) for s in have[:SKILLS_NEEDED]])}, "
            f"then list it under Projects so the resume reads clearly as {career_label}.")


def readiness(top_career, tags):
    """{"level", "career", "proven", "have", "total", "next_step"} for the top career.
    top_career is the first item of rank_careers; tags come from evidence_tags."""
    key_skills = CAREER_SKILLS.get(top_career["career"], [])
    have = [s for s in key_skills if s in tags]
    proven = [s for s in have if tags[s] == PROVEN]
    missing = [s for s in key_skills if s not in tags]
    if top_career["band"] != "weak" and len(proven) >= SKILLS_NEEDED:
        level = READY
    elif top_career["band"] != "weak" and len(have) >= SKILLS_NEEDED:
        level = ONE_PROJECT
    else:
        level = NOT_READY
    return {
        "level": level,
        "career": top_career["label"],
        "proven": len(proven),
        "have": len(have),
        "total": len(key_skills),
        "next_step": next_step(level, top_career["label"], have, proven, missing, tags),
    }


def low_information(top_careers, text):
    """A warning when the top two careers are under CLOSE_GAP points apart, else None."""
    if len(top_careers) < 2:
        return None
    first, second = top_careers[0], top_careers[1]
    if first["percent"] - second["percent"] >= CLOSE_GAP:
        return None
    # same heading test as the evidence tags, so a "Skills: Excel, Tally" line counts as a Skills section
    present = {heading_of(line) for line in lines_of(text)}
    missing = [name for name in SECTIONS_THAT_HELP if name not in present]
    if missing:
        hint = f"Add {'a ' if len(missing) == 1 else ''}{join_words(missing)} section{'s' if len(missing) > 1 else ''} with specific details."
    else:
        hint = "Add specific details to your projects and experience: tools used, what you built, results."
    return {
        "first": first["label"], "first_percent": first["percent"],
        "second": second["label"], "second_percent": second["percent"],
        "missing_sections": missing,
        "message": (f"Your top two matches are close ({first['label']} {first['percent']}%, "
                    f"{second['label']} {second['percent']}%), so this resume doesn't separate them yet. {hint}"),
    }


def resume_insights(text, top_careers, skill_gap):
    """Evidence tags, readiness verdict and low-information warning for an UPLOADED resume
    (the page and the PDF both call this, so they always agree)."""
    tags = evidence_tags(text, set(KNOWN_SKILLS) | set(skill_gap["have"]))
    return {"evidence": tags, "readiness": readiness(top_careers[0], tags),
            "low_info": low_information(top_careers, text)}
