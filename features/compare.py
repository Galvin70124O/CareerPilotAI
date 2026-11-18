"""
"Compare with another version": the page's own analysis for two versions of a resume, side by side,
with the change. Nothing is stored.

Both versions are measured against version 1's top career: the score's skill line, the skill gap, the
evidence tags and the readiness verdict all depend on the career, so a change in them then means the
resume changed, not the career it was compared with. Each version's own top 3 is still shown.
"""
from features.careers import display_name
from features.evidence import STRENGTH
from features.ranking import band
from features.readiness import NOT_READY, ONE_PROJECT, READY, resume_insights
from features.resume_score import score_resume
from features.skills import CAREER_SKILLS, analyze_skill_gap, display_skill

LEVELS = [NOT_READY, ONE_PROJECT, READY]   # worst to best
SAME = "same"
RANKS = ["1st", "2nd", "3rd"]


def snapshot(model, text, top_careers, career):
    """One version's results, all measured against `career` (version 1's top career)."""
    probability = model.predict_proba([text])[0][list(model.classes_).index(career)]
    percent = round(probability * 100)      # rounded like rank_careers, so version 1 shows the same number
    target = {"career": career, "label": display_name(career), "percent": percent, "band": band(percent)}
    gap = analyze_skill_gap(career, text)
    # [target] alone: readiness is worked out for this career, and the close-matches note is not needed
    insights = resume_insights(text, [target], gap)
    return {"top_careers": top_careers, "target": target, "score": score_resume(text, career),
            "gap": gap, "evidence": insights["evidence"], "readiness": insights["readiness"]}


def number_change(before, after):
    """"+3", "−2" (a real minus sign) or "same"."""
    if after == before:
        return SAME
    return f"+{after - before}" if after > before else f"−{before - after}"


def row(label, before, after, change):
    return {"label": label, "before": before, "after": after, "change": change, "changed": change != SAME}


def score_rows(before, after):
    rows = [row("Total", f"{before['total']} / 100", f"{after['total']} / 100",
                number_change(before["total"], after["total"]))]
    for old, new in zip(before["categories"], after["categories"]):     # same categories, same order
        rows.append(row(old["label"], f"{old['score']} / {old['max']}", f"{new['score']} / {new['max']}",
                        number_change(old["score"], new["score"])))
    return rows


def career_rows(before, after):
    target_before, target_after = before["target"], after["target"]
    rows = [row(f"Match with {target_before['label']}",
                f"{target_before['percent']}%, {target_before['band']}", f"{target_after['percent']}%, {target_after['band']}",
                number_change(target_before["percent"], target_after["percent"]))]
    for rank, old, new in zip(RANKS, before["top_careers"], after["top_careers"]):
        if old["career"] == new["career"]:
            change = number_change(old["percent"], new["percent"])
        else:
            change = "changed"
        rows.append(row(f"{rank} match", f"{old['label']} {old['percent']}%", f"{new['label']} {new['percent']}%", change))
    return rows


def skill_list(skills):
    return ", ".join(display_skill(skill) for skill in skills) or "none"


def gap_rows(before, after):
    old, new = before["gap"], after["gap"]
    found = [skill for skill in new["have"] if skill not in old["have"]]
    lost = [skill for skill in old["have"] if skill not in new["have"]]
    # the before/after cells already name the skills, so the change only counts them
    change = ", ".join(part for part in [f"found {len(found)}" if found else "",
                                         f"lost {len(lost)}" if lost else ""] if part) or SAME
    return [row("Key skills in your resume", f"{len(old['have'])} of {old['total']}", f"{len(new['have'])} of {new['total']}",
                number_change(len(old["have"]), len(new["have"]))),
            row("Missing key skills", skill_list(old["missing"]), skill_list(new["missing"]), change)]


def tag_change(old, new):
    if old == new:
        return SAME
    if old is None:
        return "new"
    if new is None:
        return "gone"
    return "stronger" if STRENGTH[new] > STRENGTH[old] else "weaker"


def evidence_rows(before, after):
    """'Proven key skills', then one row per key skill whose evidence tag changed."""
    career = before["target"]["career"]
    old_proven, new_proven = before["readiness"]["proven"], after["readiness"]["proven"]
    rows = [row("Proven key skills", f"{old_proven} of {before['readiness']['total']}",
                f"{new_proven} of {after['readiness']['total']}", number_change(old_proven, new_proven))]
    for skill in CAREER_SKILLS.get(career, []):
        old, new = before["evidence"].get(skill), after["evidence"].get(skill)
        if old != new:
            rows.append(row(display_skill(skill), old or "not found", new or "not found", tag_change(old, new)))
    return rows


def readiness_rows(before, after):
    old, new = before["readiness"]["level"], after["readiness"]["level"]
    if old == new:
        change = SAME
    else:
        change = "up" if LEVELS.index(new) > LEVELS.index(old) else "down"
    return [row("Verdict", old, new, change)]


def compare_versions(before, after):
    """The two snapshots as sections of {label, before, after, change} rows, for the page."""
    career = before["target"]["label"]
    sections = [
        {"title": "Resume Score", "rows": score_rows(before["score"], after["score"])},
        {"title": "Career match", "rows": career_rows(before, after)},
        {"title": f"Skill gap for {career}", "rows": gap_rows(before, after)},
        {"title": "Evidence", "rows": evidence_rows(before, after)},
        {"title": "Readiness", "rows": readiness_rows(before, after)},
    ]
    return {
        "career": career,
        "top_changed": before["top_careers"][0]["career"] != after["top_careers"][0]["career"],
        "unchanged": not any(r["changed"] for section in sections for r in section["rows"]),
        "sections": sections,
    }
