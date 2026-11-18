"""
Build skill_synonyms.json: other ways people write each skill the app checks for.

    python data/fetch_sources.py     # first: downloads O*NET + ESCO into data/raw/
    python data/build_synonyms.py

Three sources, kept apart in the file so you can see where every synonym came from:
- "esco":  ESCO v1.2.1 alternative labels ("Python 3K" for python), European Commission
- "onet":  O*NET 31.0 software names, only in two safe shapes:
           "<known vendor> <skill>"     ("Microsoft Excel", "Autodesk AutoCAD")
           "<long form> <ABBREVIATION>" ("Structured query language SQL" -> "structured query language")
- "india": hand-written list of how Indian resumes write things (B.Tech, NPTEL, CGPA, DSA, internship)

Short names (R, C, Go, JS) are only ever matched as whole words; R, Go and B.E. need extra context.
Those rules are code, in features/skills.py (SPECIAL_SKILL_PATTERNS), not synonyms.
"""
import csv
import json
import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_DIR)
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT = os.path.join(PROJECT_DIR, "skill_synonyms.json")

# O*NET names "<vendor> <product>". Only these vendors: a longer prefix usually means a different
# product that merely ends with the skill's name ("Beck Depression Inventory" is not inventory).
ONET_VENDORS = {
    "microsoft", "autodesk", "adobe", "atlassian", "dassault systemes", "the mathworks", "oracle",
    "sun microsystems", "intuit", "sap", "red hat enterprise", "redgate", "apache", "google", "ibm",
    "amazon web services", "salesforce.com", "tableau", "bentley", "ptc", "siemens", "cisco",
}
# O*NET long forms followed by the abbreviation the app uses ("Hypertext markup language HTML")
ONET_LONG_FORM = re.compile(r"^([a-z][a-z ]+[a-z]) ([a-z+#/.]{2,6})$")

# ESCO alternative labels we never use as synonyms: too general, or they mean something else.
# Found by reading every ESCO label the builder kept (e.g. statistics -> "metadata",
# javascript -> "mocha", which in a food resume is coffee).
ESCO_BLOCKLIST = {
    "programming", "software", "computer", "design", "testing", "analysis", "management", "data",
    "communication", "research", "teaching", "planning", "reporting", "sales", "marketing",
    # civil engineering
    "city planning", "urban planning", "civil engineering studies", "civil engineering research",
    # communication
    "correspondence", "interaction", "conversation", "communicating",
    # statistics
    "metadata", "survey data", "data analysis", "data work", "use of data", "big data",
    "administrative data", "experimental data", "organisation of data", "data collection",
    "use of statistics", "statistical analysis",
    # javascript, sql
    "mocha", "escript", "live script", "csjs", "sequel", "subquery", "informix 4gl/sql",
    # others
    "transport", "diet", "public image", "network magic", "customer well being", "customer satisfaction",
    "architectural project", "architectural proposal", "design architecture",
    # new-career skills: "transformers" is also an NLP model, "generators" and "survey" are everyday words
    "transformers", "generators", "survey",
}
MAX_WORDS = 4         # longer ESCO labels are descriptions, not words people write on a resume
MIN_LENGTH = 3        # shorter ones ("r", "go") are handled by context rules in features/skills.py

# How Indian resumes write things. Keys are skill names used in career_profiles.json / skills.json.
INDIA = {
    "b.e./b.tech": [],   # matched by a context rule in features/skills.py, not by synonyms
    "nptel": ["swayam nptel", "nptel certification", "nptel online certification"],
    "cgpa": ["sgpa", "cgpa/percentage"],   # not "cpi": in finance resumes that is Consumer Price Index
    "data structures and algorithms": ["dsa", "data structures", "data structure", "ds and algo",
                                       "ds & algo", "algorithms and data structures"],
    "internship": ["intern", "summer internship", "winter internship", "industrial training",
                   "in-plant training", "inplant training", "summer training", "apprenticeship"],
    "javascript": ["js"],
    "react": ["reactjs", "react js"],
    "excel": ["ms excel", "ms-excel", "advanced excel"],
    "powerpoint": ["ms powerpoint", "ms-powerpoint", "ppt"],
    "autocad": ["auto cad", "autocad 2d", "autocad 3d"],
    "staad pro": ["staad.pro", "staad"],
    "power bi": ["powerbi", "power-bi"],
    "manual testing": ["manual tester"],
    # skill names used by the new careers in career_profiles.json
    "object-oriented programming": ["oops", "oop", "object oriented programming", "oops concepts"],
    "debugging": ["debug", "debugged"],
    "data cleaning": ["data cleansing", "data wrangling", "data preprocessing", "data pre-processing"],
    "data visualization": ["data visualisation", "data viz"],
    "dashboards": ["dashboard"],
    "bug reporting": ["bug report", "bug reports", "defect reporting", "defect tracking", "bug tracking",
                      "defect logging", "defect reports"],
    "unit testing": ["unit test", "unit tests"],
    "ticketing": ["ticketing system", "ticketing tool", "servicenow", "zendesk", "freshdesk"],
    "help desk": ["helpdesk", "service desk"],
    "engineering drawing": ["engineering drawings", "technical drawing", "machine drawing"],
    "strength of materials": ["mechanics of materials", "mechanics of solids"],
    "creo": ["ptc creo", "creo parametric"],
    "circuit analysis": ["circuit diagrams", "circuit design", "circuit theory"],
    "electrical machines": ["electric machines", "electric motors", "induction motors"],
    "power systems": ["power system", "power distribution", "power transmission"],
    "electrical wiring": ["wiring diagrams", "house wiring", "panel wiring"],
    "electrical testing": ["megger testing", "insulation resistance testing", "testing and commissioning"],
    "electrical safety": ["electrical safety norms", "lockout/tagout", "lockout tagout"],
    "quantity surveying": ["quantity estimation", "estimation and costing", "boq", "bill of quantities",
                           "rate analysis"],
    "surveying": ["total station", "auto level", "land surveying"],
    "construction methods": ["construction techniques", "construction technology"],
    "site supervision": ["site execution", "site inspection", "site supervisor"],
    "primavera": ["primavera p6", "oracle primavera"],
}


def app_skill_names():
    names = set(INDIA)
    with open(os.path.join(PROJECT_DIR, "skills.json"), encoding="utf-8") as f:
        for info in json.load(f).values():
            names.update(info["skills"])
    with open(os.path.join(PROJECT_DIR, "roadmaps.json"), encoding="utf-8") as f:
        for stages in json.load(f).values():
            names.update(step["skill"] for steps in stages.values() for step in steps)
    profiles_path = os.path.join(PROJECT_DIR, "career_profiles.json")
    if os.path.exists(profiles_path):
        with open(profiles_path, encoding="utf-8") as f:
            for profile in json.load(f)["careers"].values():
                names.update(skill["name"] for skill in profile["key_skills"])
    return names


def plain_label(label):
    """ "Python (computer programming)" -> "python" """
    return re.sub(r"\s*\(.*?\)", "", label or "").strip().lower()


def keep(label, skill, all_skills):
    """Is this a safe synonym for skill? Not too short or long, not generic, not another skill."""
    return (len(label) >= MIN_LENGTH and len(label.split()) <= MAX_WORDS and label != skill
            and label not in ESCO_BLOCKLIST and label not in all_skills)


def esco_synonyms(all_skills):
    with open(os.path.join(RAW_DIR, "esco", "skill_search.json"), encoding="utf-8") as f:
        searches = json.load(f)["data"]
    synonyms = {}
    for skill, result in searches.items():
        match = result.get("match")
        if skill not in all_skills or not match:
            continue
        # Only when ESCO's own name for the concept IS this skill. If the search only hit one of its
        # labels, the concept is wider: "git" finds "tools for software configuration management",
        # whose other labels (CFEngine, Rundeck...) are not git.
        if plain_label(match["label"]) != skill:
            continue
        labels = [match["label"]] + match["alt_labels"]
        found = sorted({label.strip().lower() for label in labels if keep(label.strip().lower(), skill, all_skills)})
        if found:
            synonyms[skill] = found
    return synonyms


def onet_synonyms(all_skills):
    with open(os.path.join(RAW_DIR, "onet", "Software Skills.txt"), encoding="utf-8") as f:
        names = {row["Workplace Example"].strip() for row in csv.DictReader(f, delimiter="\t")}
    synonyms = {}
    for name in names:
        lower = name.lower()
        vendor_match = [skill for skill in all_skills
                        if lower.endswith(" " + skill) and lower[:-len(skill) - 1] in ONET_VENDORS]
        for skill in vendor_match:
            synonyms.setdefault(skill, set()).add(lower)
        long_form = ONET_LONG_FORM.match(lower)
        if not vendor_match and long_form and long_form.group(2) in all_skills \
                and is_abbreviation(long_form.group(2), long_form.group(1)):
            synonyms.setdefault(long_form.group(2), set()).add(long_form.group(1))
    return {skill: sorted(words) for skill, words in synonyms.items()}


def is_abbreviation(short, long_form):
    """Is short made from the first letters of long_form's words (a leading vendor name is ignored)?
    "sql" / "structured query language" -> True;  "ssis" / "microsoft sql server integration services"
    -> True;  "audit" / "elite software energy" -> False."""
    words = long_form.split()
    if words[0] in ONET_VENDORS:
        words = words[1:]
    letters = "".join(ch for ch in short if ch.isalpha())
    initials = "".join(word[0] for word in words)
    if len(words) < 2 or not letters or letters[0] != initials[0]:
        return False
    # every word's first letter appears in the abbreviation, in order ("html": h-t-m-l has h, m, l)
    position = 0
    for initial in initials:
        position = letters.find(initial, position) + 1
        if position == 0:
            return False
    return len(letters) <= 2 * len(initials)


def main():
    all_skills = app_skill_names()
    by_source = {"esco": esco_synonyms(all_skills), "onet": onet_synonyms(all_skills), "india": INDIA}
    combined = {}
    for source, synonyms in by_source.items():
        for skill, words in synonyms.items():
            combined.setdefault(skill, {})[source] = list(words)
    output = {
        "_note": "Built by data/build_synonyms.py - don't edit by hand, change the script and rerun it.",
        "sources": {
            "esco": "ESCO classification v1.2.1, European Commission (esco.ec.europa.eu), alternative labels",
            "onet": "O*NET 31.0 Database, U.S. Department of Labor/ETA, CC BY 4.0 (onetcenter.org), Software Skills",
            "india": "Hand-written list of how Indian resumes write degrees, courses and skills "
                     "(incl. common short forms like OOPs, BOQ, MS Excel)",
        },
        "synonyms": dict(sorted(combined.items())),
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)
    counts = {source: sum(len(words) for words in synonyms.values()) for source, synonyms in by_source.items()}
    print(f"{len(combined)} skills with synonyms; synonyms per source: {counts}")


if __name__ == "__main__":
    main()
