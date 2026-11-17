"""
Build career_profiles.json: a weighted skill list for every career, plus which classifier
careers ("parents") each new career grows out of.

    python data/fetch_sources.py      # O*NET + ESCO into data/raw/
    python data/build_synonyms.py
    python data/build_profiles.py

How a skill's weight is made (every number can be traced back to its source):
- New careers (no ResumeAtlas resumes exist for them):
    ESCO: the skill is an ESCO "essential" skill of the career's occupation -> 1.0, "optional" -> 0.6
  + O*NET: the skill is a "hot" or "in demand" technology of the US occupation -> 0.5, listed -> 0.25
    A skill with neither (India-specific: DSA, internship, B.E./B.Tech, STAAD Pro) gets 0.5.
- The 42 existing careers:
    the share of that career's TRAINING resumes that mention the skill (0 to 1)
  + the same O*NET bonus.
  Only the training part of train.py's split is used, so the held-out test resumes never shape
  the profiles. ESCO occupations were fetched for the new careers only.

The script stops if a cited ESCO skill is not on that occupation, so evidence can't be made up.
"""
import csv
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_DIR)
sys.path.insert(0, PROJECT_DIR)
from features.skills import SYNONYMS, has_skill  # noqa: E402  (needs the project folder on the path)

RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT = os.path.join(PROJECT_DIR, "career_profiles.json")

ESCO_WEIGHT = {"essential": 1.0, "optional": 0.6}
ONET_WEIGHT = {"hot": 0.5, "listed": 0.25}
NO_EVIDENCE_WEIGHT = 0.5

# The US O*NET occupation that best matches each career
ONET_SOC = {
    "Accountant": "13-2011.00", "Advocate": "23-1011.00", "Agriculture": "11-9013.00",
    "Apparel": "27-1022.00", "Architecture": "17-1011.00", "Arts": "27-1013.00",
    "Automobile": "49-3023.00", "Aviation": "49-3011.00", "BPO": "43-4051.00", "Banking": "13-2072.00",
    "Blockchain": "15-1252.00", "Building and Construction": "11-9021.00",
    "Business Analyst": "13-1111.00", "Civil Engineer": "17-2051.00", "Consultant": "13-1111.00",
    "Data Science": "15-2051.00", "Database": "15-1242.00", "Designing": "27-1024.00",
    "DevOps": "15-1244.00", "Digital Media": "13-1161.00", "DotNet Developer": "15-1252.00",
    "ETL Developer": "15-1243.00", "Education": "25-2031.00", "Electrical Engineering": "17-2071.00",
    "Finance": "13-2051.00", "Food and Beverages": "35-1012.00", "Health and Fitness": "39-9031.00",
    "Human Resources": "13-1071.00", "Information Technology": "15-1231.00",
    "Java Developer": "15-1252.00", "Mechanical Engineer": "17-2141.00",
    "Network Security Engineer": "15-1212.00", "Operations Manager": "11-1021.00", "PMO": "13-1082.00",
    "Public Relations": "27-3031.00", "Python Developer": "15-1252.00", "React Developer": "15-1254.00",
    "SAP Developer": "15-1252.00", "SQL Developer": "15-1242.00", "Sales": "41-4012.00",
    "Testing": "15-1253.00", "Web Designing": "15-1255.00",
    # new careers
    "Software Development Engineer (Fresher)": "15-1252.00", "Data Analyst": "15-2051.01",
    "QA/Test Engineer": "15-1253.00", "Technical Support Engineer": "15-1232.00",
    "Graduate Engineer Trainee (Mechanical)": "17-2141.00",
    "Graduate Engineer Trainee (Electrical)": "17-2071.00",
    "Graduate Engineer Trainee (Civil)": "17-2051.00",
}

# New careers: (skill as the app matches it, ESCO skill label of the occupation or None,
#               O*NET software name when it isn't the skill or one of its synonyms, else None)
NEW_CAREERS = {
    "Software Development Engineer (Fresher)": {
        "parents": ["Java Developer", "Python Developer", "DotNet Developer", "React Developer"],
        "skills": [
            ("java", "Java (computer programming)", None),
            ("python", "Python (computer programming)", None),
            ("c++", "C++", None),
            ("javascript", "JavaScript", None),
            ("sql", "SQL", None),
            ("git", "tools for software configuration management", None),
            ("object-oriented programming", "use object-oriented programming", None),
            ("debugging", "debug software", None),
            ("data structures and algorithms", None, None),
            ("internship", None, None),
            ("b.e./b.tech", None, None),
        ],
    },
    "Data Analyst": {
        "parents": ["Data Science", "Business Analyst", "SQL Developer"],
        "skills": [
            ("sql", "query languages", None),
            ("excel", "use spreadsheets software", None),
            ("power bi", "data visualisation software", None),
            ("tableau", "data visualisation software", None),
            ("statistics", "statistics", None),
            ("data cleaning", "perform data cleansing", None),
            ("data visualization", "visual presentation techniques", None),
            ("dashboards", "deliver visual presentation of data", None),
            ("python", None, None),
            ("business intelligence", "business intelligence", None),
            ("internship", None, None),
        ],
    },
    "QA/Test Engineer": {
        "parents": ["Testing"],
        "skills": [
            ("manual testing", "execute software tests", None),
            ("test cases", "provide software testing documentation", None),
            ("bug reporting", "report test findings", None),
            ("regression testing", "levels of software testing", None),
            ("unit testing", "perform software unit testing", None),
            ("selenium", "tools for ICT test automation", None),
            ("jira", None, None),
            ("sql", "query languages", None),
            ("java", "Java (computer programming)", None),
            ("python", "Python (computer programming)", None),
            ("test plans", "plan software testing", None),
        ],
    },
    "Technical Support Engineer": {
        "parents": ["Information Technology", "Network Security Engineer"],
        "skills": [
            ("technical support", "provide ICT support", None),
            ("troubleshooting", "solve ICT system problems", None),
            ("ticketing", "use ICT ticketing system", None),
            ("hardware", "use ICT hardware", None),
            ("customer service", "assist customers", None),
            ("communication", "communicate with customers", None),
            ("help desk", "handle helpdesk problems", None),
            ("windows", None, None),
            ("active directory", None, "Microsoft Active Directory"),
            ("networking", "ICT infrastructure", None),
            ("vpn", "implement a virtual private network", None),
        ],
    },
    "Graduate Engineer Trainee (Mechanical)": {
        "parents": ["Mechanical Engineer"],
        "skills": [
            ("mechanical engineering", "mechanical engineering", None),
            ("autocad", "use technical drawing software", None),
            ("solidworks", "3D modelling", None),
            ("engineering drawing", "technical drawings", None),
            ("fluid mechanics", "fluid mechanics", None),
            ("strength of materials", "material mechanics", None),
            ("ansys", "computer simulation", None),
            ("creo", None, "PTC Creo Parametric"),
            ("matlab", None, None),
            ("internship", None, None),
            ("b.e./b.tech", None, None),
        ],
    },
    "Graduate Engineer Trainee (Electrical)": {
        "parents": ["Electrical Engineering"],
        "skills": [
            ("electrical engineering", "electrical engineering", None),
            ("circuit analysis", "interpret circuit diagrams", None),
            ("electrical machines", "electrical machines", None),
            ("power systems", "design electric power systems", None),
            ("electrical wiring", "electrical wiring diagrams", None),
            ("electrical testing", "electrical testing methods", None),
            ("electrical safety", "ensure safety in electrical power operations", None),
            ("electronics", "electronics", None),
            ("autocad", "use technical drawing software", None),
            ("matlab", None, None),
            ("internship", None, None),
            ("b.e./b.tech", None, None),
        ],
    },
    "Graduate Engineer Trainee (Civil)": {
        "parents": ["Civil Engineer"],
        "skills": [
            ("civil engineering", "civil engineering", None),
            ("autocad", "create CAD drawings", None),
            ("quantity surveying", "quantity surveying", None),
            ("surveying", "surveying", None),
            ("construction methods", "construction methods", None),
            ("site supervision", "oversee construction project", None),
            ("project management", "project management", None),
            ("revit", None, None),
            ("primavera", None, "Oracle Primavera Enterprise Project Portfolio Management"),
            ("staad pro", None, None),
            ("internship", None, None),
            ("b.e./b.tech", None, None),
        ],
    },
}


def load_onet():
    """{soc: {software name (lowercase): "hot" or "listed"}}"""
    onet = {}
    with open(os.path.join(RAW_DIR, "onet", "Software Skills.txt"), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            level = "hot" if "Y" in (row["Hot Technology"], row["In Demand"]) else "listed"
            names = onet.setdefault(row["O*NET-SOC Code"], {})
            if names.get(row["Workplace Example"].lower()) != "hot":
                names[row["Workplace Example"].lower()] = level
    return onet


def onet_evidence(skill, soc, onet, named_example=None):
    """("hot"/"listed", the O*NET name) if O*NET lists this skill for the occupation, else (None, None)."""
    names = onet.get(soc, {})
    candidates = [named_example.lower()] if named_example else [skill] + SYNONYMS.get(skill, [])
    found = [(names[name], name) for name in candidates if name in names]
    if named_example and not found:
        sys.exit(f"O*NET {soc} does not list '{named_example}'")
    found.sort(key=lambda pair: pair[0] != "hot")   # prefer a "hot" match
    return found[0] if found else (None, None)


def esco_evidence(career, label, occupations):
    occupation = occupations[career]
    for kind, level in [("hasEssentialSkill", "essential"), ("hasOptionalSkill", "optional")]:
        if any(skill["label"] == label for skill in occupation[kind]):
            return level
    sys.exit(f"ESCO occupation '{occupation['title']}' has no skill '{label}' ({career})")


def new_career_profile(career, spec, occupations, onet):
    soc = ONET_SOC[career]
    key_skills = []
    for skill, esco_label, onet_name in spec["skills"]:
        esco_level = esco_evidence(career, esco_label, occupations) if esco_label else None
        onet_level, onet_match = onet_evidence(skill, soc, onet, onet_name)
        weight = ESCO_WEIGHT.get(esco_level, 0) + ONET_WEIGHT.get(onet_level, 0)
        evidence = {"esco": f"{esco_level}: {esco_label}" if esco_level else None,
                    "onet": f"{onet_level}: {onet_match}" if onet_level else None}
        if not weight:
            weight, evidence = NO_EVIDENCE_WEIGHT, {"india": "common on Indian fresher resumes"}
        key_skills.append({"name": skill, "weight": round(weight, 2),
                           "evidence": {k: v for k, v in evidence.items() if v}})
    key_skills.sort(key=lambda s: -s["weight"])
    return {"new": True, "parents": spec["parents"], "onet_soc": soc,
            "esco_occupation": occupations[career]["uri"], "key_skills": key_skills}


def training_resumes():
    """The same 80% training split as train.py (test_size=0.2, stratified, random_state=42)."""
    data = pd.read_parquet(os.path.join(DATA_DIR, "resumes.parquet"))
    x_train, _, y_train, _ = train_test_split(data["Resume"], data["Category"], test_size=0.2,
                                              stratify=data["Category"], random_state=42)
    return pd.DataFrame({"Resume": x_train, "Category": y_train})


def existing_career_profile(career, skills, resumes, onet):
    soc = ONET_SOC[career]
    texts = resumes.loc[resumes["Category"] == career, "Resume"]
    key_skills = []
    for skill in skills:
        share = sum(has_skill(skill, text) for text in texts) / len(texts)
        onet_level, onet_match = onet_evidence(skill, soc, onet)
        evidence = {"resumeatlas": f"in {share:.0%} of {len(texts)} training resumes"}
        if onet_level:
            evidence["onet"] = f"{onet_level}: {onet_match}"
        key_skills.append({"name": skill, "weight": round(share + ONET_WEIGHT.get(onet_level, 0), 2),
                           "evidence": evidence})
    key_skills.sort(key=lambda s: -s["weight"])
    return {"new": False, "parents": [], "onet_soc": soc, "key_skills": key_skills}


def main():
    with open(os.path.join(RAW_DIR, "esco", "occupations.json"), encoding="utf-8") as f:
        occupations = json.load(f)["data"]
    with open(os.path.join(PROJECT_DIR, "skills.json"), encoding="utf-8") as f:
        existing = {career: info["skills"] for career, info in json.load(f).items()}
    onet = load_onet()
    resumes = training_resumes()
    careers = {career: existing_career_profile(career, skills, resumes, onet)
               for career, skills in existing.items() if career not in NEW_CAREERS}
    for career, spec in NEW_CAREERS.items():
        for parent in spec["parents"]:
            assert parent in careers, f"{career}: unknown parent {parent}"
        careers[career] = new_career_profile(career, spec, occupations, onet)
    output = {
        "_note": "Built by data/build_profiles.py - don't edit by hand, change the script and rerun it.",
        "weights": "new careers: ESCO essential 1.0 / optional 0.6 + O*NET hot 0.5 / listed 0.25, "
                   "else 0.5; existing careers: share of training resumes + the same O*NET bonus",
        "sources": {
            "resumeatlas": "ResumeAtlas (MIT), training split only (train.py: 80%, random_state=42)",
            "esco": "ESCO classification v1.2.1, European Commission, occupation essential/optional skills",
            "onet": "O*NET 31.0 Database, U.S. Department of Labor/ETA, CC BY 4.0, Software Skills",
        },
        "careers": careers,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)
    print(f"{len(careers)} careers ({sum(p['new'] for p in careers.values())} new) -> career_profiles.json")


if __name__ == "__main__":
    main()
