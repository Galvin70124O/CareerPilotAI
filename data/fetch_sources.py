"""
Download the two outside skill sources into data/raw/ (git-ignored).

    python data/fetch_sources.py

- O*NET 31.0 database (text files, CC BY 4.0, onetcenter.org): we keep "Software Skills.txt"
  (the technologies each US occupation uses) and "Occupation Data.txt" (occupation titles).
- ESCO v1.2.1 (European Commission, free reuse with attribution): the full CSV download needs an
  email sign-up, so we use the public ESCO API with selectedVersion=v1.2.1 instead and save the
  raw JSON. We fetch:
    * 7 occupations that match the new careers, with their essential/optional skills
    * a skill search for every skill name the app already uses (skills.json + roadmaps.json),
      which gives ESCO's preferred and alternative labels (synonyms) for them

Running it again skips files that already exist. Delete data/raw/ to download fresh copies.
"""
import hashlib
import json
import os
import re
import sys
import time
import zipfile

import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_DIR)
RAW_DIR = os.path.join(DATA_DIR, "raw")

ONET_VERSION = "31.0"
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_31_0_text.zip"
ONET_FILES = ["Software Skills.txt", "Occupation Data.txt"]

ESCO_VERSION = "v1.2.1"
ESCO_API = "https://ec.europa.eu/esco/api"
# ESCO occupation searches for the new careers (ESCO uses European job titles)
ESCO_OCCUPATIONS = {
    "Software Development Engineer (Fresher)": "software developer",
    "Data Analyst": "data analyst",
    "QA/Test Engineer": "software tester",
    "Technical Support Engineer": "ICT help desk agent",
    "Graduate Engineer Trainee (Mechanical)": "mechanical engineer",
    "Graduate Engineer Trainee (Electrical)": "electrical engineer",
    "Graduate Engineer Trainee (Civil)": "civil engineer",
}
TIMEOUT = 30              # seconds per request
PAUSE = 0.3               # seconds between ESCO calls, to be polite to the free API
RETRY_WAITS = [2, 4, 8]   # after a failed call, wait 2 s, 4 s, 8 s and try again (4 attempts in total)
HEADERS = {"User-Agent": "CareerPilotAI data script (student project)"}
ESCO_CACHE_DIR = os.path.join(RAW_DIR, "esco", "cache")

# Counts written to data/raw/esco/fetch_log.json at the end
fetch_log = {"network_calls": 0, "cache_hits": 0, "retries": 0, "failed": []}


def download_onet():
    folder = os.path.join(RAW_DIR, "onet")
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "db_31_0_text.zip")
    if not os.path.exists(zip_path):
        print("Downloading O*NET", ONET_VERSION, "...")
        response = requests.get(ONET_URL, headers=HEADERS, timeout=120)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(response.content)
    with zipfile.ZipFile(zip_path) as archive:
        for name in ONET_FILES:
            with open(os.path.join(folder, name), "wb") as f:
                f.write(archive.read("db_31_0_text/" + name))
    print("O*NET files:", ", ".join(ONET_FILES))


def cache_path(path, params):
    """One cache file per request. The name is a hash of the URL and its parameters."""
    key = path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return os.path.join(ESCO_CACHE_DIR, hashlib.sha1(key.encode("utf-8")).hexdigest() + ".json")


def esco_get(path, **params):
    """GET from the ESCO API, using the cache first. Returns None if every attempt failed."""
    params.update(language="en", selectedVersion=ESCO_VERSION)
    cached = cache_path(path, params)
    last_error = ""
    if os.path.exists(cached):
        fetch_log["cache_hits"] += 1
        with open(cached, encoding="utf-8") as f:
            return json.load(f)
    for attempt, wait in enumerate([0] + RETRY_WAITS):
        if attempt > 0:
            fetch_log["retries"] += 1
            time.sleep(wait)
        try:
            fetch_log["network_calls"] += 1
            response = requests.get(ESCO_API + path, params=params, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()   # a wrong version gives HTTP 500, so we never save wrong data
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
            continue
        finally:
            time.sleep(PAUSE)
        os.makedirs(ESCO_CACHE_DIR, exist_ok=True)
        # Write to a temporary file, then rename: if the script is stopped mid-write,
        # no half-written cache file is left behind
        with open(cached + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(cached + ".tmp", cached)
        return data
    fetch_log["failed"].append({"path": path, "params": params, "error": last_error})
    print("FAILED after retries:", path, params.get("text") or params.get("uri"))
    return None


def esco_api_software_version():
    """The API reports its own software version at its root (not the data version)."""
    response = requests.get(ESCO_API + "/", headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json().get("version")


def skill_summary(resource):
    """Keep only what we need from an ESCO skill: its id, preferred label and English synonyms."""
    return {
        "uri": resource["uri"],
        "label": resource.get("preferredLabel", {}).get("en") or resource.get("title"),
        "alt_labels": resource.get("alternativeLabel", {}).get("en", []),
    }


def fetch_esco_occupations():
    occupations = {}
    for career, query in ESCO_OCCUPATIONS.items():
        found = esco_get("/search", text=query, type="occupation", limit=1)
        if found is None:
            continue
        hit = found["_embedded"]["results"][0]
        occupation = esco_get("/resource/occupation", uri=hit["uri"])
        if occupation is None:
            continue
        links = occupation.get("_links", {})
        skills = {}
        for kind in ["hasEssentialSkill", "hasOptionalSkill"]:
            skills[kind] = []
            for link in links.get(kind, []):
                skill = esco_get("/resource/skill", uri=link["uri"])
                if skill is not None:
                    skills[kind].append(skill_summary(skill))
        occupations[career] = {"query": query, "uri": hit["uri"], "title": occupation.get("title"), **skills}
        print(f"ESCO occupation for {career}: {occupation.get('title')}")
    return occupations


def app_skill_names():
    """Every skill name the app already uses, so we can look up ESCO synonyms for them."""
    names = set()
    with open(os.path.join(PROJECT_DIR, "skills.json"), encoding="utf-8") as f:
        for info in json.load(f).values():
            names.update(info["skills"])
    with open(os.path.join(PROJECT_DIR, "roadmaps.json"), encoding="utf-8") as f:
        for stages in json.load(f).values():
            names.update(step["skill"] for steps in stages.values() for step in steps)
    profiles_path = os.path.join(PROJECT_DIR, "career_profiles.json")
    if os.path.exists(profiles_path):   # the new careers (added in step 3)
        with open(profiles_path, encoding="utf-8") as f:
            for profile in json.load(f)["careers"].values():
                names.update(skill["name"] for skill in profile["key_skills"])
    return sorted(names)


def plain_label(label):
    """ "Python (computer programming)" -> "python" """
    return re.sub(r"\s*\(.*?\)", "", label or "").strip().lower()


def fetch_esco_skill_search():
    results = {}
    for name in app_skill_names():
        found = esco_get("/search", text=name, type="skill", limit=3)
        if found is None:
            continue
        hits = found["_embedded"]["results"]
        results[name] = {"hits": [{"uri": hit["uri"], "label": hit.get("title"), "search_hit": hit.get("searchHit")}
                                  for hit in hits],
                         "match": None}
        # If the best hit IS this skill, fetch it to get ESCO's alternative labels (synonyms)
        best = hits[0] if hits else None
        if best and name in (plain_label(best.get("title")), (best.get("searchHit") or "").lower()):
            skill = esco_get("/resource/skill", uri=best["uri"])
            if skill is not None:
                results[name]["match"] = skill_summary(skill)
    print("ESCO skill searches:", len(results))
    return results


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def main():
    download_onet()
    folder = os.path.join(RAW_DIR, "esco")
    os.makedirs(folder, exist_ok=True)
    api_version = esco_api_software_version()
    # Always rebuild these two files: the cache makes it cheap, and a rerun fills any gaps
    save_json(os.path.join(folder, "occupations.json"),
              {"esco_version": ESCO_VERSION, "data": fetch_esco_occupations()})
    save_json(os.path.join(folder, "skill_search.json"),
              {"esco_version": ESCO_VERSION, "data": fetch_esco_skill_search()})
    log = {"esco_data_version": ESCO_VERSION, "esco_api_software_version": api_version,
           "onet_version": ONET_VERSION, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **fetch_log}
    save_json(os.path.join(folder, "fetch_log.json"), log)
    print(f"ESCO data {ESCO_VERSION} (API software {api_version}): {fetch_log['network_calls']} network calls, "
          f"{fetch_log['cache_hits']} from cache, {fetch_log['retries']} retries, {len(fetch_log['failed'])} failed")
    if fetch_log["failed"]:
        print("Some calls failed. Run the script again: it only fetches what is missing.")
        sys.exit(1)
    print("Done. Files are in", RAW_DIR)


if __name__ == "__main__":
    main()
