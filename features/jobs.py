"""
Live remote jobs from the Remotive API (https://remotive.com/api/remote-jobs).

fetch_job_listings(career) returns:
  - a list of jobs (it may be empty) when the API answered, or
  - None when the API failed (down, slow, bad answer), so the page can say
    "Live jobs are unavailable right now" instead of showing nothing.

Relevance: we keep jobs whose TITLE contains one of the career's "job_keywords"
(skills.json). If fewer than MIN_MATCHING jobs match, the list is filled up with other
Remotive jobs marked "related": True, so the user still sees something.

Remotive's public API currently returns the same small feed whatever you search for,
so we download the feed ONCE and filter it for every career ourselves. The feed is
cached for CACHE_SECONDS; Remotive asks for at most a few requests a day.
Failures are NOT cached, so the next request tries again.
"""
import time

import requests

from features.skills import JOB_KEYWORDS, skill_pattern

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
TIMEOUT_SECONDS = 4               # a slow API must never make an upload hang
CACHE_SECONDS = 30 * 60           # keep the downloaded feed for 30 minutes
MIN_MATCHING = 3                  # fewer matching jobs than this -> fill with "related" jobs

# The one shared feed: {"saved_at": time, "jobs": [...]}
_cache = {}


def clear_cache():
    _cache.clear()


def _download_feed():
    response = requests.get(REMOTIVE_URL, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()   # turns 4xx/5xx answers (like 526) into an error
    jobs = []
    for job in response.json().get("jobs", []):
        jobs.append({
            "title": job["title"],
            "company": job["company_name"],
            "url": job["url"],
            "location": job["candidate_required_location"],
        })
    return jobs


def get_feed():
    """All jobs from Remotive (cached), or None if the API failed."""
    now = time.monotonic()   # a clock that never jumps back, good for measuring ages
    if _cache and now - _cache["saved_at"] < CACHE_SECONDS:
        return _cache["jobs"]
    try:
        jobs = _download_feed()
    except (requests.RequestException, ValueError, KeyError, AttributeError):
        # RequestException: timeout, no connection, HTTP error
        # ValueError: the answer was not JSON; KeyError/AttributeError: unexpected JSON shape
        return None
    _cache.update(saved_at=now, jobs=jobs)
    return jobs


def title_matches(title, keywords):
    # Whole words only, like skill matching: "art" must not match "Smart Contract Engineer"
    title = title.lower()
    return any(skill_pattern(keyword).search(title) for keyword in keywords)


def pick_jobs(jobs, career, limit=5):
    """Matching jobs first; if fewer than MIN_MATCHING match, fill up to `limit` with related ones."""
    keywords = JOB_KEYWORDS.get(career, [])
    matching = [dict(job, related=False) for job in jobs if title_matches(job["title"], keywords)]
    if len(matching) >= MIN_MATCHING:
        return matching[:limit]
    others = [dict(job, related=True) for job in jobs if not title_matches(job["title"], keywords)]
    return (matching + others)[:limit]


def fetch_job_listings(career, limit=5):
    jobs = get_feed()
    if jobs is None:
        return None
    return pick_jobs(jobs, career, limit)
