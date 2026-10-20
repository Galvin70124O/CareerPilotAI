"""
Live remote jobs from the Remotive API (https://remotive.com/api/remote-jobs).

fetch_job_listings() returns:
  - a list of jobs (it may be empty) when the API answered, or
  - None when the API failed (down, slow, bad answer), so the page can say
    "Live jobs are unavailable right now" instead of showing nothing.

Successful results are cached per career for 30 minutes, so the same search doesn't
call the API on every upload. Failures are NOT cached, so the next request tries again.
"""
import time

import requests

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
TIMEOUT_SECONDS = 4               # a slow API must never make an upload hang
CACHE_SECONDS = 30 * 60           # keep results for 30 minutes

# (job_title, limit) -> (time saved, jobs)
_cache = {}


def clear_cache():
    _cache.clear()


def _call_remotive(job_title, limit):
    response = requests.get(REMOTIVE_URL, params={"search": job_title, "limit": str(limit)},
                            timeout=TIMEOUT_SECONDS)
    response.raise_for_status()   # turns 4xx/5xx answers (like 526) into an error
    jobs = []
    for job in response.json().get("jobs", [])[:limit]:
        jobs.append({
            "title": job["title"],
            "company": job["company_name"],
            "url": job["url"],
            "location": job["candidate_required_location"],
        })
    return jobs


def fetch_job_listings(job_title, limit=5):
    key = (job_title.lower(), limit)
    now = time.monotonic()   # a clock that never jumps back, good for measuring ages

    if key in _cache:
        saved_at, jobs = _cache[key]
        if now - saved_at < CACHE_SECONDS:
            return jobs

    try:
        jobs = _call_remotive(job_title, limit)
    except (requests.RequestException, ValueError, KeyError, AttributeError):
        # RequestException: timeout, no connection, HTTP error
        # ValueError: the answer was not JSON; KeyError/AttributeError: unexpected JSON shape
        return None

    _cache[key] = (now, jobs)
    return jobs
