import pytest
import requests

from features import jobs
from features.skills import CAREER_SKILLS, JOB_KEYWORDS


def remotive_job(title, company="Acme"):
    # The shape of one job in Remotive's JSON answer (extra fields are ignored)
    return {"title": title, "company_name": company, "url": f"https://remotive.com/job/{title.replace(' ', '-')}",
            "candidate_required_location": "Worldwide", "salary": "ignored"}


FEED = {"jobs": [
    remotive_job("Senior Data Scientist"),
    remotive_job("Remote Office Assistant"),
    remotive_job("Data Analyst (SQL)"),
    remotive_job("Freelance Copywriter"),
    remotive_job("Machine Learning Engineer"),
    remotive_job("Smart Contract Engineer"),
    remotive_job("Analytics Lead"),
]}


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if self.body is None:
            raise ValueError("not JSON")
        return self.body


@pytest.fixture(autouse=True)
def fresh_cache():
    jobs.clear_cache()
    yield
    jobs.clear_cache()


def fake_get(response=None, error=None, calls=None):
    def get(url, params=None, timeout=None):
        if calls is not None:
            calls.append({"url": url, "timeout": timeout})
        if error:
            raise error
        return response
    return get


def use_feed(monkeypatch, body=FEED, calls=None):
    monkeypatch.setattr(jobs.requests, "get", fake_get(FakeResponse(200, body), calls=calls))


# --- job_keywords in skills.json -------------------------------------------------

def test_every_career_has_job_keywords():
    assert set(JOB_KEYWORDS) == set(CAREER_SKILLS)
    for career, keywords in JOB_KEYWORDS.items():
        assert keywords, career
        assert all(k == k.lower() and k.strip() == k for k in keywords), career


# --- relevance -------------------------------------------------------------------

def test_only_matching_titles_when_enough_match(monkeypatch):
    use_feed(monkeypatch)
    result = jobs.fetch_job_listings("Data Science")
    assert [j["title"] for j in result] == ["Senior Data Scientist", "Data Analyst (SQL)",
                                            "Machine Learning Engineer", "Analytics Lead"]
    assert not any(j["related"] for j in result)


def test_few_matches_are_filled_with_related_jobs(monkeypatch):
    use_feed(monkeypatch)
    result = jobs.fetch_job_listings("Digital Media")        # only "Freelance Copywriter" matches
    assert len(result) == 5
    assert result[0]["title"] == "Freelance Copywriter" and result[0]["related"] is False
    assert all(j["related"] for j in result[1:])            # the rest are labelled "Related"


def test_no_matches_gives_only_related_jobs(monkeypatch):
    use_feed(monkeypatch)
    result = jobs.fetch_job_listings("Aviation")
    assert len(result) == 5 and all(j["related"] for j in result)


def test_keywords_match_whole_words_only(monkeypatch):
    # "art" (Arts) must not match "Smart Contract Engineer"; "ml" must not match "html"
    assert not jobs.title_matches("Smart Contract Engineer", ["art"])
    assert not jobs.title_matches("HTML Email Developer", ["ml engineer", "ml"])
    assert jobs.title_matches("Senior .NET Developer", [".net"])
    assert jobs.title_matches("ML Engineer", ["ml engineer"])


def test_limit_is_respected(monkeypatch):
    use_feed(monkeypatch)
    assert len(jobs.fetch_job_listings("Data Science", limit=2)) == 2


def test_unknown_career_gets_related_jobs(monkeypatch):
    use_feed(monkeypatch)
    result = jobs.fetch_job_listings("Astronaut")
    assert result and all(j["related"] for j in result)


# --- API success / failure -----------------------------------------------------

def test_success_uses_new_url_and_short_timeout(monkeypatch):
    calls = []
    use_feed(monkeypatch, calls=calls)
    jobs.fetch_job_listings("Data Science")
    assert calls == [{"url": "https://remotive.com/api/remote-jobs", "timeout": 4}]


def test_empty_feed_returns_empty_list(monkeypatch):
    use_feed(monkeypatch, body={"jobs": []})
    assert jobs.fetch_job_listings("Data Science") == []


def test_timeout_returns_none(monkeypatch):
    monkeypatch.setattr(jobs.requests, "get", fake_get(error=requests.Timeout("too slow")))
    assert jobs.fetch_job_listings("Data Science") is None


@pytest.mark.parametrize("response, error", [
    (FakeResponse(526, None), None),                          # Cloudflare SSL error, like remotive.io
    (FakeResponse(500, {"jobs": []}), None),                  # server error
    (FakeResponse(200, None), None),                          # answer is not JSON
    (FakeResponse(200, {"jobs": [{"title": "x"}]}), None),    # unexpected JSON shape
    (None, requests.ConnectionError("no internet")),          # no connection
])
def test_failures_return_none(monkeypatch, response, error):
    monkeypatch.setattr(jobs.requests, "get", fake_get(response, error))
    assert jobs.fetch_job_listings("Data Science") is None


# --- cache -------------------------------------------------------------------------

def test_one_download_serves_every_career(monkeypatch):
    calls = []
    use_feed(monkeypatch, calls=calls)
    jobs.fetch_job_listings("Data Science")
    jobs.fetch_job_listings("Data Science")
    jobs.fetch_job_listings("Java Developer")
    assert len(calls) == 1


def test_cache_expires_after_6_hours(monkeypatch):
    calls = []
    clock = {"now": 1000.0}
    monkeypatch.setattr(jobs.time, "monotonic", lambda: clock["now"])
    use_feed(monkeypatch, calls=calls)
    jobs.fetch_job_listings("Data Science")
    clock["now"] += 5 * 60 * 60 + 59 * 60        # 5 h 59 min later: still cached
    jobs.fetch_job_listings("Data Science")
    clock["now"] += 2 * 60                       # now 6 h 1 min old
    jobs.fetch_job_listings("Data Science")
    assert len(calls) == 2


def test_at_most_4_downloads_a_day():
    # Remotive's terms advise at most ~4 requests a day
    assert 24 * 60 * 60 / jobs.CACHE_SECONDS <= 4


def test_failures_are_not_cached(monkeypatch):
    monkeypatch.setattr(jobs.requests, "get", fake_get(error=requests.Timeout()))
    assert jobs.fetch_job_listings("Data Science") is None
    use_feed(monkeypatch)
    assert len(jobs.fetch_job_listings("Data Science")) == 4
