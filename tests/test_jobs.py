import pytest
import requests

from features import jobs

ONE_JOB = {"jobs": [{"title": "Data Analyst", "company_name": "Acme", "url": "https://example.com/1",
                     "candidate_required_location": "Worldwide", "salary": "ignored"}]}


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
            calls.append({"url": url, "params": params, "timeout": timeout})
        if error:
            raise error
        return response
    return get


def test_success_returns_jobs_and_uses_new_url_and_short_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.requests, "get", fake_get(FakeResponse(200, ONE_JOB), calls=calls))
    result = jobs.fetch_job_listings("Data Science")
    assert result == [{"title": "Data Analyst", "company": "Acme", "url": "https://example.com/1", "location": "Worldwide"}]
    assert calls[0]["url"] == "https://remotive.com/api/remote-jobs"
    assert calls[0]["timeout"] == 4
    assert calls[0]["params"]["search"] == "Data Science"


def test_success_with_no_jobs_returns_empty_list(monkeypatch):
    monkeypatch.setattr(jobs.requests, "get", fake_get(FakeResponse(200, {"jobs": []})))
    assert jobs.fetch_job_listings("Astronaut") == []


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


def test_results_are_cached_per_career(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.requests, "get", fake_get(FakeResponse(200, ONE_JOB), calls=calls))
    jobs.fetch_job_listings("Data Science")
    jobs.fetch_job_listings("Data Science")      # from the cache
    jobs.fetch_job_listings("Java Developer")    # different career -> new call
    assert [c["params"]["search"] for c in calls] == ["Data Science", "Java Developer"]


def test_cache_expires_after_30_minutes(monkeypatch):
    calls = []
    clock = {"now": 1000.0}
    monkeypatch.setattr(jobs.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(jobs.requests, "get", fake_get(FakeResponse(200, ONE_JOB), calls=calls))
    jobs.fetch_job_listings("Data Science")
    clock["now"] += 29 * 60
    jobs.fetch_job_listings("Data Science")      # still cached
    clock["now"] += 2 * 60                       # now 31 minutes old
    jobs.fetch_job_listings("Data Science")
    assert len(calls) == 2


def test_failures_are_not_cached(monkeypatch):
    monkeypatch.setattr(jobs.requests, "get", fake_get(error=requests.Timeout()))
    assert jobs.fetch_job_listings("Data Science") is None
    monkeypatch.setattr(jobs.requests, "get", fake_get(FakeResponse(200, ONE_JOB)))
    assert len(jobs.fetch_job_listings("Data Science")) == 1
