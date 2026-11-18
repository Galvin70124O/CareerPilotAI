"""features/rate_limit.py and the 429 answer on the /api routes."""
import app
from features import rate_limit
from features.rate_limit import LIMIT, WINDOW, allow


def test_an_ip_is_stopped_after_the_limit_and_allowed_again_later():
    assert all(allow("1.2.3.4", now=0) for _ in range(LIMIT))
    assert not allow("1.2.3.4", now=1)
    assert allow("5.6.7.8", now=1)               # other visitors are not affected
    assert allow("1.2.3.4", now=WINDOW + 0.1)    # the old requests left the window


def test_api_routes_answer_429_when_over_the_limit():
    client = app.app.test_client()
    for _ in range(LIMIT):
        rate_limit.allow("127.0.0.1")
    for route in ["/api/simulate", "/api/jd-match"]:
        response = client.post(route, json={})
        assert response.status_code == 429
        assert "Too many requests" in response.get_json()["error"]
