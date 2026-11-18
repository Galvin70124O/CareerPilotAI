import pytest

from features import rate_limit


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    # every test-client request comes from 127.0.0.1, so start each test with an empty count
    rate_limit.clear()
