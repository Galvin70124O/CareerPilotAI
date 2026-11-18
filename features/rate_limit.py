"""
A tiny rate limit for the /api routes and /report: each visitor (IP address) gets at most LIMIT requests
per WINDOW seconds; after that the route answers 429 "Too many requests" until the window moves on.

How it works: we keep the times of each IP's recent requests in a queue. On every request we
drop the times older than WINDOW seconds, then count what is left.
"""
import time
from collections import defaultdict, deque

LIMIT = 20       # design choice: a person clicking "Simulate" or "Check match" stays far below this
WINDOW = 60      # seconds
MAX_IPS = 10_000

# shortcut: lives in this process's memory, so it resets on restart and each gunicorn worker counts
# on its own (render.yaml runs one worker). Move to Redis or Flask-Limiter if the app runs several.
_recent = defaultdict(deque)


def allow(ip, now=None):
    """True if this IP may make another request now (and records it)."""
    now = time.monotonic() if now is None else now
    if len(_recent) > MAX_IPS:      # keeps memory bounded if many different IPs show up
        _recent.clear()
    times = _recent[ip]
    while times and times[0] <= now - WINDOW:
        times.popleft()
    if len(times) >= LIMIT:
        return False
    times.append(now)
    return True


def clear():
    """Forget every IP (used by the tests, which all come from the same address)."""
    _recent.clear()
