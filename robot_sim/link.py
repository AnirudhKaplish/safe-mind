"""Read-only localhost link to the dashboard.

The robot can only *read* its state: this module performs plain GET
requests to ``/api/robot/state`` on 127.0.0.1/localhost and nothing else.
There is no code path that sends a request body or uses another verb
(a unit test inspects this file's source to enforce it). Proxies from the
environment are bypassed so nothing ever leaves the machine.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

from . import config


class NonLocalURL(ValueError):
    pass


def check_local_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "http":
        raise NonLocalURL(f"robot link must use http:// on localhost, got {url!r}")
    host = (parts.hostname or "").lower()
    if host not in config.ALLOWED_HOSTS:
        raise NonLocalURL(f"robot link may only contact {' or '.join(config.ALLOWED_HOSTS)}; refused host {host!r}")
    if parts.username or parts.password:
        raise NonLocalURL("credentials in the robot URL are not allowed")
    return url


# Opener with an empty ProxyHandler: environment proxies are ignored.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LinkError(Exception):
    pass


def fetch_state(url: str, timeout: float = config.POLL_TIMEOUT_S) -> tuple[Any, float]:
    """GET the robot state. Returns (parsed JSON or a ``_BadJSON`` marker, latency s)."""
    check_local_url(url)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    t0 = time.monotonic()
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            body = resp.read(65536)
    except urllib.error.HTTPError as e:
        raise LinkError(f"http_{e.code}") from None
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        name = "timeout" if "timed out" in str(reason).lower() else "unreachable"
        raise LinkError(name) from None
    latency = time.monotonic() - t0
    try:
        return json.loads(body.decode("utf-8")), latency
    except (UnicodeDecodeError, json.JSONDecodeError):
        return BAD_JSON, latency


class _BadJSON:
    def __repr__(self) -> str:
        return "<bad json>"


BAD_JSON = _BadJSON()


class PayloadGate:
    """seq / boot_id ordering rules (spec 9)."""

    def __init__(self) -> None:
        self.boot_id: Optional[str] = None
        self.seq = -1
        self.duplicates = 0
        self.out_of_order = 0

    def accept(self, payload: dict) -> bool:
        b, s = payload["boot_id"], payload["seq"]
        if b != self.boot_id:
            self.boot_id, self.seq = b, s
            return True
        if s > self.seq:
            self.seq = s
            return True
        if s == self.seq:
            self.duplicates += 1
        else:
            self.out_of_order += 1
        return False

    def force(self, payload: dict) -> None:
        self.boot_id, self.seq = payload["boot_id"], payload["seq"]


@dataclass
class PollResult:
    ok: bool
    payload: Any = None
    latency: float = 0.0
    error: str = ""
    t: float = 0.0


class LinkPoller:
    """Background thread polling the dashboard at ``poll_hz``."""

    def __init__(self, url: str = config.DEFAULT_URL, poll_hz: float = config.POLL_HZ,
                 timeout: float = config.POLL_TIMEOUT_S) -> None:
        self.url = check_local_url(url)
        self.period = 1.0 / max(poll_hz, 0.1)
        self.timeout = timeout
        self.results: "queue.Queue[PollResult]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="robot-link", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                payload, lat = fetch_state(self.url, self.timeout)
                self.results.put(PollResult(True, payload, lat, t=time.monotonic()))
            except LinkError as e:
                self.results.put(PollResult(False, error=str(e), t=time.monotonic()))
            next_t += self.period
            delay = next_t - time.monotonic()
            if delay < 0:
                next_t = time.monotonic()
                delay = 0
            self._stop.wait(delay)

    def drain(self) -> list[PollResult]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out
