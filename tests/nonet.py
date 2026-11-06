"""pytest plugin ``-p nonet``: any attempt to reach a non-loopback host fails immediately.

Usage (this file lives in ``tests/``)::

    PYTHONPATH=tests python -m pytest -p nonet tests/...
"""

from __future__ import annotations

import socket

_ALLOWED = {"localhost", "127.0.0.1", "::1", ""}
_real_getaddrinfo = socket.getaddrinfo
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


class NetworkBlocked(RuntimeError):
    pass


def _host(address) -> str:
    return str(address[0]) if isinstance(address, tuple) and address else str(address)


def _check(host: str) -> None:
    if host not in _ALLOWED and not host.startswith("/"):
        raise NetworkBlocked(f"network access blocked by -p nonet: {host}")


def _getaddrinfo(host, *a, **k):
    _check(str(host) if host is not None else "")
    return _real_getaddrinfo(host, *a, **k)


def _connect(self, address):
    if self.family != getattr(socket, "AF_UNIX", -1):
        _check(_host(address))
    return _real_connect(self, address)


def _connect_ex(self, address):
    if self.family != getattr(socket, "AF_UNIX", -1):
        _check(_host(address))
    return _real_connect_ex(self, address)


def pytest_configure(config):
    socket.getaddrinfo = _getaddrinfo
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex


def pytest_unconfigure(config):
    socket.getaddrinfo = _real_getaddrinfo
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
