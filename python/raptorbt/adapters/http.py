"""A small HTTP layer the broker mappings share.

The mappings call a ``Transport``: any callable with the signature of
:func:`urllib_transport`. Tests pass a fake; production uses the standard
library one, so the broker extras add no dependency.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes = b""

    def json(self) -> Any:
        try:
            return json.loads(self.body or b"null")
        except ValueError:
            return None


class NotSent(Exception):
    """The request certainly never reached the server (refused, DNS)."""


class MaybeSent(Exception):
    """The request may have reached the server (timeout, reset mid-flight)."""


Transport = Callable[..., HttpResponse]


def urllib_transport(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    form: Mapping[str, Any] | None = None,
    body: Any = None,
    timeout: float = 10.0,
) -> HttpResponse:
    """``form`` is sent urlencoded, ``body`` as JSON, ``params`` in the query."""
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    data = None
    hdrs = dict(headers or {})
    if form is not None:
        data = urllib.parse.urlencode({k: v for k, v in form.items() if v is not None}).encode()
        hdrs.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - https URLs set by the mapping
            return HttpResponse(r.status, r.read())
    except urllib.error.HTTPError as e:
        return HttpResponse(e.code, e.read())
    except urllib.error.URLError as e:
        if isinstance(e.reason, (ConnectionRefusedError, socket.gaierror)):
            raise NotSent(str(e.reason)) from e
        raise MaybeSent(str(e.reason)) from e
    except (TimeoutError, ConnectionError, OSError) as e:
        raise MaybeSent(str(e)) from e
