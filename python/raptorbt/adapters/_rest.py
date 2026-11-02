"""Shared plumbing of the broker mappings."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Mapping
from typing import Any

from .base import Ambiguous, Transient
from .http import HttpResponse, MaybeSent, NotSent, Transport

IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))


def ist_ns(text: Any) -> int:
    """``"2026-11-02 09:15:01"`` (IST, as the brokers print it) to epoch ns; 0 if it is not one."""
    if not isinstance(text, str):
        return 0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            d = _dt.datetime.strptime(text[:19], fmt).replace(tzinfo=IST)
        except ValueError:
            continue
        return int(d.timestamp()) * 1_000_000_000
    return 0


def num(x: Any, default: float | None = None) -> float | None:
    """A broker number: ``None`` and empty become ``default``; ``0`` stays ``0``."""
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def price(x: Any) -> float | None:
    """A price field: the brokers send 0 for 'none' on market and limit orders."""
    v = num(x)
    return v if v else None


def rest(
    transport: Transport,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    classify: Callable[[HttpResponse], Any],
    write: bool,
    params: Mapping[str, Any] | None = None,
    form: Mapping[str, Any] | None = None,
    body: Any = None,
    timeout: float = 10.0,
) -> Any:
    """One call. A transport failure is ``Transient`` when nothing was sent (or for a read),
    ``Ambiguous`` when a write may have been processed. ``classify`` turns the response into
    data or raises the adapter error that fits."""
    try:
        resp = transport(method, url, headers=headers, params=params, form=form, body=body, timeout=timeout)
    except NotSent as e:
        raise Transient(f"not sent: {e}") from e
    except MaybeSent as e:
        raise (Ambiguous if write else Transient)(f"no answer: {e}") from e
    return classify(resp)
