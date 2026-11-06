"""What the engine supports, as data.

``raptorbt.capabilities()`` returns a JSON-serialisable dict an embedding platform
reads to know what to offer. Nothing in it is typed out for the manifest: the
engine half comes from the engine's own types (``_raptorbt.capabilities_json``,
which includes the expression ``functions`` from the function registry),
the hooks from the strategy hook tables, the indicators from the functions the
extension registered as indicators and the streaming constructors of
``Indicator``, the adapters from ``raptorbt.execution`` and the error codes from
``raptorbt.errors.CODES``. A test fails when something public exists that the
manifest does not list.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from raptorbt import _raptorbt
from raptorbt import errors as _errors
from raptorbt.strategy import api as _api

__all__ = ["capabilities"]

#: Bump when a key is removed or changes meaning (adding keys is not a break).
SCHEMA = 1


def _params(fn: Any) -> list[dict[str, Any]]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    out = []
    for p in signature.parameters.values():
        row: dict[str, Any] = {"name": p.name}
        if p.default is not inspect.Parameter.empty:
            row["default"] = p.default if isinstance(p.default, (int, float, str, bool, type(None))) else repr(p.default)
        out.append(row)
    return out


def _hooks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, hook in _api.HOOKS.items():
        rows.append({"name": name, "aliases": list(hook.aliases)})
    # engine events delivered to plain methods, with no channel of their own
    for name in _api.PLAIN_EVENT_BITS:
        if name not in {r["name"] for r in rows}:
            rows.append({"name": name, "aliases": []})
    # called by the recovery sequence of a live node
    from raptorbt.strategy._recovery import RECOVERY_HOOKS

    for name in RECOVERY_HOOKS:
        if name not in {r["name"] for r in rows}:
            rows.append({"name": name, "aliases": []})
    return rows


def _indicators() -> dict[str, Any]:
    arrays = [
        {"name": name, "parameters": _params(getattr(_raptorbt, name))}
        for name in _raptorbt._INDICATOR_FUNCTIONS
    ]
    streaming = []
    for name, member in inspect.getmembers(_raptorbt.Indicator):
        if name.startswith("_") or not callable(member):
            continue
        if name in {"reset", "update_bar"}:
            continue
        streaming.append({"name": name, "parameters": _params(member)})
    return {"arrays": arrays, "streaming": streaming}


def _adapters() -> list[dict[str, Any]]:
    from raptorbt import execution

    protocol = sorted(getattr(execution.ExecutionAdapter, "__protocol_attrs__", ()))
    rows = []
    for name in execution.__all__:
        obj = getattr(execution, name)
        if inspect.isclass(obj) and name.endswith("Execution") and name != "ExecutionAdapter":
            doc = (inspect.getdoc(obj) or "").split("\n\n")[0].replace("\n", " ")
            rows.append({"name": name, "summary": doc})
    return [{**r, "custom_adapter_methods": protocol} for r in rows]


def capabilities() -> dict[str, Any]:
    """The engine's capability manifest (see the module docs)."""
    out: dict[str, Any] = json.loads(_raptorbt.capabilities_json())
    out["schema"] = SCHEMA
    out["engine"] = {"name": "raptorbt", "version": _raptorbt.__version__ if hasattr(_raptorbt, "__version__") else None}
    out["hooks"] = _hooks()
    out["indicators"] = _indicators()
    out["adapters"] = _adapters()
    out["error_codes"] = sorted(_errors.CODES)
    return out
