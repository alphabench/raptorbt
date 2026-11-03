"""Turn the engine's money and fee-schedule faults into their structured errors, and name the call.

The kernel books amounts as integers. When one cannot be booked (not finite,
past the range) or a sum would overflow, it stops with a panic whose message
starts ``money_overflow:``; PyO3 raises a panic as ``PanicException``, which is
not a ``RaptorBTError``. This module wraps every callable the extension exports
(functions, and the methods of its classes) so that exception leaves Python as
:class:`raptorbt.errors.MoneyOverflow`. A fault raised under a strategy callback
passes up through the callback and is converted at the outer entry point, so
the per-event ``ctx`` objects are not wrapped. Any other panic is left as it is:
those are internal invariants.

A fill priced on a date the fee schedule has no rate for stops the same way, with a panic whose
message starts ``fee_schedule_gap:``; it leaves Python as ``fee_schedule_gap``
(:class:`raptorbt.errors.InvalidParameter`).
"""

from __future__ import annotations

import functools
import inspect
import re

from raptorbt import errors

_FAULT = re.compile(r"money_overflow:\s*(.*)", re.S)
_GAP = re.compile(r"fee_schedule_gap:\s*(.*)", re.S)
_SKIP_CLASSES = re.compile(r"(Context|Ctx)$")


def _convert(exc: BaseException, operation: str):
    if type(exc).__name__ != "PanicException":
        return None
    gap = _GAP.search(str(exc))
    if gap is not None:
        return errors.InvalidParameter(
            gap.group(1).strip(), code="fee_schedule_gap", operation=operation
        )
    match = _FAULT.search(str(exc))
    if match is None:
        return None
    detail = match.group(1).strip()
    return errors.MoneyOverflow(
        f"a booked amount cannot be represented: {detail}",
        code="money_overflow",
        operation=operation,
    )


def _wrap(fn, operation: str):
    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except errors.RaptorBTError as exc:
            # A refusal that names no operation is the wrapped call's.
            if exc.operation is None:
                exc.operation = operation
            raise
        except BaseException as exc:  # PanicException derives from BaseException
            converted = _convert(exc, operation)
            if converted is None:
                raise
            raise converted from None

    return guarded


def install(module) -> None:
    """Wrap the callables of the native ``module`` in place (idempotent)."""
    for name, obj in list(vars(module).items()):
        if name.startswith("_"):
            continue
        if inspect.isclass(obj):
            if _SKIP_CLASSES.search(name):
                continue
            for attr, member in list(vars(obj).items()):
                if attr.startswith("__") or type(member).__name__ != "method_descriptor":
                    continue  # properties, static and class methods, dunders
                try:
                    wrapped = _wrap(member, f"{name}.{attr}")
                    wrapped._raptorbt_guarded = True
                    setattr(obj, attr, wrapped)
                except (TypeError, AttributeError):
                    continue
        elif inspect.isroutine(obj) and not getattr(obj, "_raptorbt_guarded", False):
            wrapped = _wrap(obj, name)
            wrapped._raptorbt_guarded = True
            setattr(module, name, wrapped)
