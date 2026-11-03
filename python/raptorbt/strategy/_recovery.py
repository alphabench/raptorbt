"""Checkpoint, restore and reconcile for :class:`~raptorbt.LiveNode`.

The engine is deterministic: the same inputs in the same order give the same
books, orders, indicators and ids. A node is therefore rebuilt by **replaying
its journal** into a freshly built node, with the adapter muted, checking that
every record the replay produces is the record the journal holds. A checkpoint
carries that journal (or says where to find it), the state digest to check the
replay against, and the strategies' own state. See
``docs/architecture/journal-and-recovery.md``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Iterable, Mapping

from raptorbt import journal as jr
from raptorbt.errors import InvalidData, InvalidParameter, StateError, UnsupportedOperation
from raptorbt.execution import AdapterError
from raptorbt.journal import Discrepancy

# Strategy methods a live node calls itself (not routed through api.HOOKS): the recovery sequence
# and the instrument set changing.
RECOVERY_HOOKS: tuple[str, ...] = ("on_discrepancy", "on_instrument_added", "on_instrument_retired")

#: Statuses of an order the venue may still act on.
_WORKING = {
    "submitted",
    "accepted",
    "triggered",
    "partiallyfilled",
    "pendingcancel",
    "pendingreplace",
}

#: Record kinds the replay driver feeds in; every other kind is either made by
#: the engine while those run (and checked against the journal) or is passed
#: through unchanged.
_INPUTS = {"tick", "bar", "depth", "time", "report", "halt", "annotation", "instrument_add", "instrument_retire", "group_begin", "group_end"}
_DERIVED = {"command", "refusal", "adapter_request", "adapter_result", "error", "fence", "start", "restatement", "unbooked_fill", "order_resized", "adoption", "engine_exit", "lifecycle", "rejection", "simulated_report"}
_PASSTHROUGH = {"reconcile", "discrepancy", "resume", "heartbeat", "state", "epoch"}


def _f(x):
    return None if x is None else float(x)


# -- state ---------------------------------------------------------------------


def state_summary(node) -> dict:
    """What the node holds, in plain values: per strategy its equity, cash,
    positions and orders, and the orders sent to a venue."""
    rt = node._rt
    out = {"strategies": [], "venue_orders": []}
    for k, ctx in enumerate(node._ctxs):
        view = node._views[k]
        positions = {}
        orders = []
        for sym in node._symbols_of[k]:
            positions[sym] = [
                [p.direction, float(p.size), float(p.entry_price)] for p in ctx.positions(sym)
            ]
            orders.extend(
                [o.order_id, o.client_id, o.side, o.kind, o.status, float(o.filled_qty), sym]
                for o in ctx.open_orders(sym)
            )
        out["strategies"].append(
            {
                "equity": float(view.equity()),
                "cash": float(view.cash()),
                "halted": bool(view.is_halted()),
                "positions": positions,
                "orders": orders,
            }
        )
    if node._added:
        # Only a node whose instrument set moved says so: older digests stay valid.
        out["instruments"] = [list(node._symbols), sorted(node._retired)]
    out["venue_orders"] = [list(o) for o in rt.venue_orders()]
    out["unconfirmed"] = sorted(rt.unconfirmed())
    return out


def state_digest(node) -> str:
    return jr.chain_start(state_summary(node))


# -- checkpoint ------------------------------------------------------------------


def strategy_states(node) -> list:
    out = []
    for s in node._strategies:
        fn = getattr(s, "state", None)
        value = fn() if callable(fn) else None
        try:
            out.append(None if value is None else json.dumps(value, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as e:
            raise InvalidData(
                f"{type(s).__name__}.state() must return JSON-serialisable plain data: {e}",
                code="invalid_checkpoint",
                operation="LiveNode.checkpoint",
                parameter="state",
            ) from None
    return out


def checkpoint(node, include_log: bool = True) -> bytes:
    j = node._journal
    if j is None:
        raise StateError(
            "checkpoint() needs a node built with journal=",
            code="invalid_checkpoint",
            operation="LiveNode.checkpoint",
        )
    count, chain, header = j.head()
    log = None
    if include_log:
        try:
            log = [r.canonical().decode() for r in j.since(j.seq_base)]
        except Exception as e:
            raise InvalidParameter(
                "checkpoint() with include_log=True needs every record since seq_base in "
                "memory (Journal(memory=True)); use include_log=False and pass records= to "
                f"restore ({e})",
                code="invalid_checkpoint",
                operation="LiveNode.checkpoint",
                parameter="include_log",
            ) from None
    return jr.encode_checkpoint(
        {
            "v": jr.CHECKPOINT_VERSION,
            "lib": _version(),
            "header": header,
            "seq_base": j.seq_base,
            "chain_from": j.chain_from,
            "count": count,
            "chain": chain,
            "epoch": node._epoch,
            "digest": state_digest(node),
            "states": strategy_states(node),
            "log": log,
        }
    )


def _version() -> str:
    import raptorbt

    return getattr(raptorbt, "__version__", "unknown")


# -- restore ---------------------------------------------------------------------


def restore(
    cls,
    checkpoint_bytes,
    strategies,
    symbols,
    config=None,
    *,
    journal=None,
    records=None,
    reconcile: bool | str = True,
    accept=None,
    on_discrepancy: Callable[[Discrepancy], None] | None = None,
    **kwargs,
):
    ck = jr.decode_checkpoint(checkpoint_bytes) if checkpoint_bytes is not None else None
    if records is None:
        if ck is None or ck.get("log") is None:
            raise InvalidParameter(
                "restore() needs the journal's records: pass records= (from "
                "raptorbt.journal.open_for_recovery or your own store), or a checkpoint "
                "taken with include_log=True",
                code="invalid_checkpoint",
                operation="LiveNode.restore",
                parameter="records",
            )
        records = [s.encode() for s in ck["log"]]
    records = list(records)
    if not records:
        raise InvalidParameter(
            "restore() was given no records", code="invalid_checkpoint",
            operation="LiveNode.restore", parameter="records",
        )
    if journal is None:
        if ck is None:
            raise InvalidParameter(
                "restore() without a checkpoint needs journal= (open_for_recovery)",
                code="invalid_checkpoint", operation="LiveNode.restore", parameter="journal",
            )
        journal = jr.Journal(
            ck["header"], seq_base=ck["seq_base"], chain=ck["chain_from"], memory=True
        )
    first = _record_dict(records[0])
    if first["kind"] != "start":
        raise InvalidData(
            f"the journal begins with {first['kind']!r}, not 'start'",
            code="invalid_checkpoint", operation="LiveNode.restore", parameter="records",
        )
    if any(_record_dict(r)["kind"] == "finish" for r in records):
        raise StateError(
            "the journal is of a node that finished; there is nothing to recover",
            code="node_finished", operation="LiveNode.restore",
        )
    _check_namespace(first["payload"], kwargs.get("execution"))
    journal_epochs = [
        _record_dict(r)["payload"]["epoch"] for r in records if _record_dict(r)["kind"] == "epoch"
    ]
    # The node is rebuilt with the epoch it was built with, so the replay
    # reproduces its `epoch` record; the epoch asked for is taken afterwards.
    built_epoch = journal_epochs[0] if journal_epochs else None
    last_epoch = journal_epochs[-1] if journal_epochs else None
    epoch = kwargs.pop("epoch", None)
    if last_epoch is not None:
        # A journal that holds an epoch belongs to a node that fenced the account with it. The
        # node that restores it takes the account over, and only a newer epoch fences the old
        # one: restoring under the epoch the journal ends on would leave two nodes with one.
        if epoch is None:
            raise StateError(
                f"the journal holds epoch {last_epoch}: restore() needs epoch= greater than it, "
                "so the node that wrote it is fenced out",
                code="epoch_required", operation="LiveNode.restore",
            )
        if epoch <= last_epoch:
            raise StateError(
                f"restore() was given epoch {epoch}, but the journal already holds epoch "
                f"{last_epoch}: a restored node takes a newer one, and never an older or the same",
                code="stale_epoch", operation="LiveNode.restore",
            )
    if epoch is not None and kwargs.get("execution") is not None and not getattr(
        kwargs["execution"], "can_fence", False
    ):
        raise InvalidParameter(
            "epoch= fences a stale node through the adapter's fence(epoch), and only an adapter "
            "with can_fence true enforces it (a BrokerAdapter needs a host lease=), so the epoch "
            "would protect nothing",
            code="fence_unsupported", operation="LiveNode.restore", parameter="epoch", value=epoch,
        )
    if "strategy_states" not in kwargs:
        initial = first["payload"].get("strategy_states")
        kwargs["strategy_states"] = None if not initial else [None if s is None else json.loads(s) for s in initial]
    node = cls(
        strategies, symbols, config, journal=journal, epoch=built_epoch, _replay=records, **kwargs
    )
    node._on_discrepancy = on_discrepancy
    try:
        _replay(node, journal, ck)
    except BaseException:
        node._rt.close()
        raise
    node._rt.set_replay(False)
    node._replaying = False
    node._state = "adding" if node._requests else "live"
    if epoch is not None:
        try:
            node._claim_epoch(epoch, recorded=last_epoch)
        except BaseException:
            node._rt.close()
            raise
    if node._adapter_kind != "adapter":
        return node
    node._state = "recovering"
    _teach_adapter(node)
    if reconcile == "skip":
        node._resume_record({"skipped": True})
        node._state = "live"
    elif reconcile:
        found = node.reconcile()
        if not found:
            node.resume()
        elif accept is not None:
            node.resume(accept=accept)
    return node


def _teach_adapter(node) -> None:
    """Give a fresh adapter the orders the rebuilt node holds as working at the venue.

    The replay ran with the adapter muted, so an adapter built after the restart has never
    seen them. Without this it ignores their fills and refuses to cancel them. An adapter with
    no ``restore_orders`` must say it ``keeps_own_records`` (a venue of your own that remembers
    the node's orders itself); one that does neither is refused, because it would silently
    ignore the node's working orders."""
    teach = getattr(node._execution, "restore_orders", None)
    if not callable(teach):
        if getattr(node._execution, "keeps_own_records", False) is True:
            return
        node._rt.close()
        raise StateError(
            f"{type(node._execution).__name__} cannot account for the orders this node holds as "
            "working: it has no restore_orders(rows) to adopt them and does not declare "
            "keeps_own_records = True. A restored node must be able to cancel and follow them",
            code="adapter_cannot_adopt", operation="LiveNode.restore",
            parameter="execution", value=type(node._execution).__name__,
            valid=["an adapter with restore_orders(rows)", "an adapter with keeps_own_records = True"],
        )
    rows = [
        {"request": req, "status": status, "filled_qty": filled, "avg_price": avg}
        for req, status, filled, avg in node._rt.working_requests()
    ]
    try:
        teach(rows)
    except AdapterError as e:
        node._rt.close()
        raise StateError(
            f"the adapter cannot account for the orders this node holds as working: {e}. "
            "A restored node must be able to cancel and follow them",
            code="adapter_cannot_adopt", operation="LiveNode.restore",
            parameter="execution", value=type(node._execution).__name__,
            valid=["an adapter with restore_orders(rows)", "an adapter with keeps_own_records = True"],
        ) from e


def _check_namespace(start_payload: dict, adapter) -> None:
    """The adapter must be the one the journal was written for: its orders are found by
    the namespace in their ids, and a different one would not recognise them (and
    ``resend`` would place a second live order)."""
    written = start_payload.get("namespace")
    if written is None or adapter is None:
        return
    if getattr(adapter, "namespace", None) != written:
        raise StateError(
            f"the journal was written for an adapter in namespace {written!r}; the adapter "
            f"given to restore() is in {getattr(adapter, 'namespace', None)!r}. It cannot "
            "recognise this node's working orders",
            code="adapter_namespace_mismatch", operation="LiveNode.restore",
            parameter="execution", value=getattr(adapter, "namespace", None), valid=[written],
        )


def _record_dict(r) -> dict:
    """A record, whether a ``Record`` or its canonical bytes, as a dict."""
    if isinstance(r, (bytes, bytearray)):
        return json.loads(r)
    return {
        "seq": r.seq,
        "kind": r.kind,
        "ts_event": r.ts_event,
        "ts_arrival": r.ts_arrival,
        "source": r.source,
        "client_order_id": r.client_order_id,
        "payload": r.payload,
    }


def _replay(node, journal, ck) -> None:
    """Feed the journal's inputs to the node until every record is reproduced."""
    check_at = ck["count"] if ck is not None else None
    checked = check_at is None
    node._replaying = True
    while journal.replaying:
        if not checked and journal.count == check_at:
            _check_digest(node, journal, ck)
            checked = True
        rec = journal.peek()
        kind, payload = rec.kind, rec.payload
        try:
            _apply(node, journal, rec, kind, payload)
        except Exception:
            # A push that failed in the original run failed here too, and the
            # engine wrote the same `error` record. Anything else is a replay
            # that went its own way.
            if journal.last_kind != "error" or journal.count <= rec.seq:
                raise
        node._rt.take_replay_errors()
    if not checked:
        if journal.count == check_at:
            _check_digest(node, journal, ck)
        else:
            raise StateError(
                f"the journal holds {journal.count} records; the checkpoint was taken at {check_at}",
                code="replay_diverged", operation="LiveNode.restore",
            )


def _check_digest(node, journal, ck) -> None:
    count, chain, _ = journal.head()
    if (count, chain) != (ck["count"], ck["chain"]):
        raise StateError(
            f"the recomputed chain head ({count}, {chain[:12]}...) is not the checkpoint's "
            f"({ck['count']}, {str(ck['chain'])[:12]}...): the records are not the ones the "
            "checkpoint was taken over",
            code="replay_diverged", operation="LiveNode.restore",
        )
    got = state_digest(node)
    if got != ck["digest"]:
        raise StateError(
            "the rebuilt node's state differs from the checkpoint's: the node is not the one "
            "that wrote the journal, or a strategy is not deterministic",
            code="replay_diverged", operation="LiveNode.restore",
        )


def _apply(node, journal, rec, kind, p) -> None:
    arrival = rec.ts_arrival
    if kind == "tick":
        node.push_tick(
            p["symbol"], rec.ts_event, float(p["ltp"]), float(p["bid"]), float(p["ask"]),
            float(p["buy_qty_delta"]), float(p["sell_qty_delta"]), float(p["ltq"]),
            float(p["bid_qty"]), float(p["ask_qty"]), float(p["oi"]), arrival_ns=arrival,
        )
    elif kind == "bar":
        node.push_bar(
            p["symbol"], p["timestamp"], float(p["open"]), float(p["high"]), float(p["low"]),
            float(p["close"]), float(p["volume"]), arrival_ns=arrival,
        )
    elif kind == "depth":
        node.push_depth(
            p["symbol"], rec.ts_event,
            [(float(a), float(b)) for a, b in p["bids"]],
            [(float(a), float(b)) for a, b in p["asks"]],
            arrival_ns=arrival,
        )
    elif kind == "time":
        node.advance_time(p["timestamp"])
    elif kind == "group_begin":
        node.begin_group(p["timestamp"])
    elif kind == "group_end":
        node.flush()
    elif kind == "report":
        node.push_report(rec.client_order_id, jr.report_from_payload(p), arrival_ns=arrival)
    elif kind == "halt" and rec.source == "host":
        if "account" in p:
            node.halt_account(p["account"], p["flatten"], p["reason"])
        else:
            node.halt(p["strategy"], p["flatten"], p["reason"])
    elif kind == "instrument_add" and rec.source == "host":
        from raptorbt import InstrumentSpec

        node.add_instrument(
            p["symbol"],
            None if p["spec"] is None else InstrumentSpec.from_json(p["spec"]),
            strategies=list(p["strategies"]),
            history=p["history"],
            direction=p["direction"],
            bar_ns=p["offset"] or None,
        )
    elif kind == "instrument_retire" and rec.source == "host":
        node.retire_instrument(p["symbol"])
    elif kind == "annotation":
        if p.get("what") == "resend":
            node.resend(rec.client_order_id)
        else:
            node.annotate(rec.ts_event, p["kind"], p["detail"])
    elif kind in _PASSTHROUGH or kind == "finish":
        if kind == "finish":
            raise StateError("the journal ends with a finished node", code="node_finished",
                             operation="LiveNode.restore")
        journal.append(kind, rec.ts_event, p, ts_arrival=arrival, source=rec.source,
                       client_order_id=rec.client_order_id)
    else:
        raise StateError(
            f"the journal holds a {kind!r} record from {rec.source!r} at {rec.seq} that the "
            "replay did not produce",
            code="replay_diverged", operation="LiveNode.restore",
        )


# -- reconcile ---------------------------------------------------------------------


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def venue_positions(raw) -> dict[str, float]:
    """The net quantity per symbol of what an adapter's ``positions()`` returned: a
    ``{symbol: quantity}`` mapping, or a list of rows with ``symbol`` and ``net_qty`` (the
    :class:`~raptorbt.adapters.VenuePosition` a broker adapter returns; ``quantity`` also
    reads). Rows of one symbol add."""
    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    out: dict[str, float] = {}
    for row in raw:
        symbol = _get(row, "symbol")
        qty = _get(row, "net_qty")
        if qty is None:
            qty = _get(row, "quantity")
        if symbol is None or qty is None:
            raise InvalidData(
                f"positions() returned a row without a symbol and a net_qty: {row!r}",
                code="invalid_execution", operation="LiveNode.reconcile", parameter="positions",
            )
        out[symbol] = out.get(symbol, 0.0) + float(qty)
    return out


def venue_cash(raw) -> tuple[float | None, dict]:
    """``(cash, snapshot)`` of what an adapter's ``balances()`` returned.

    A ``{"cash": x}`` mapping is compared as it is. A :class:`~raptorbt.adapters.VenueBalances`
    is compared like with like: the node's cash is what the account holds *before* margin is
    set aside for positions, so the balance it is compared with is the venue's ``total`` (the
    whole balance) when the broker gives one, else ``available + used``. ``available`` alone is
    what is left to trade with after margin, which the node's cash is not: it is never compared
    on its own (``cash`` is ``None`` and the snapshot says what was missing). The snapshot's
    ``compared`` names the field used."""
    if isinstance(raw, Mapping):
        snap = {k: _f(v) if isinstance(v, (int, float)) else v for k, v in sorted(raw.items())}
        if raw.get("cash") is not None:
            snap["compared"] = "cash"
        return (None if raw.get("cash") is None else float(raw["cash"])), snap
    available = _get(raw, "available")
    if available is None:
        raise InvalidData(
            f"balances() returned neither a mapping with 'cash' nor an object with 'available': {raw!r}",
            code="invalid_execution", operation="LiveNode.reconcile", parameter="balances",
        )
    snap = {"available": float(available), "currency": _get(raw, "currency")}
    for name in ("used", "total"):
        if _get(raw, name) is not None:
            snap[name] = float(_get(raw, name))
    if "total" in snap:
        snap["compared"] = "total"
        return snap["total"], snap
    if "used" in snap:
        snap["compared"] = "available + used"
        return float(available) + snap["used"], snap
    return None, snap


def reconcile(node, cash_tolerance: float = 0.0) -> list[Discrepancy]:
    """Ask the adapter what the venue holds and compare it with the node."""
    adapter = node._execution
    has_orders = callable(getattr(adapter, "open_orders", None))
    has_positions = callable(getattr(adapter, "positions", None))
    has_balances = callable(getattr(adapter, "balances", None))
    if node._adapter_kind != "adapter" or not (has_orders or has_positions or has_balances):
        raise UnsupportedOperation(
            "reconcile() needs an execution adapter with open_orders(), positions() or balances()",
            code="no_reconcile", operation="LiveNode.reconcile",
        )
    rt = node._rt
    found: list[Discrepancy] = []
    sent = rt.venue_orders()
    unconfirmed = set(rt.unconfirmed())
    snapshot: dict[str, Any] = {}
    if has_orders:
        venue = {}
        for o in adapter.open_orders():
            # An order the node did not send has no client id of ours: it is keyed by
            # the venue's id, so two such orders are two discrepancies.
            cid = _get(o, "client_order_id")
            if cid is None:
                vid = _get(o, "venue_order_id")
                if vid is None:
                    raise InvalidData(
                        f"open_orders() returned an order with neither a client_order_id nor a venue_order_id: {o!r}",
                        code="invalid_execution", operation="LiveNode.reconcile", parameter="open_orders",
                    )
                cid = f"venue:{vid}"
            venue[cid] = o
        snapshot["open_orders"] = sorted(venue)
        known = {row[0] for row in sent}
        # Orders an exit rule held at the venue will make: the venue lists them only once the
        # rule fires, so their absence from its open orders is not a missing order.
        held = set(rt.venue_held_orders())
        for cid, strat, sym, side, qty, filled, status in sent:
            ours = status in _WORKING
            theirs = venue.get(cid)
            if ours and theirs is None and cid in held:
                if cid in unconfirmed:
                    found.append(Discrepancy(
                        "exit_rule_unconfirmed", sym, cid, status, None,
                        "the venue may hold the exit rule that fires this order: its answer is unknown",
                    ))
                continue
            if ours and theirs is None:
                why = "its request has no confirmed outcome" if cid in unconfirmed else "the venue lists it as not open"
                found.append(Discrepancy("missing_order", sym, cid, status, None, why))
            elif theirs is not None:
                rt.confirm(cid)
                vfilled = _get(theirs, "filled_qty")
                if vfilled is not None and abs(float(vfilled) - filled) > 1e-9:
                    found.append(Discrepancy("order_mismatch", sym, cid, filled, float(vfilled),
                                             "filled quantity"))
                elif not ours:
                    found.append(Discrepancy("order_mismatch", sym, cid, status, "open",
                                             "the node holds it closed"))
        for cid, o in venue.items():
            if cid not in known:
                found.append(Discrepancy("unknown_order", _get(o, "symbol"), cid, None,
                                         _get(o, "quantity"), "the node never sent it"))
    if has_positions:
        theirs = venue_positions(adapter.positions())
        snapshot["positions"] = dict(sorted(theirs.items()))
        # A position the broker names by its own key and the adapter could not map to a symbol
        # of this node is not compared with anything: say so, never skip it.
        resolve = getattr(adapter, "symbol_of", None)
        mine_symbols = set(node._symbols)
        for key, vq in sorted(theirs.items()):
            if key in mine_symbols:
                continue
            sym = resolve(key) if callable(resolve) else None
            if sym is not None and sym in mine_symbols:
                theirs.pop(key)
                theirs[sym] = theirs.get(sym, 0.0) + vq
                continue
            found.append(Discrepancy(
                "unmapped_position", key, None, None, vq,
                "the venue holds a position under a key that is no symbol of this node",
            ))
        for sym in node._symbols:
            mine = math.fsum(
                p.direction * float(p.size)
                for k in range(len(node._ctxs))
                if sym in node._symbols_of[k]
                for p in node._ctxs[k].positions(sym)
            )
            vq = theirs.get(sym, 0.0)
            if abs(mine - vq) > 1e-9 * max(1.0, abs(mine), abs(vq)):
                found.append(Discrepancy("position_mismatch", sym, None, mine, vq, ""))
    if has_balances:
        cash, snapshot["balances"] = venue_cash(adapter.balances())
        field = snapshot["balances"].get("compared")
        if cash is not None:
            mine = math.fsum(float(v.cash()) for v in node._views)
            if abs(mine - cash) > cash_tolerance:
                found.append(Discrepancy(
                    "cash_mismatch", None, None, mine, cash,
                    f"the node's cash against the venue's {field or 'cash'}, tolerance {cash_tolerance}",
                ))
        else:
            snapshot["balances"]["compared"] = field or "none: the venue gave no balance comparable with the node's cash"
    ts = rt.clock()[0]
    node._journal.append("reconcile", ts, {"found": len(found), "venue": snapshot}, source="host")
    for d in found:
        node._journal.append(
            "discrepancy", ts,
            {"kind": d.kind, "symbol": d.symbol, "expected": d.expected, "actual": d.actual,
             "detail": d.detail},
            source="host", client_order_id=d.client_order_id,
        )
    node._pending = {d.key: d for d in found}
    for d in found:
        _deliver(node, d)
    return found


def _strategy_of(client_order_id: str) -> int | None:
    """The strategy index in a venue id ``s{strategy}-{seq}``."""
    head, _, tail = client_order_id.partition("-")
    if head[:1] == "s" and head[1:].isdigit() and tail.isdigit():
        return int(head[1:])
    return None


def _deliver(node, d: Discrepancy) -> None:
    targets = range(len(node._strategies))
    if d.client_order_id:
        parsed = _strategy_of(d.client_order_id)
        if parsed is not None and parsed < len(node._strategies):
            targets = [parsed]
    elif d.symbol is not None:
        targets = [k for k in targets if d.symbol in node._symbols_of[k]]
    for k in targets:
        node._strategies[k].on_discrepancy(node._ctxs[k], d)
    if node._on_discrepancy is not None:
        node._on_discrepancy(d)
