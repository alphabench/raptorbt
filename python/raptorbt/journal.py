"""The journal: what a node was told, what it decided and what it sent.

A :class:`~raptorbt.LiveNode` given ``journal=Journal(...)`` writes an
append-only, hash-chained record of its life, **before** it acts on what each
record describes:

=================  ========  ===================================================
kind               source    what it holds
=================  ========  ===================================================
``start``          engine    the node's shape (symbols, strategies, execution)
``tick`` ``bar``   feed      one push, with the event and arrival times
``depth`` ``time``
``report``         venue     an execution report the venue sent (``push_report``)
``command``        strategy  a batch of commands, before the engine acts on it
``refusal``        engine    the code the engine refused a batch with
``adapter_request``  engine  a request, **before** the adapter is called
``adapter_result``   engine  what the call came to: ok, refused, unsupported,
                             error, fenced, unknown_outcome, not_sent
``halt``           host or   a halt, from outside (``node.halt``) or inside
                   strategy  (``ctx.halt``)
``unbooked_fill``  engine    a venue fill the book could not book yet
``order_resized``  engine    the engine cut a working order's units (OCO sibling)
``rejection``      engine    the book refused an order or entry (position state,
                             sizing, a limit)
``engine_exit``    engine    a position closed by an engine rule (stop, target,
                             trail, liquidation, time exit, square-off)
``lifecycle``      engine    a contract settled, a corporate action applied or
                             refused
``adoption``       host      a position the account held was given to the book
``annotation``     host      ``node.annotate(ts, kind, detail)``; also a resend
``error``          engine    a push that failed after it was journaled
``reconcile`` ``discrepancy`` ``resume`` ``fence`` ``heartbeat``  recovery and
                             fencing (see :meth:`LiveNode.restore`)
``finish``         host      ``node.finish()``
=================  ========  ===================================================

Every record is encoded canonically: JSON with sorted keys and no spaces,
integers as integers, text escaped to ASCII, and **no float** (a float is
written as the string of its shortest round-trip decimal, ``"100.5"``).
``record.canonical()`` is exactly the bytes the chain is taken over;
:func:`chain_step` is one link: ``sha256(prev_hex || canonical)``, and the
chain starts at ``sha256(canonical header)`` unless it continues an earlier
run (``Journal(header, seq_base=N, chain=prev_head)``).

On disk (``file=``) each record is a frame ``u32 length, u32 crc32,
canonical bytes``. A frame cut by a crash is dropped on reopen (reported in
``torn_bytes``); a corrupt frame with whole frames after it is an error.

``fsync`` says when the file is forced to stable storage. A record is always
handed to the operating system when it is appended, so killing the process
loses nothing under any policy; the policy decides what a power loss can take:

``"every"``
    after every record.
``"orders"`` (default)
    after commands, adapter requests and results, reports, halts, refusals and
    the finish: everything on the path of an order. Market data is not forced.
``N`` (an int)
    after every Nth record.
``"never"``
    never; the operating system flushes when it chooses.

``memory=True`` (or ``memory=N``, the latest N) keeps records for
:meth:`Journal.since`. ``sink=callable`` is called with each ``Record`` before
the node acts on it, so an embedding application can be the durable store; if
it raises, the record is not durable and the node stops (``journal_failed``).
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterable

from raptorbt._raptorbt import Journal, Record, canonical, chain_start, chain_step

__all__ = [
    "Discrepancy",
    "Journal",
    "Record",
    "canonical",
    "chain_start",
    "chain_step",
    "open_for_recovery",
    "read",
    "report_from_payload",
    "verify",
]


def read(path):
    """Read a journal file. Returns ``(header, seq_base, chain_from, records,
    torn_bytes)`` after verifying every checksum and the whole chain."""
    return Journal.read(path)


def verify(records: Iterable[Record], header: dict, *, chain: str | None = None) -> str:
    """Recompute the chain over ``records`` and return the final head, raising
    ``ValueError`` where a record's stored chain differs. ``chain`` is the head
    of the run being continued, else the chain starts at the header."""
    head = chain if chain is not None else chain_start(header)
    for r in records:
        head = chain_step(head, r.canonical())
        if head != r.chain:
            raise ValueError(f"record {r.seq} does not continue the chain")
    return head


def open_for_recovery(path, *, fsync=None, memory=None, sink=None):
    """Reopen a journal file to recover from it: returns ``(journal, records)``.

    The journal appends after the records already in the file (a torn tail is
    cut); pass both to :meth:`LiveNode.restore`.
    """
    header, seq_base, chain_from, records, _torn = Journal.read(path)
    journal = Journal(
        header,
        seq_base=seq_base,
        chain=chain_from,
        file=path,
        fsync=fsync,
        memory=memory,
        sink=sink,
    )
    return journal, records


@dataclass(frozen=True)
class Discrepancy:
    """A difference between what the node holds and what the venue says.

    ``kind`` is one of:

    ``missing_order``
        the node has an order working (or sent with an unknown outcome) that
        the venue does not list as open. It may have filled, been cancelled,
        or never arrived; ``detail`` says whether its request is unconfirmed.
        Push the reports you have, or :meth:`LiveNode.resend` it.
    ``unknown_order``
        the venue has an open order the node never sent.
    ``order_mismatch``
        both know the order and disagree on how much of it has filled.
    ``position_mismatch``
        the net position in a symbol differs.
    ``cash_mismatch``
        the cash differs.
    """

    kind: str
    symbol: str | None = None
    client_order_id: str | None = None
    expected: Any = None
    actual: Any = None
    detail: str = ""

    @property
    def key(self) -> tuple:
        return (self.kind, self.symbol, self.client_order_id)


_REPORT_FIELDS = {
    "accepted": ("venue_order_id", "ts_ns"),
    "rejected": ("reason", "ts_ns"),
    "cancel_ack": ("ts_ns",),
    "cancel_rejected": ("reason", "ts_ns"),
    "replace_rejected": ("reason", "ts_ns"),
    "expired": ("ts_ns",),
    "bust": ("exec_id", "ts_ns"),
}


def report_from_payload(payload: dict):
    """Rebuild the :class:`ExecutionReport` a ``report`` record describes."""
    from raptorbt.execution import ExecutionReport as R

    def num(key):
        value = payload[key]
        return None if value is None else float(value)

    kind = payload["report"]
    if kind == "fill":
        return R.fill(payload["exec_id"], num("qty"), num("price"), payload["ts_ns"], num("fee"))
    if kind == "correction":
        return R.correction(payload["exec_id"], num("qty"), num("price"), payload["ts_ns"])
    if kind == "replace_ack":
        return R.replace_ack(
            payload["ts_ns"], num("new_qty"), num("new_limit"), num("new_trigger")
        )
    args = [payload[f] for f in _REPORT_FIELDS[kind]]
    return getattr(R, kind)(*args)


# -- checkpoint container -------------------------------------------------------

CHECKPOINT_VERSION = 1
_MAGIC = b"RBTCKPT1"


def encode_checkpoint(body: dict) -> bytes:
    data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return _MAGIC + struct.pack("<II", len(data), zlib.crc32(data)) + data


def decode_checkpoint(blob: bytes) -> dict:
    from raptorbt.errors import InvalidData

    def bad(why):
        return InvalidData(
            f"not a usable checkpoint: {why}",
            code="invalid_checkpoint",
            operation="LiveNode.restore",
            parameter="checkpoint",
        )

    if not isinstance(blob, (bytes, bytearray)) or blob[: len(_MAGIC)] != _MAGIC:
        raise bad("bad magic")
    head = len(_MAGIC) + 8
    if len(blob) < head:
        raise bad("truncated")
    size, crc = struct.unpack("<II", blob[len(_MAGIC) : head])
    data = bytes(blob[head : head + size])
    if len(data) != size or zlib.crc32(data) != crc:
        raise bad("checksum mismatch")
    body = json.loads(data)
    if body.get("v") != CHECKPOINT_VERSION:
        raise bad(
            f"version {body.get('v')!r}; this library reads version {CHECKPOINT_VERSION}"
        )
    return body
