"""Client order ids on the wire.

The engine names an order ``s{strategy}-{n}``. A broker's tag field has its own
length and charset, so the id is encoded to a tag and decoded back. The map is
injective and needs no state for engine-shaped ids: ``s0-12`` becomes ``s0x12``
where a hyphen is not allowed, and stays ``s0-12`` where it is. Any other id
(one that does not have the engine's shape) is hashed to a fixed-length tag and
remembered in a registry, which is part of the adapter's exported state.
"""

from __future__ import annotations

import hashlib
import re
import threading

_ENGINE = re.compile(r"^s(\d+)-(\d+)$")
_ENGINE_ALNUM = re.compile(r"^s(\d+)x(\d+)$")


class IdCodec:
    """Encode a client order id to a broker tag and back.

    ``charset`` is a regex class body of the characters the broker allows
    (``"A-Za-z0-9"``, ``"A-Za-z0-9_-"``). ``namespace`` is a prefix that marks
    the tags of one node, so two nodes on one account do not read each other's
    orders; tags without it decode to ``None``.
    """

    def __init__(self, *, max_len: int, charset: str = "A-Za-z0-9", namespace: str = "") -> None:
        if max_len < 8:
            raise ValueError("max_len must leave room for an id")
        self._ok = re.compile(f"^[{charset}]*$")
        if not self._ok.match(namespace):
            raise ValueError("namespace uses characters the broker does not allow")
        self.max_len = max_len
        self.namespace = namespace
        self._hyphen = bool(self._ok.match("-"))
        self._registry: dict[str, str] = {}
        self._lock = threading.Lock()

    def encode(self, client_order_id: str) -> str:
        m = _ENGINE.match(client_order_id)
        if m:
            body = client_order_id if self._hyphen else f"s{m.group(1)}x{m.group(2)}"
            tag = self.namespace + body
            if len(tag) <= self.max_len and self._ok.match(tag):
                return tag
        digest = hashlib.sha256(client_order_id.encode()).hexdigest()
        tag = self.namespace + "h" + digest[: self.max_len - len(self.namespace) - 1]
        with self._lock:
            self._registry[tag] = client_order_id
        return tag

    def decode(self, tag: str | None) -> str | None:
        if not tag or not tag.startswith(self.namespace):
            return None
        body = tag[len(self.namespace) :]
        with self._lock:
            if tag in self._registry:
                return self._registry[tag]
        if self._hyphen and _ENGINE.match(body):
            return body
        m = _ENGINE_ALNUM.match(body)
        if m and not self._hyphen:
            return f"s{m.group(1)}-{m.group(2)}"
        return None

    def export(self) -> dict[str, str]:
        with self._lock:
            return dict(self._registry)

    def restore(self, registry: dict[str, str]) -> None:
        with self._lock:
            self._registry.update(registry)
