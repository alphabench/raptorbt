"""Account groups: several strategies on one account.

By default every strategy has its own account. Name an account with
:class:`AccountSpec` and put strategies on it, and they share its cash, its
margin and its account-level risk limits::

    results, accounts = run_strategies(
        [
            (Momentum(), ["NSE:TCS"], {"account": "user42"}),
            {"strategy": MeanReversion(), "symbols": ["NSE:INFY"], "account": "user42"},
            Breakout(),                       # no account: its own, as before
        ],
        data,
        accounts={"user42": AccountSpec(initial_capital=500_000, max_positions=3)},
    )

What a group changes
--------------------

* **Cash and margin.** The account holds one pool. An entry by one strategy
  takes cash the others then do not have, and the locked margin of every
  strategy counts against the free capital of each.
* **Order.** Within one event the strategies on an account step in the order
  they appear in the call (the order of a ``LiveNode``'s strategy list). That
  decides who gets scarce cash: the first one stepped takes it. Accounts do not
  touch each other, so their relative order, and the position of an isolated
  strategy, change nothing.
* **Account-level checks.** Equity, the maintenance requirement, the drawdown
  halt, ``max_positions``, ``max_gross_exposure`` and the daily loss are
  computed over the whole account, not per strategy. They run after each
  strategy's step, on the marks every strategy has so far: a strategy stepped
  later in the same event is still on its previous mark when an earlier one's
  check runs.
* **Halts fan out.** A margin call, the drawdown halt, a daily-loss flatten or
  ``LiveNode.halt_account`` halts every strategy on the account, and their
  ``halted_at`` is the account's applied-event ordinal (the events applied to
  any of its strategies before the halt). A strategy that halts itself with
  ``ctx.halt`` halts only itself.
* **What stays per strategy.** Each strategy's trades, orders, positions and
  P&L, and every setting that is not account-level (position and order size
  limits, stops, instrument configs).

The call's ``config`` supplies those settings, and the account-level ones on a
strategy in a group are the :class:`AccountSpec`'s, not the config's: the
config's ``initial_capital``, ``max_drawdown_pct``, ``max_daily_loss``,
``max_daily_loss_pct``, ``max_gross_exposure`` and ``max_positions`` apply to
isolated strategies only. ``account_type`` and ``leverage`` likewise come from
the spec. ``flatten_on_daily_loss`` and ``liquidate_on_margin_call`` are the
config's and apply to the account.

A strategy's equity curve
-------------------------

A strategy on an account has no cash of its own, so its
``PortfolioResult.result.equity_curve`` is *attributed*: the strategy starts
with an equal share of the account's ``initial_capital`` (capital divided by
the strategies on the account), and its cash moves only by its own entries,
exits and fees. Its drawdown, returns and metrics are computed from that
curve. ``ctx.equity`` reads the same figure. ``ctx.cash`` and
``ctx.free_capital`` are the account's, since that is what the strategy can
spend. The account's own curve, in the :class:`~raptorbt.AccountResult`, is
the sum of the strategies' cash flows and marks.

A group of one is the same account an isolated strategy has: its result is
bit-for-bit the one it would get alone with the same capital and settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from raptorbt.errors import InvalidParameter

#: Most strategies that may share one account. Every step of a strategy on an
#: account touches each other strategy on it, so cost per event grows with the
#: group; past this a group is refused rather than run slowly.
MAX_ACCOUNT_STRATEGIES = 500

_ACCOUNT_TYPES = ("cash", "margin")
_ENTRY_KEYS = ("strategy", "symbols", "account")


def _number(spec, name, value, *, low_open=0.0, high=None, integer=False):
    kind = "a whole number" if integer else "a number"
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if ok and integer:
        ok = isinstance(value, int)
    if ok:
        ok = math.isfinite(value) and value > low_open and (high is None or value <= high)
    if not ok:
        bound = f"above {low_open:g}" + (f" and at most {high:g}" if high is not None else "")
        raise InvalidParameter(
            f"AccountSpec.{name} must be {kind} {bound}, got {value!r}",
            code="invalid_account_spec",
            operation="AccountSpec",
            parameter=name,
            value=value,
        )


@dataclass(frozen=True)
class AccountSpec:
    """One account that several strategies share; see :mod:`raptorbt.strategy.accounts`.

    Args:
        initial_capital: Starting cash of the account.
        account_type: ``"cash"`` (positions are paid for in full) or
            ``"margin"`` (initial margin is locked instead).
        leverage: For ``"margin"``, the leverage that sets the initial margin
            of an instrument without its own rate.
        max_drawdown_pct: Peak-to-trough drawdown of the account's equity, in
            percent, that halts every strategy on it. ``None`` for no limit.
        max_daily_loss: Money the account may lose in a day before new entries
            are refused. ``None`` for no limit.
        max_daily_loss_pct: The same as a percent of the day's starting equity.
        max_gross_exposure: Cap on the notional held across the account.
        max_positions: Cap on the positions open across the account.
    """

    initial_capital: float
    account_type: str = "cash"
    leverage: float = 1.0
    max_drawdown_pct: float | None = None
    max_daily_loss: float | None = None
    max_daily_loss_pct: float | None = None
    max_gross_exposure: float | None = None
    max_positions: int | None = None

    def __post_init__(self):
        _number(self, "initial_capital", self.initial_capital)
        if self.account_type not in _ACCOUNT_TYPES:
            raise InvalidParameter(
                f"AccountSpec.account_type must be 'cash' or 'margin', got {self.account_type!r}",
                code="invalid_account_spec",
                operation="AccountSpec",
                parameter="account_type",
                value=self.account_type,
                valid=list(_ACCOUNT_TYPES),
            )
        _number(self, "leverage", self.leverage)
        for name, high in (
            ("max_drawdown_pct", 100.0),
            ("max_daily_loss", None),
            ("max_daily_loss_pct", 100.0),
            ("max_gross_exposure", None),
        ):
            value = getattr(self, name)
            if value is not None:
                _number(self, name, value, high=high)
        if self.max_positions is not None:
            _number(self, "max_positions", self.max_positions, low_open=0, integer=True)


def check_accounts(accounts, call):
    """``accounts`` as a plain dict of name -> :class:`AccountSpec`."""
    if accounts is None:
        return None
    if not isinstance(accounts, dict) or not all(
        isinstance(k, str) and isinstance(v, AccountSpec) for k, v in accounts.items()
    ):
        raise InvalidParameter(
            "accounts must be a dict of account name (str) -> AccountSpec",
            code="invalid_accounts",
            operation=call,
            parameter="accounts",
            value=type(accounts).__name__,
            valid=["dict[str, AccountSpec]"],
        )
    return dict(accounts)


def _bad_entry(call, item, why):
    return InvalidParameter(
        f"a strategy entry is a strategy, (strategy, symbols), "
        f"(strategy, symbols, {{'account': name}}) or a dict: {why}",
        code="invalid_strategy_entry",
        operation=call,
        parameter="strategies",
        value=item,
    )


def unpack_entries(strategies, accounts, call):
    """Split account names off the strategy entries.

    Returns ``(entries, names)``: ``entries`` in the forms
    :func:`~raptorbt.strategy.multi_runner._normalise` takes, and ``names`` the
    account of each (``None`` for an isolated strategy). A name that is not in
    ``accounts`` raises ``unknown_account``.
    """
    entries, names = [], []
    for item in strategies:
        name: Any = None
        if isinstance(item, dict):
            extra = set(item) - set(_ENTRY_KEYS)
            if "strategy" not in item or extra:
                raise _bad_entry(
                    call, item, "a dict needs 'strategy' and takes 'symbols' and 'account'"
                )
            name = item.get("account")
            item = (
                item["strategy"]
                if item.get("symbols") is None
                else (item["strategy"], item["symbols"])
            )
        elif isinstance(item, tuple) and len(item) == 3:
            options = item[2]
            if not isinstance(options, dict) or set(options) - {"account"}:
                raise _bad_entry(call, item, "the third item is {'account': name}")
            name = options.get("account")
            item = item[:2]
        if name is not None and not isinstance(name, str):
            raise _bad_entry(call, item, f"an account name is a str, got {name!r}")
        if name is not None and name not in (accounts or {}):
            raise InvalidParameter(
                f"{call}: no account named {name!r}"
                + ("" if accounts else "; pass accounts={name: AccountSpec(...)}"),
                code="unknown_account",
                operation=call,
                parameter="account",
                value=name,
                valid=list(accounts or {}),
            )
        entries.append(item)
        names.append(name)
    return entries, names


def check_group_sizes(names, call):
    """Refuse an account with more than :data:`MAX_ACCOUNT_STRATEGIES`."""
    counts: dict[str, int] = {}
    for name in names:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    for name, n in counts.items():
        if n > MAX_ACCOUNT_STRATEGIES:
            raise InvalidParameter(
                f"{call}: account {name!r} has {n} strategies; "
                f"an account holds at most {MAX_ACCOUNT_STRATEGIES}",
                code="account_too_large",
                operation=call,
                parameter="accounts",
                value=n,
                valid=[f"at most {MAX_ACCOUNT_STRATEGIES} strategies per account"],
            )


def plan_batches(names, size):
    """Indices of the strategies in each batch, in the order they run.

    Consecutive batches of ``size`` strategies in input order, except that an
    account is never split: all its strategies go together, at the position of
    its first, and a batch that holds one is as large as it needs to be.
    Without accounts this is plain consecutive chunking. Within a batch the
    strategies of an account keep their input order.
    """
    members: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        if name is not None:
            members.setdefault(name, []).append(i)
    batches: list[list[int]] = []
    current: list[int] = []
    for i, name in enumerate(names):
        if name is None:
            unit = [i]
        elif members[name][0] == i:
            unit = members[name]
        else:
            continue
        if current and len(current) + len(unit) > size:
            batches.append(current)
            current = []
        current = current + unit
    if current:
        batches.append(current)
    return batches
