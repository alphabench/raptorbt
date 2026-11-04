"""Strategy base class."""

from __future__ import annotations

import logging

from raptorbt._raptorbt import CommandQueue
from raptorbt.errors import InvalidOrder, InvalidSubscription
from raptorbt.strategy import api
from raptorbt.strategy.cache import Cache
from raptorbt.strategy.clock import Clock
from raptorbt.strategy.config import StrategyConfig
from raptorbt.strategy.context import StrategyContext
from raptorbt.strategy.orders import Market
from raptorbt.strategy.orders.types import Twap, _OrderBase, check_term


def _structures_first(hook):
    """Wrap a subclass's ``on_order_event`` so the structure groups hear the event first."""
    import functools

    @functools.wraps(hook)
    def wrapped(self, ctx, event):
        self._structure_event(ctx, event)
        # a handler that calls ``super().on_order_event`` reaches the base class's forward
        # inside this call; the event has been heard already
        outer = self.__dict__.get("_structure_heard")
        self.__dict__["_structure_heard"] = event
        try:
            hook(self, ctx, event)
        finally:
            self.__dict__["_structure_heard"] = outer

    wrapped._raptorbt_structures = True  # type: ignore[attr-defined]
    return wrapped


class Strategy:
    """Base class for event-driven strategies.

    Subclass and override the hooks you need. All hooks default to no-ops.
    Order intents are emitted from inside ``on_bar`` via :meth:`enter` /
    :meth:`close_position`; the engine applies them on the same bar using its
    configured fill model, then feeds resulting events back through the
    ``on_order_*`` / ``on_position_*`` hooks.
    """

    #: The run's API state (subscriptions, views); set by the runner.
    _api = None

    #: What the strategy needs, declared as data (see ``raptorbt.strategy.Requirements``);
    #: a subclass sets ``requirements = Requirements(...)``.
    requirements = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        api.check_class(cls)
        own = cls.__dict__.get("on_order_event")
        if own is not None and not getattr(own, "_raptorbt_structures", False):
            setattr(cls, "on_order_event", _structures_first(own))

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config if config is not None else StrategyConfig()
        # Orders, cancels, closes and entry intents queued by the hooks; the
        # runner takes them out each event. Orders are converted to plain
        # data as they are submitted.
        self._queue = CommandQueue()
        self._order_seq = 0
        # (step, unit) bar subscriptions declared in on_start.
        self._bar_subscriptions: list[tuple[int, str]] = []
        #: Bar-driven clock: ``set_time_alert`` / ``set_timer``; fresh per run.
        self.clock = Clock()
        #: Event-sourced order/trade cache; fresh per run.
        self.cache = Cache()
        # (indicator, stream_id) registrations; stream_id None = primary bars.
        self._indicators: list[tuple[object, int | None]] = []

    # -- lifecycle hooks ----------------------------------------------------
    #
    # The canonical names. Hooks marked "deprecated" further down receive the
    # same events under their old names; define one or the other, never both.

    def on_init(self, ctx: StrategyContext) -> None:
        """Called once per run, before :meth:`on_start` and before any
        position is adopted. Declare subscriptions and indicators here; the
        account is not yet seeded, so read positions and equity in
        :meth:`on_start`."""

    def on_start(self, ctx: StrategyContext) -> None:
        """Called once before the first bar.

        The full OHLCV arrays on ``ctx`` are available here for indicator
        precomputation.
        """

    def on_bar(self, ctx: StrategyContext) -> None:
        """Called once per bar, in ascending time order.

        Defined as ``on_bar(self, ctx)`` it sees primary bars only. Defined as
        ``on_bar(self, ctx, event)`` it receives the bar as ``event`` and also
        every completed composite bar (``event.timeframe`` is ``None`` for a
        primary bar and a label such as ``"5m"`` for a composite one), which
        replaces :meth:`on_composite_bar`.
        """

    def on_market_data(self, ctx: StrategyContext, event) -> None:
        """Every bar, composite bar, trade, quote and book the strategy
        receives, after the hook specific to that event."""

    def on_trade(self, ctx: StrategyContext, tick) -> None:
        """A trade printed. Only fires in tick runs. Replaces
        :meth:`on_trade_tick`; see there for the book it sees."""

    def on_book(self, ctx: StrategyContext, book) -> None:
        """The order book changed. Only fires in tick runs with depth data.
        Replaces :meth:`on_order_book`."""

    def on_timer(self, ctx: StrategyContext, event) -> None:
        """Called for each due clock alert/timer, *before* the bar's data
        handlers. Replaces :meth:`on_time_event`."""

    def on_fill(self, ctx: StrategyContext, event) -> None:
        """Called when an entry or exit fill occurs. Replaces
        :meth:`on_order_filled`."""

    def on_position_event(self, ctx: StrategyContext, event) -> None:
        """Called after a position opens (``event.kind == "entered"``) or
        closes (``"exited"``; ``event.trade`` is the round trip). Replaces
        :meth:`on_position_opened` and :meth:`on_position_closed`."""

    def on_risk_event(self, ctx: StrategyContext, event) -> None:
        """Called when a risk limit refuses an order or stops trading.

        ``event.kind`` is ``"risk_rejected"`` (an order or entry was refused;
        ``event.reject_reason`` is ``order_size``, ``max_position``,
        ``max_exposure``, ``daily_loss`` or ``killed``) or ``"risk_halted"``
        (the day's loss limit was reached, ``reject_reason`` is
        ``"daily_loss"``; or the strategy was halted, ``reject_reason`` is
        the halt reason). A refused order also reaches ``on_order_rejected``.
        """

    def state(self):
        """Plain data (JSON-serialisable: dicts, lists, str, numbers, bool,
        None) that is your own state and is not derived from the market, to be
        carried across runs by ``LiveNode.checkpoint()``. Default: none.

        Whatever the engine owns (positions, orders, cash, indicators) is
        rebuilt by the node; keep here only what your hooks computed and
        nothing else knows, such as a regime flag set by an operator.
        """
        return None

    def load_state(self, state) -> None:
        """Receive what :meth:`state` returned in an earlier run, before
        :meth:`on_start`. Called only when a node is built with
        ``strategy_states=``."""

    def on_discrepancy(self, ctx: StrategyContext, discrepancy) -> None:
        """A recovery found the venue and the node disagree
        (:class:`raptorbt.journal.Discrepancy`). Informational: the node does
        not resume until the host accepts or resolves it, and orders queued here
        are sent with the first event after it resumes."""

    def on_instrument_added(self, ctx: StrategyContext, symbol: str) -> None:
        """A :class:`~raptorbt.LiveNode` added ``symbol`` to this strategy's symbols (or brought a
        retired one back): it is in ``ctx.symbols`` and takes data from now on. Orders queued here
        are sent with the strategy's next event."""

    def on_instrument_retired(self, ctx: StrategyContext, symbol: str) -> None:
        """A :class:`~raptorbt.LiveNode` retired ``symbol``: it left ``ctx.symbols`` and takes no
        more data. It held no position and no working order of this strategy."""

    def on_stop(self, ctx: StrategyContext) -> None:
        """Called once after the last bar, before finalization."""

    # -- deprecated names ---------------------------------------------------

    def on_time_event(self, ctx: StrategyContext, event) -> None:
        """Deprecated: use :meth:`on_timer`.

        Called for each due clock alert/timer, *before* the bar's data
        handlers — the scheduled time precedes the bar that revealed it."""

    def on_trade_tick(self, ctx: StrategyContext, tick) -> None:
        """Deprecated: use :meth:`on_trade`.

        A trade printed. Only fires in tick runs.

        ``ctx.best_bid`` / ``ctx.best_ask`` hold the last book observed
        *before* this print — the quote from the same feed row arrives in
        the following :meth:`on_quote`, so acting on it here would be
        reading a book this very print already moved.
        """

    def on_quote(self, ctx: StrategyContext, quote) -> None:
        """The top of book changed. Only fires in tick runs.

        Quotes do not fill orders. An order submitted here rests and matches
        against the next print, which is the first evidence of a trade at
        that price.
        """

    def on_order_book(self, ctx: StrategyContext, book) -> None:
        """Deprecated: use :meth:`on_book`.

        The order book changed. Only fires in tick runs with depth data.

        Like quotes, books do not fill orders — displayed size is intent,
        not a trade. An order submitted here rests and matches on the next
        print.
        """

    def on_composite_bar(self, ctx: StrategyContext, bar) -> None:
        """Deprecated: use ``on_bar(self, ctx, event)``.

        Called when a subscribed higher-timeframe bar completes.

        ``bar`` is a :class:`~raptorbt.strategy.context.CompositeBar`;
        dispatched *before* ``on_bar`` of the primary bar that completed it,
        since the composite closed strictly earlier.
        """

    # -- order/position event hooks -----------------------------------------

    def on_order_filled(self, ctx: StrategyContext, event) -> None:
        """Deprecated: use :meth:`on_fill`.

        Called when an entry or exit fill occurs."""

    def on_order_rejected(self, ctx: StrategyContext, event) -> None:
        """Called when an entry intent or order is refused."""

    def on_order_accepted(self, ctx: StrategyContext, event) -> None:
        """Called when a submitted order starts working."""

    def on_order_triggered(self, ctx: StrategyContext, event) -> None:
        """Called when a stop-limit's trigger fires."""

    def on_order_canceled(self, ctx: StrategyContext, event) -> None:
        """Called when an order is canceled (explicitly or IOC/FOK)."""

    def on_order_expired(self, ctx: StrategyContext, event) -> None:
        """Called when an order's time-in-force lapses."""

    def on_order_event(self, ctx: StrategyContext, event) -> None:
        """Called for every order/account event, after its granular hook. The base class
        forwards the event to the strategy's structure groups (hedge-first sequencing and the
        leg-failure close run here, in the step of the fill); a subclass's own handler runs
        after that."""
        self._structure_event(ctx, event)

    def _structure_event(self, ctx, event) -> None:
        if self.__dict__.get("_structure_heard") is event:
            return
        book = self.__dict__.get("_structures")
        if book is not None and book.cache is self.cache:
            # A fill's rules of time use the fill's instant: under next_bar_open the bar's open,
            # not the stamp of the bar's close.
            fill_ns = getattr(ctx, "fill_ns", None)
            book.on_event(event, fill_ns if fill_ns is not None else self._event_ns(ctx, None))

    def on_algo_started(self, ctx: StrategyContext, event) -> None:
        """An execution schedule was registered. Slices arrive afterwards
        as ordinary order events, with client ids of ``"<parent>#<n>"``."""

    def on_algo_completed(self, ctx: StrategyContext, event) -> None:
        """A schedule released its last slice, or was cancelled.

        "Completed" means fully *released*, not necessarily fully filled.
        """

    def on_margin_call(self, ctx: StrategyContext, event) -> None:
        """Called when equity breaches the maintenance requirement (margin
        accounts). ``event.price`` carries the equity, ``event.size`` the
        requirement; new entries halt for the rest of the run."""

    def on_contract_event(self, ctx: StrategyContext, event) -> None:
        """Called when a contract the strategy holds expires and settles
        (``event.kind == "contract_settled"``), or when its settlement was
        refused (``"lifecycle_refused"``).

        ``event.outcome`` is ``"settled"`` (cash), or for physical settlement
        ``"exercised"``, ``"assigned"``, ``"expired"`` (out of the money) or
        ``"delivered"`` (a future); ``event.price`` is the price the position
        closed at and ``event.source`` where it came from (``"declared"``,
        ``"underlying"`` or ``"last_price"``). ``event.quantity`` is the
        underlying taken in (positive) or given away (negative) at
        ``event.amount``. The position's own close reaches
        :meth:`on_position_event` first, with ``exit_reason == "settlement"``.
        Orders on an expired contract are rejected with reason ``expired``."""

    def on_corporate_action(self, ctx: StrategyContext, event) -> None:
        """Called when a declared corporate action is applied to an
        instrument (``event.kind == "corporate_action"``).

        ``event.outcome`` is ``"split"``, ``"bonus"`` or ``"dividend"``;
        ``event.quantity`` the ratio (new units per old; 0 for a dividend),
        ``event.amount`` the dividend per unit and ``event.cash`` the cash the
        account received (negative for a short that paid a dividend; for a
        split, the cash in lieu of a fractional unit). A split the engine
        could not apply arrives as ``"lifecycle_refused"``."""

    def on_position_opened(self, ctx: StrategyContext, event) -> None:
        """Deprecated: use :meth:`on_position_event`.

        Called after a position opens."""

    def on_position_closed(self, ctx: StrategyContext, event) -> None:
        """Deprecated: use :meth:`on_position_event`.

        Called after a position closes; ``event.trade`` is the round trip."""

    # -- order API -----------------------------------------------------------

    def enter(
        self,
        size_frac: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
        side: str | None = None,
    ) -> None:
        """Queue a market entry for this bar.

        Without ``side``, opens in the session's configured direction (the
        run's ``direction``, or the leg's entry in ``directions``).

        With ``side="buy"``/``"sell"``, the *order's* side decides: a leg
        registered long can open a short and flip back on a later bar, once
        flat. This is what a cross-sectional long/short book needs, where a
        name's side follows its rank rather than being fixed for the run.
        Prefer :meth:`enter_long` / :meth:`enter_short`, which read better
        and cannot be mis-spelled.
        """
        check_term("size_frac", size_frac, maximum=1.0)
        check_term("stop_price", stop_price)
        check_term("target_price", target_price)
        if side is None:
            self._queue.enter(size_frac, stop_price, target_price)
            return
        if side not in ("buy", "sell"):
            raise InvalidOrder(
                f"side must be 'buy' or 'sell', got {side!r}",
                code="invalid_order_side",
                operation="enter",
                parameter="side",
                value=side,
            )
        # A sided entry rides the typed-order path, where the side survives
        # to the kernel. `size_frac` must be explicit: omitting both sizing
        # kwargs means "close the whole position", which an opening order
        # refuses.
        self.submit_order(
            Market(
                side=side,
                size_frac=1.0 if size_frac is None else size_frac,
                stop_price=stop_price,
                target_price=target_price,
            )
        )

    # ``buy`` reads naturally in long-only strategies; it is the same intent.
    buy = enter

    def enter_long(
        self,
        size_frac: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
    ) -> None:
        """Queue a LONG market entry for this bar, whatever the leg's
        configured direction. See :meth:`enter`."""
        self.enter(
            size_frac=size_frac,
            stop_price=stop_price,
            target_price=target_price,
            side="buy",
        )

    def enter_short(
        self,
        size_frac: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
    ) -> None:
        """Queue a SHORT market entry for this bar, whatever the leg's
        configured direction. See :meth:`enter`."""
        self.enter(
            size_frac=size_frac,
            stop_price=stop_price,
            target_price=target_price,
            side="sell",
        )

    def close_position(
        self, position_id: int | None = None, symbol: str | None = None
    ) -> None:
        """Queue a close of the open position for this bar.

        With ``position_id`` (from ``ctx.positions``), closes that specific
        position — required under the hedging policy, where several are open
        at once. Without it, the legacy whole-position close intent.
        ``symbol`` routes in multi-instrument runs (defaults to the current
        bar's symbol).
        """
        if position_id is None and symbol is None:
            self._queue.close_intent()
        elif position_id is None:
            self._queue.close_all_for(symbol)
        else:
            self._queue.close(position_id, symbol)

    def submit_order(
        self,
        order: _OrderBase,
        client_id: str | None = None,
        parent: str | None = None,
        symbol: str | None = None,
    ) -> str:
        """Queue a typed order (:mod:`raptorbt.strategy.orders`).

        Returns the client order id — auto-generated as
        ``"{order_id_tag}-{seq}"`` when not supplied — which identifies the
        order on every subsequent event and in :meth:`cancel_order` /
        :meth:`modify_order`. ``parent`` (another order's client id) holds
        this order until the parent fills; it dies if the parent does.
        The order's ``instrument`` routes it in multi-instrument runs;
        single-instrument runs ignore it, and the portfolio runner defaults
        it to the symbol whose bar is being processed. ``symbol`` is the
        deprecated spelling of the same thing (it does not warn).
        """
        if not isinstance(order, _OrderBase):
            raise TypeError(
                "submit_order takes a typed order (orders.Market/Limit/StopMarket/"
                f"StopLimit/...), got {type(order).__name__}"
            )
        if order.instrument is not None:
            if symbol is not None and symbol != order.instrument:
                raise InvalidOrder(
                    f"submit_order: symbol={symbol!r} contradicts the order's "
                    f"instrument={order.instrument!r}",
                    code="conflicting_order_args",
                    operation="submit_order",
                    parameter="symbol, instrument",
                    value=[symbol, order.instrument],
                )
            symbol = order.instrument
        if client_id is None:
            tag = getattr(self.config, "order_id_tag", None) or "O"
            client_id = f"{tag}-{self._order_seq}"
        self._order_seq += 1
        self._queue.submit(order, client_id, parent, symbol, isinstance(order, Twap))
        return client_id

    def register_indicator(
        self, indicator, stream_id: int | None = None, symbol: str | None = None
    ):
        """Auto-update a streaming indicator (``raptorbt.Indicator``) from
        bar data, *before* handlers see the bar.

        When several instruments print at one timestamp, every instrument's
        indicators are current as of that timestamp before any handler of it
        runs, so a handler reads another symbol's indicator at the value after
        that symbol's bar, whatever the symbols are called. The event's own
        primary-stream indicator shows the value before the bar until
        ``on_bar``; one on a composite stream shows the value after the bar the
        handler is on (``docs/architecture/event-semantics.md``).

        ``stream_id=None`` feeds it primary-stream bars; a handle from
        :meth:`subscribe_bars` feeds it that composite stream instead.
        Returns the indicator for chained assignment. Call from
        ``on_start``.

        ``symbol`` routes the indicator in portfolio runs and is ignored in
        single-instrument ones. **Pass it.** Without it the indicator is fed
        every symbol's bars interleaved, which is almost never meaningful —
        one indicator cannot track N series. The usual shape is one
        indicator per symbol::

            self.fast = {
                s: self.register_indicator(Indicator.sma(10), symbol=s)
                for s in ctx.symbols
            }

        or, equivalently, :meth:`register_indicators`.
        """
        self._indicators.append((indicator, stream_id, symbol))
        return indicator

    def register_indicators(
        self, factory, symbols, stream_id: int | None = None
    ) -> dict:
        """Register one indicator per symbol, built by ``factory()``.

        Returns ``symbol -> indicator``::

            self.fast = self.register_indicators(
                lambda: Indicator.sma(10), ctx.symbols
            )
        """
        return {
            symbol: self.register_indicator(
                factory(), stream_id=stream_id, symbol=symbol
            )
            for symbol in symbols
        }

    def indicators_initialized(self) -> bool:
        """Whether every registered indicator has completed warmup.

        In a portfolio run that means every symbol's indicator is warm.
        """
        return all(ind.initialized for ind, _, _ in self._indicators)

    def subscribe_bars(self, step: int, unit: str, *, brick_size: float = 0.0) -> int:
        """Subscribe to bars aggregated from the primary stream.

        Call from ``on_start``. Completed bars arrive via
        :meth:`on_composite_bar`. Returns the stream handle carried on each
        bar's ``stream_id``. Units: time (``"ms"``/``"s"``/``"m"``/``"h"``/
        ``"d"``/``"w"``), ``"tick"``, ``"volume"``, ``"value"``.

        In a portfolio run one subscription yields one aggregated stream
        *per symbol*, each built only from that symbol's bars. The symbol
        that completed a bar arrives as ``bar.symbol`` (and ``ctx.symbol``).
        """
        if step < 1:
            raise InvalidSubscription(
                "step must be >= 1",
                code="invalid_bar_step",
                operation="subscribe_bars",
                parameter="step",
                value=step,
            )
        self._bar_subscriptions.append((step, unit, brick_size))
        return len(self._bar_subscriptions) - 1

    def subscribe(
        self,
        instrument: str,
        *,
        data_type: str = "bar",
        timeframe=None,
        history: int | None = None,
    ):
        """Declare interest in an instrument's data. Call from ``on_init`` or
        ``on_start``.

        ``data_type`` is ``"bar"``, ``"trade"``, ``"quote"`` or ``"book"``.
        ``timeframe`` (bars only) is ``None`` for the primary bars, one of
        ``"1m" "3m" "5m" "10m" "15m" "30m" "1h" "2h" "4h" "1d" "1w"``, or a
        :class:`~raptorbt.strategy.Timeframe` / ``(step, unit)`` tuple for tick,
        volume, value and renko bars. ``history`` is the lookback the
        strategy will read: it sizes the rows a live node keeps and the
        buffer of a composite timeframe.

        Once a strategy calls ``subscribe`` it receives trades, quotes and
        books only for what it subscribed to; a strategy that never calls it
        receives everything, as before. Returns the stream handle (the
        ``stream_id`` of its composite bars) for a ``timeframe``, else
        ``None``.
        """
        return api.subscribe(self, instrument, data_type, timeframe, history)

    def link_oco(self, *client_ids: str) -> None:
        """Queue a one-cancels-other link between submitted orders."""
        if len(client_ids) < 2:
            raise InvalidOrder(
                "link_oco needs at least two client ids",
                code="invalid_order_link",
                operation="link_oco",
                parameter="client_ids",
                value=list(client_ids),
            )
        self._queue.link_oco(list(client_ids))

    def submit_bracket(
        self,
        entry: _OrderBase,
        stop_trigger: float,
        target_price: float,
        stop_limit_price: float | None = None,
    ) -> tuple[str, str, str]:
        """Queue an entry with protective legs: one-triggers-other children
        (stop + target activate when the entry fills) linked
        one-cancels-other with each other.

        The legs close the full position on the opposite side. They are
        ``reduce_only``, so a leg still working after the position closed by
        another route can never open a fresh position on the opposite side.
        Netting policy only — under hedging every order opens its own
        position, so the legs could never close the bracket's: submitting one
        raises ``bracket_under_hedging`` rather than leaving the position
        open. Use per-position ``stop_price``/``target_price`` attachments
        instead.

        Returns ``(entry_id, stop_id, target_id)``.
        """
        from raptorbt.strategy.orders.types import Limit, StopLimit, StopMarket

        exit_side = "sell" if entry.side == "buy" else "buy"
        entry_id = self.submit_order(entry)
        stop_order = (
            StopMarket(
                side=exit_side,
                trigger=stop_trigger,
                reduce_only=True,
                instrument=entry.instrument,
            )
            if stop_limit_price is None
            else StopLimit(
                side=exit_side,
                trigger=stop_trigger,
                price=stop_limit_price,
                reduce_only=True,
                instrument=entry.instrument,
            )
        )
        stop_id = self.submit_order(stop_order, parent=entry_id)
        target_id = self.submit_order(
            Limit(
                side=exit_side,
                price=target_price,
                reduce_only=True,
                instrument=entry.instrument,
            ),
            parent=entry_id,
        )
        self.link_oco(stop_id, target_id)
        return entry_id, stop_id, target_id

    # -- option structures --------------------------------------------------

    def _structure_book(self):
        # One book per run: the run replaces ``self.cache``, which drops the old book.
        from raptorbt.structures.book import StructureBook

        book = self.__dict__.get("_structures")
        if book is None or book.cache is not self.cache:
            book = self.__dict__["_structures"] = StructureBook(self)
        return book

    def _exit_book(self):
        # One book per run, like the structure book.
        from raptorbt.strategy.exits import ExitBook

        book = self.__dict__.get("_exits")
        if book is None or book.cache is not self.cache:
            book = self.__dict__["_exits"] = ExitBook(self)
        return book

    def set_exit(self, ctx, rule, *, symbol=None, symbols=None, group=None, name=None):
        """Set an :class:`~raptorbt.strategy.exits.ExitRule` on one symbol's positions
        (``symbol=``), a named group of symbols (``symbols=`` and ``name=``) or a structure
        group (``group=``). Returns an :class:`~raptorbt.strategy.exits.ExitHandle`. The engine
        closes the positions with reduce-only orders when the rule fires."""
        return self._exit_book().set(ctx, rule, symbol=symbol, symbols=symbols, group=group, name=name)

    def update_exits(self, ctx, marks=None):
        """Evaluate the exit rules (``marks`` maps symbol to price; default the last prices) and
        send the closes of those that fire. Mix in :class:`~raptorbt.strategy.exits.ExitEvents`
        to have this run on every price."""
        return self._exit_book().update(ctx, marks)

    def cancel_exit(self, handle) -> None:
        """Disarm an exit rule that has not fired."""
        self._exit_book().cancel(handle)

    @staticmethod
    def _event_ns(ctx, now_ns):
        if now_ns is not None:
            return now_ns
        ts = getattr(ctx, "bar_close_ts", None)
        return ts if ts is not None else ctx.timestamp

    def submit_structure(self, ctx, structure, *, now_ns: int | None = None, **kwargs):
        """Resolve a :class:`~raptorbt.structures.Structure` against an option chain at the event
        time and place its legs as one group of ordinary orders, hedge-first: the bought legs go
        out at once and each sold leg once the hedges covering it have filled (for the quantity
        they cover). Keyword arguments: ``chain``, ``underlying``, ``spot``, ``lots``,
        ``underlying_spec``, ``order_type`` (``"market"``/``"limit"``), ``prices``, ``net_limit``,
        ``exit``, ``on_leg_failure``, ``margin_model``, ``available``, ``tif`` and ``group_id``.
        The same sequencing runs in a backtest, on ``PaperExecution`` and on a broker adapter.
        Returns the :class:`~raptorbt.structures.StructureGroup`. See ``docs/structures.md``."""
        return self._structure_book().submit(
            structure,
            now_ns=self._event_ns(ctx, now_ns),
            **kwargs,
        )

    def update_structures(self, ctx, marks=None, *, exits: bool = True, now_ns: int | None = None):
        """Advance every structure group: leg-failure policy, exits (``marks`` maps leg symbol to
        its last price) and pending cancels and closes. Call it from ``on_bar``."""
        return self._structure_book().update(self._event_ns(ctx, now_ns), marks, exits=exits)

    def close_structure(self, group, reason: str = "manual") -> None:
        """Close a whole group: cancel its working legs, close its filled ones."""
        self._structure_book().close(group, reason)

    def structure_status(self, group) -> str:
        """The group's status (see :meth:`StructureGroup.status`)."""
        return group.status(self.cache)

    def cancel_order(self, client_id: str) -> None:
        """Queue cancellation of a working order, or of a ``Twap`` schedule's
        unreleased slices, by client id. An order that finished first (filled,
        canceled, expired) is left alone, as at a venue; a client id this run
        never submitted raises ``unknown_order``."""
        self._queue.cancel(client_id)

    def cancel_all_orders(self) -> None:
        """Queue cancellation of every working order."""
        self._queue.cancel_all()

    def modify_order(
        self,
        client_id: str,
        units: float | None = None,
        size_frac: float | None = None,
        limit_price: float | None = None,
        trigger_price: float | None = None,
    ) -> None:
        """Queue a price/quantity replacement of a working order. A client id this run
        never submitted raises ``unknown_order``; one that has finished is left alone."""
        self._queue.modify(client_id, units, size_frac, limit_price, trigger_price)

    def drain_orders(self) -> list[tuple]:
        """Return and clear pending intents, as ``("enter", size_frac, stop,
        target)`` / ``("close",)`` tuples. Called by the runner."""
        return self._queue.drain_intents()

    def drain_commands(self) -> list[tuple]:
        """Return and clear pending typed-order commands. Called by the runner."""
        return self._queue.drain_commands()

    # -- logging -------------------------------------------------------------

    @property
    def log(self) -> logging.Logger:
        """Logger namespaced to the concrete strategy class."""
        return logging.getLogger(f"raptorbt.strategy.{type(self).__name__}")


api._install_base(Strategy)
