"""Build an OrderRequest for tests. The engine is the only thing that makes them, so run a node once."""

from raptorbt import BacktestConfig, LiveNode, Strategy
from raptorbt.strategy import orders

MIN = 60_000_000_000


class _Rec:
    namespace = "T1"
    keeps_own_records = True  # a test venue: it remembers the node's orders itself

    def __init__(self):
        self.requests = []

    def submit(self, r):
        self.requests.append(r)

    def cancel(self, c):
        pass

    def modify(self, c, **k):
        pass


_cache: dict = {}


def request(side="buy", quantity=10.0, price=99.0, order_type="limit", tif="gtc", post_only=False,
            reduce_only=False, symbol="A", trigger=95.0, strategy=0, capital=1e6):  # fmt: skip
    """The request the engine sends for one order. ``strategy=k`` gives the id ``s{k}-0``."""
    key = (side, quantity, price, order_type, tif, post_only, reduce_only, symbol, trigger, strategy, capital)
    if key in _cache:
        return _cache[key]
    rec = _Rec()
    extra = {"expire_ns": 10**18} if tif == "gtd" else {}
    common = dict(side=side, units=quantity, reduce_only=reduce_only)
    order = {
        "limit": lambda: orders.Limit(price=price, tif=tif, post_only=post_only, **common, **extra),
        "market": lambda: orders.Market(**common),
        "stop_market": lambda: orders.StopMarket(trigger=trigger, **common),
        "stop_limit": lambda: orders.StopLimit(trigger=trigger, price=price, **common),
        "trailing_stop_market": lambda: orders.TrailingStopMarket(offset=1.0, **common),
        "market_if_touched": lambda: orders.MarketIfTouched(trigger=trigger, **common),
    }[order_type]()

    class S(Strategy):
        def __init__(self, active):
            super().__init__()
            self.active, self.n = active, 0

        def on_bar(self, ctx, bar):
            self.n += 1
            if self.active and self.n == 1:
                self.submit_order(order, symbol=symbol)

    strategies = [S(i == strategy) for i in range(strategy + 1)]
    node = LiveNode(strategies, ["A", "B"], BacktestConfig(initial_capital=capital, fees=0.0), bar_stamp="close", execution=rec)
    for i in range(2):
        for sym in ("A", "B"):
            node.push_bar(sym, (i + 1) * MIN, 100.0, 101.0, 99.0, 100.5, 1000.0)
    _cache[key] = rec.requests[0]
    return rec.requests[0]
