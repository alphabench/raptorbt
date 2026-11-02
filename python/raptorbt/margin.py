"""Margin models.

``MarginModel.no_margin()``, ``cash_only()``, ``leverage({"CASH": 5})``,
``percent_of_notional(pct)``, ``option_simple(seller_pct)``,
``table(per_unit, default)``, ``by_type({...})``, ``by_product(intraday,
carry)``, ``defined_risk()``, ``spread_credit(seller_pct, spread_pct)`` and
``from_spec(leverage)``, the rule the engine itself applies on a margin
account.

``required(spec, quantity, price)`` is the initial margin of one line;
``required_legs([(spec, quantity, price), ...])`` margins option legs
together; ``describe()`` is what a result says the margin was. Instrument
classes are ``CASH`` (``EQUITY`` and ``ETF`` read as ``CASH``),
``CURRENCY_PAIR``, ``CONTRACT`` (``FUTURE`` and ``PERPETUAL`` read as
``CONTRACT``) and ``OPTION``.
"""

from raptorbt._raptorbt import MarginModel, defined_risk

__all__ = ["MarginModel", "defined_risk"]
