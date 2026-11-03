"""Fees as data.

A :class:`FeeSchedule` is a set of rows, one per venue, segment and (where it
differs) underlying, each a handful of declared rates: brokerage (flat,
percent, cap, minimum, per order or per fill), the transaction tax by side,
the exchange transaction charge, the regulatory fee, stamp duty on the buy
side, GST on a base the row declares, and a flat fee owed once per instrument
per day. Rates are fractions of turnover in the Python API (``0.0003`` is
0.03%) and percent text in a rate card document.

``FeeSchedule.exchange_presets()`` is the statutory schedule: what the tax
authority, the exchanges and the regulator levy on NSE, BSE, NFO, BFO, MCX, CDS
and BCD, each row from the date it took effect (STT on options rose on
2026-04-01, and the tax on an option exercised at expiry is a settlement row).
It carries no brokerage and no depository fee: those are a broker's, from a
rate card. Load a card with ``RateCard.from_json`` / ``RateCard.from_dict`` and
run on it with ``BacktestConfig(fee_schedule=...)``, which prices each
instrument on the row its venue and kind select; ``fee_segment="NFO-OPT"`` is
the explicit override that prices every instrument on one row.

>>> from raptorbt.costs import FeeSchedule
>>> s = FeeSchedule.exchange_presets()
>>> b = s.charge("NFO", "OPTIONS", "SELL", price=100.0, quantity=75)
>>> round(b["total"], 2), b["brokerage"]
(14.4, 0.0)
"""

from raptorbt._raptorbt import FeeSchedule, RateCard

__all__ = ["FeeSchedule", "RateCard"]
