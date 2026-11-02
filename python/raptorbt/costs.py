"""Fees as data.

A :class:`FeeSchedule` is a set of rows, one per venue, segment and (where it
differs) underlying, each a handful of declared rates: brokerage (flat,
percent, cap, minimum, per order or per fill), the transaction tax by side,
the exchange transaction charge, the regulatory fee, stamp duty on the buy
side, GST on a base the row declares, and a flat fee owed once per instrument
per day. Rates are fractions of turnover in the Python API (``0.0003`` is
0.03%) and percent text in a rate card document.

``FeeSchedule.exchange_presets()`` is what the engine charges when you give it
no schedule: NSE, BSE, NFO, BFO, MCX, CDS and BCD, named by venue. Load your
own with ``RateCard.from_json`` / ``RateCard.from_dict`` and run on it with
``BacktestConfig(fee_schedule=..., fee_segment="NFO-OPT")``.

>>> from raptorbt.costs import FeeSchedule
>>> s = FeeSchedule.exchange_presets()
>>> b = s.charge("NFO", "OPTIONS", "SELL", price=100.0, quantity=75)
>>> round(b["total"], 2)
38.0
"""

from raptorbt._raptorbt import FeeSchedule, RateCard

__all__ = ["FeeSchedule", "RateCard"]
