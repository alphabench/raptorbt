"""Gate fixes of structure submission, each from a failing run.

* a group the margin model cannot price is refused with that reason (``margin_unpriced``), not
  as a funding failure, and is not counted as an unfunded entry;
* ``submit_structure`` defaults a leg's ``tif`` to ``"auto"`` (the instrument-driven default), and
  the docs say so.
"""

import inspect
import pathlib

from raptorbt import BacktestConfig, InstrumentSpec, MarginModel, OptionChain, Strategy, run_strategies
from raptorbt import structures as st
from raptorbt.structures.book import StructureBook

from _chain_fixture import U, specs, universe

CHAIN = OptionChain(specs())
DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "structures.md"


def test_a_covered_call_whose_underlying_has_no_spec_is_refused_as_unpriced_not_unfunded():
    class Covered(Strategy):
        group = None

        def on_bar(self, ctx):
            self.update_structures(ctx, exits=False)
            if ctx.symbol == U and self.group is None and ctx.idx == 3:
                self.group = self.submit_structure(
                    ctx, st.covered_call(1), chain=CHAIN, underlying=U, spot=ctx.bar.close,
                    underlying_spec=InstrumentSpec.equity(U),
                )

    s = Covered()
    data, inst = universe()  # the underlying is not among the registered instruments
    cfg = BacktestConfig(initial_capital=1_000_000.0, fees=0.0, margin_model=MarginModel.defined_risk())
    res = run_strategies([s], data, cfg, instruments=inst, account_type="margin", leverage=1.0,
                         bar_stamp="close")[0]
    reasons = {o.symbol: o.reject_reason for o in res.result.orders()}
    assert set(reasons.values()) == {"margin_unpriced"}, reasons
    assert res.result.unfunded_entries == 0
    assert res.result.trades() == []


def test_a_structure_leg_defaults_to_the_instrument_driven_tif_and_the_docs_say_so():
    assert inspect.signature(StructureBook.submit).parameters["tif"].default == "auto"
    text = DOCS.read_text()
    assert 'tif="auto"' in text and 'tif="gtc"' not in text
