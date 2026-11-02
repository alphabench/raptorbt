"""Names kept as deprecated aliases until 1.0."""

import pytest

import raptorbt
import raptorbt.strategy


@pytest.mark.parametrize("module", [raptorbt, raptorbt.strategy])
def test_portfolio_context_is_a_deprecated_alias_of_runtime_context(module):
    with pytest.warns(DeprecationWarning, match="PortfolioContext is deprecated"):
        cls = module.PortfolioContext
    assert cls is raptorbt._raptorbt.RuntimeContext


def test_unknown_names_still_raise_attribute_error():
    with pytest.raises(AttributeError):
        raptorbt.strategy.NoSuchThing
    with pytest.raises(AttributeError):
        raptorbt.NoSuchThing
