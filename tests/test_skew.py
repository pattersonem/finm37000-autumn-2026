import numpy as np
import pandas as pd
import pytest

from finm37000 import (
    imply_american_rate,
    select_rate_inputs,
)
from finm37000.options import calc_american_price


def _make_synthetic_chain(  # noqa: PLR0913
    strikes: np.ndarray,
    underlying_symbol: str,
    futures_price: float,
    years_to_expiration: float,
    rate: float,
    vols: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an option_chain/top_df pair shaped like the Databento fetch output."""
    call_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=vols,
        r=rate,
        option_type="C",
    )
    put_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=vols,
        r=rate,
        option_type="P",
    )

    rows = []
    midprices = {underlying_symbol: futures_price}
    for k, call_price, put_price in zip(strikes, call_prices, put_prices, strict=True):
        call_symbol = f"C{k:.0f}"
        put_symbol = f"P{k:.0f}"
        rows.append(
            {
                "symbol": call_symbol,
                "strike_price": k,
                "instrument_class": "C",
                "years_to_expiration": years_to_expiration,
            }
        )
        rows.append(
            {
                "symbol": put_symbol,
                "strike_price": k,
                "instrument_class": "P",
                "years_to_expiration": years_to_expiration,
            }
        )
        midprices[call_symbol] = call_price
        midprices[put_symbol] = put_price

    option_chain = pd.DataFrame(rows)
    top_df = pd.DataFrame({"midprice": midprices})
    top_df.index.name = "symbol"
    return option_chain, top_df


def test_select_rate_inputs_feeds_imply_american_rate() -> None:
    strikes = np.array([55.0, 60.0, 65.0, 70.0, 75.0])
    underlying_symbol = "ZCZ5"
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045
    true_vols = np.array([0.28, 0.25, 0.24, 0.27, 0.28])

    option_chain, top_df = _make_synthetic_chain(
        strikes,
        underlying_symbol,
        futures_price,
        years_to_expiration,
        true_rate,
        true_vols,
    )

    inputs = select_rate_inputs(top_df, underlying_symbol, option_chain)
    fitted_rate, _ = imply_american_rate(
        strikes=inputs.strikes,
        call_prices=inputs.call_prices,
        put_prices=inputs.put_prices,
        futures_price=inputs.futures_price,
        years_to_expiration=inputs.years_to_expiration,
    )

    assert fitted_rate == pytest.approx(true_rate, abs=1e-4)


def test_select_rate_inputs_keeps_only_n_strikes_nearest_the_money() -> None:
    strikes = np.array([40.0, 55.0, 60.0, 61.0, 63.0, 65.0, 80.0])
    underlying_symbol = "ZCZ5"
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045
    true_vols = np.full(strikes.shape, 0.25)

    option_chain, top_df = _make_synthetic_chain(
        strikes,
        underlying_symbol,
        futures_price,
        years_to_expiration,
        true_rate,
        true_vols,
    )

    inputs = select_rate_inputs(top_df, underlying_symbol, option_chain, n_strikes=3)
    assert sorted(inputs.strikes) == [60.0, 61.0, 63.0]


def test_select_rate_inputs_allows_a_single_paired_strike() -> None:
    underlying_symbol = "ZCZ5"
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045

    option_chain, top_df = _make_synthetic_chain(
        strikes=np.array([60.0]),
        underlying_symbol=underlying_symbol,
        futures_price=futures_price,
        years_to_expiration=years_to_expiration,
        rate=true_rate,
        vols=np.array([0.25]),
    )

    inputs = select_rate_inputs(top_df, underlying_symbol, option_chain, n_strikes=1)

    assert list(inputs.strikes) == [60.0]


def test_select_rate_inputs_rejects_n_strikes_below_one() -> None:
    underlying_symbol = "ZCZ5"
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045

    option_chain, top_df = _make_synthetic_chain(
        strikes=np.array([60.0]),
        underlying_symbol=underlying_symbol,
        futures_price=futures_price,
        years_to_expiration=years_to_expiration,
        rate=true_rate,
        vols=np.array([0.25]),
    )

    with pytest.raises(ValueError, match="n_strikes must be at least 1"):
        select_rate_inputs(top_df, underlying_symbol, option_chain, n_strikes=0)


def test_select_rate_inputs_raises_without_enough_paired_strikes() -> None:
    underlying_symbol = "ZCZ5"
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045

    option_chain, top_df = _make_synthetic_chain(
        strikes=np.array([60.0, 65.0]),
        underlying_symbol=underlying_symbol,
        futures_price=futures_price,
        years_to_expiration=years_to_expiration,
        rate=true_rate,
        vols=np.array([0.25, 0.26]),
    )
    call_chain = option_chain[option_chain["instrument_class"] == "C"]
    with pytest.raises(ValueError, match="at least 2 strikes"):
        select_rate_inputs(top_df, underlying_symbol, call_chain, n_strikes=2)
    with pytest.raises(ValueError, match="at least 3 strikes"):
        select_rate_inputs(top_df, underlying_symbol, option_chain, n_strikes=3)
