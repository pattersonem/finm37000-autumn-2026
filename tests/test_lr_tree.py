import numpy as np
import pytest

from finm37000 import calc_american_price, imply_american_vol


def test_broadcasting_matches_elementwise_scalar_calls() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    batched = calc_american_price(
        future=100.0, strike=strikes, t=0.5, vol=0.25, r=0.04, option_type="C"
    )
    scalars = np.array(
        [
            calc_american_price(
                future=100.0, strike=k, t=0.5, vol=0.25, r=0.04, option_type="C"
            )
            for k in strikes
        ]
    )
    np.testing.assert_allclose(batched, scalars)


def test_invalid_option_type_raises() -> None:
    with pytest.raises(ValueError, match="Invalid option type"):
        calc_american_price(
            future=100.0, strike=100.0, t=0.5, vol=0.25, r=0.04, option_type="X"
        )


def test_nan_vol_propagates_without_breaking_other_entries() -> None:
    vols = np.array([0.2, np.nan, 0.3])
    prices = calc_american_price(
        future=100.0,
        strike=100.0,
        t=0.5,
        vol=vols,
        r=0.04,
        option_type="C",
    )
    assert np.isnan(prices[1])
    assert np.isfinite(prices[0])
    assert np.isfinite(prices[2])


def test_even_n_steps_bumped_to_next_odd() -> None:
    kwargs: dict[str, float | str] = {
        "future": 100.0,
        "strike": 100.0,
        "t": 0.5,
        "vol": 0.25,
        "r": 0.04,
        "option_type": "C",
    }
    even = calc_american_price(**kwargs, n_steps=200)
    odd = calc_american_price(**kwargs, n_steps=201)
    assert even == pytest.approx(odd)


@pytest.mark.parametrize("option_type", ["C", "P"])
def test_price_is_at_least_intrinsic_value(option_type: str) -> None:
    future, strike = 100.0, 90.0
    price = calc_american_price(
        future=future, strike=strike, t=0.5, vol=0.25, r=0.04, option_type=option_type
    )
    cp = 1.0 if option_type == "C" else -1.0
    intrinsic = max(cp * (future - strike), 0.0)
    assert price >= intrinsic - 1e-8


@pytest.mark.parametrize("option_type", ["C", "P"])
def test_imply_american_vol_recovers_true_vol_scalar(option_type: str) -> None:
    true_vol = 0.27
    future, strike, t, r = 100.0, 95.0, 0.5, 0.04
    price = calc_american_price(
        future=future,
        strike=strike,
        t=t,
        vol=true_vol,
        r=r,
        option_type=option_type,
        n_steps=201,
    )
    implied = imply_american_vol(
        future=future,
        strike=strike,
        t=t,
        price=price,
        r=r,
        option_type=option_type,
        n_steps=201,
    )
    assert implied == pytest.approx(true_vol, abs=1e-4)


def test_imply_american_vol_recovers_true_vol_across_a_chain() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    true_vols = np.array([0.35, 0.30, 0.25, 0.28, 0.33])
    prices = calc_american_price(
        future=100.0, strike=strikes, t=0.5, vol=true_vols, r=0.04, option_type="C"
    )
    implied = imply_american_vol(
        future=100.0, strike=strikes, t=0.5, price=prices, r=0.04, option_type="C"
    )
    np.testing.assert_allclose(implied, true_vols, atol=1e-4)


def test_imply_american_vol_returns_nan_when_unbracketed() -> None:
    # An American price can never fall below intrinsic value, at any vol.
    implied = imply_american_vol(
        future=100.0, strike=90.0, t=0.5, price=5.0, r=0.04, option_type="C"
    )
    assert np.isnan(implied)
