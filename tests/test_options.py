import numpy as np
import pytest

from finm37000 import imply_american_rate
from finm37000.options import (
    calc_american_greeks,
    calc_american_price,
    calc_numerical_delta,
    calc_numerical_rho,
    calc_numerical_theta,
    calc_numerical_vega,
)


def test_imply_american_rate_recovers_exact_rate_and_vols() -> None:
    strikes = np.array([55.0, 60.0, 65.0, 70.0])
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045
    true_vols = np.array([0.28, 0.25, 0.24, 0.27])

    call_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=true_vols,
        r=true_rate,
        option_type="C",
    )
    put_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=true_vols,
        r=true_rate,
        option_type="P",
    )

    fitted_rate, fitted_vols = imply_american_rate(
        strikes=strikes,
        call_prices=call_prices,
        put_prices=put_prices,
        futures_price=futures_price,
        years_to_expiration=years_to_expiration,
    )

    assert fitted_rate == pytest.approx(true_rate, abs=1e-4)
    np.testing.assert_allclose(fitted_vols, true_vols, atol=1e-4)


def test_imply_american_rate_raises_with_no_strikes() -> None:
    """Zero strikes means zero residuals for the optimizer to fit against.

    Without the guard, scipy would otherwise silently "converge" at whatever
    initial_rate/initial_vol it started from, since there's nothing to minimize.
    """
    with pytest.raises(ValueError, match="at least 1 strike"):
        imply_american_rate(
            strikes=np.array([]),
            call_prices=np.array([]),
            put_prices=np.array([]),
            futures_price=62.0,
            years_to_expiration=0.25,
        )


def test_imply_american_rate_manages_noisy_prices() -> None:
    strikes = np.array([55.0, 60.0, 65.0, 70.0])
    futures_price = 62.0
    years_to_expiration = 0.25
    true_rate = 0.045
    true_vols = np.array([0.28, 0.25, 0.24, 0.27])
    call_noise = np.array([0.05, 0.01, -0.02, 0.01])
    put_noise = np.array([-0.02, 0.01, -0.03, 0.05])

    call_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=true_vols,
        r=true_rate,
        option_type="C",
    )
    put_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=true_vols,
        r=true_rate,
        option_type="P",
    )

    fitted_rate, fitted_vols = imply_american_rate(
        strikes=strikes,
        call_prices=call_prices + call_noise,
        put_prices=put_prices + put_noise,
        futures_price=futures_price,
        years_to_expiration=years_to_expiration,
    )

    # Note loose tolerance is intended just to confirm the
    # function passes.
    assert fitted_rate == pytest.approx(true_rate, abs=1e-1)
    np.testing.assert_allclose(fitted_vols, true_vols, atol=1e-1)


def test_imply_american_rate_handles_single_strike() -> None:
    strikes = np.array([60.0])
    futures_price = 62.0
    years_to_expiration = 0.3
    true_rate = 0.06
    true_vols = np.array([0.3])

    call_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=true_vols,
        r=true_rate,
        option_type="C",
    )
    put_prices = calc_american_price(
        future=futures_price,
        strike=strikes,
        t=years_to_expiration,
        vol=true_vols,
        r=true_rate,
        option_type="P",
    )

    fitted_rate, fitted_vols = imply_american_rate(
        strikes=strikes,
        call_prices=call_prices,
        put_prices=put_prices,
        futures_price=futures_price,
        years_to_expiration=years_to_expiration,
        initial_rate=0.001,
        initial_vol=1.0,
    )

    assert fitted_rate == pytest.approx(true_rate, abs=1e-4)
    np.testing.assert_allclose(fitted_vols, true_vols, atol=1e-4)


def test_calc_american_greeks_matches_individual_numerical_functions() -> None:
    future, strike, t, vol, r, option_type = (
        100.0,
        np.array([90.0, 100.0, 110.0]),
        0.5,
        0.25,
        0.04,
        "C",
    )
    greeks = calc_american_greeks(
        future=future, strike=strike, t=t, vol=vol, r=r, option_type=option_type
    )

    np.testing.assert_allclose(
        greeks[:, 0],
        calc_american_price(
            future=future, strike=strike, t=t, vol=vol, r=r, option_type=option_type
        ),
    )
    np.testing.assert_allclose(
        greeks[:, 1],
        calc_numerical_delta(
            future=future,
            strike=strike,
            r=r,
            t=t,
            vol=vol,
            option_type=option_type,
            dfuture=0.01,
        ),
    )
    np.testing.assert_allclose(
        greeks[:, 2],
        calc_numerical_vega(
            future=future,
            strike=strike,
            r=r,
            t=t,
            vol=vol,
            option_type=option_type,
            dvol=0.01,
        ),
    )
    np.testing.assert_allclose(
        greeks[:, 3],
        calc_numerical_theta(
            future=future,
            strike=strike,
            r=r,
            t=t,
            vol=vol,
            option_type=option_type,
            dt=1 / 365,
        ),
    )
    np.testing.assert_allclose(
        greeks[:, 4],
        calc_numerical_rho(
            future=future,
            strike=strike,
            r=r,
            t=t,
            vol=vol,
            option_type=option_type,
            dr=0.0001,
        ),
    )


def test_calc_american_greeks_shape_for_scalar_input() -> None:
    greeks = calc_american_greeks(
        future=100.0, strike=95.0, t=0.5, vol=0.25, r=0.04, option_type="C"
    )
    assert greeks.shape == (1, 5)


@pytest.mark.parametrize(
    ("option_type", "delta_bounds"), [("C", (0.0, 1.0)), ("P", (-1.0, 0.0))]
)
def test_calc_american_greeks_sign_conventions(
    option_type: str, delta_bounds: tuple[float, float]
) -> None:
    """Futures options: rho is negative for both calls and puts."""
    greeks = calc_american_greeks(
        future=100.0, strike=100.0, t=0.5, vol=0.25, r=0.04, option_type=option_type
    )
    _npv, delta, vega, _theta, rho = greeks[0]
    assert delta_bounds[0] < delta < delta_bounds[1]
    assert vega > 0
    assert rho < 0
