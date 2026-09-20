import datetime

import databento as db
import pandas as pd
import pytest

from finm37000 import (
    get_databento_api_key,
    get_short_rate_curve,
    get_stats_by_date,
    temp_env,
    tz_chicago,
)


@pytest.fixture
def client() -> db.Historical:
    with temp_env(DATABENTO_API_KEY=get_databento_api_key()):
        client = db.Historical()
    return client


@pytest.mark.db
def test_get_stats_by_date_with_delayed_settle(client: db.Historical) -> None:
    date = datetime.date(2022, 2, 24)
    crude_defs = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="definition",
        symbols="CL.FUT",
        stype_in="parent",
        start=date,
    ).to_df()
    crude_at_war = get_stats_by_date(client, date, crude_defs)
    valid_prices = crude_at_war[crude_at_war["Settlement price"].notna()]
    assert not valid_prices.empty


@pytest.mark.db
def test_get_short_rate_curve_gives_plausible_near_term_rates(
    client: db.Historical,
) -> None:
    start = pd.Timestamp("2025-10-10T12:00:00", tz=tz_chicago)
    rate_curve = get_short_rate_curve(client, start)

    # A generous sanity band, not a precise expected value -- the curve moves
    # with the market. This is here to catch gross errors (wrong sign, wrong
    # units from the 100(1-r) conversion, empty curve), not to pin the rate.
    plausible_short_rate = (0.0, 0.10)
    for years_to_expiration in (0.02, 0.1, 0.25, 0.5, 0.75):
        rate = rate_curve(years_to_expiration)
        assert plausible_short_rate[0] < rate < plausible_short_rate[1]
