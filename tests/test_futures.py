import math

import pandas as pd
import pytest

from finm37000 import build_short_rate_curve


def test_build_short_rate_curve_bootstraps_expected_zero_rates() -> None:
    start = pd.Timestamp("2025-01-01")
    expired_leg = start - pd.Timedelta(days=1)
    near_leg = start + pd.Timedelta(days=90)
    far_leg = start + pd.Timedelta(days=180)

    # Rows deliberately unsorted and include an already-expired leg, to check
    # both get handled (sorted, filtered out) rather than assumed by the caller.
    stats = pd.DataFrame(
        {
            "expiration": [far_leg, expired_leg, near_leg],
            "Settlement price": [94.0, 99.0, 95.0],
        }
    )

    rate_curve = build_short_rate_curve(stats, start, days_per_year=360.0)

    t_near, t_far = 0.25, 0.5
    forward_near, forward_far = 0.05, 0.06

    discount_near = 1 / (1 + forward_near * t_near)
    expected_r_near = -math.log(discount_near) / t_near
    discount_far = discount_near / (1 + forward_far * (t_far - t_near))
    expected_r_far = -math.log(discount_far) / t_far

    assert rate_curve(t_near) == pytest.approx(expected_r_near)
    assert rate_curve(t_far) == pytest.approx(expected_r_far)

    # Flat extrapolation beyond the last listed leg.
    assert rate_curve(t_far + 1.0) == pytest.approx(expected_r_far)

    # Linear-in-rate interpolation between two curve points stays bracketed.
    midpoint_rate = rate_curve((t_near + t_far) / 2)
    lo, hi = sorted((expected_r_near, expected_r_far))
    assert lo <= midpoint_rate <= hi
