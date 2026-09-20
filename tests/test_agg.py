import pandas as pd

from finm37000 import make_ohlcv


def test_make_ohlcv_labels_data_on_right_by_default() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "price": [1.0, 2.0, 3.0],
            "size": [10, 10, 10],
        },
        index=pd.to_datetime(
            ["2024-01-01 00:00:01", "2024-01-01 00:00:02", "2024-01-01 00:00:03"]
        ).rename("ts_recv"),
    )

    actual = make_ohlcv(trades, rule="2s")

    expected = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "open": [1.0, 2.0],
            "high": [1.0, 3.0],
            "low": [1.0, 2.0],
            "close": [1.0, 3.0],
            "volume": [10, 20],
        },
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00:02", "2024-01-01 00:00:04"],
            name="ts_event",
        ),
    )

    pd.testing.assert_frame_equal(actual, expected)
