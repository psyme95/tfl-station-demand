import numpy as np
import pandas as pd
import pytest

import tfl

WEEKLY = np.array([1.0, 1.1, 1.15, 1.15, 1.05, 0.7, 0.5])     # Monday to Sunday


def synthetic_counts(station="A", start="2024-01-01", end="2025-09-13", level=1000, seed=0):
    """Daily counts with a fixed weekly pattern and small Poisson noise."""
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    mean = level * WEEKLY[dates.dayofweek]
    return pd.DataFrame({"TravelDate": dates, "Station": station,
                         "EntryTapCount": rng.poisson(mean)})


def test_build_series_keeps_missing_dates():
    counts = synthetic_counts()
    counts = counts[counts["TravelDate"] != "2024-03-06"]
    df = tfl.build_series(counts, "A")
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert np.isnan(df.loc[df["Date"] == "2024-03-06", "Entries"]).all()


def test_build_series_masks_strike_and_disruption_but_not_holidays():
    counts = synthetic_counts()
    closed = counts["TravelDate"] == "2024-06-12"
    counts.loc[closed, "EntryTapCount"] = 10
    xmas = counts["TravelDate"] == "2024-12-25"
    counts.loc[xmas, "EntryTapCount"] = 50
    df = tfl.build_series(counts, "A").set_index("Date")

    assert df.loc["2025-09-08", "IsStrike"] and np.isnan(df.loc["2025-09-08", "y"])
    assert df.loc["2024-06-12", "IsDisrupted"] and np.isnan(df.loc["2024-06-12", "y"])
    assert df.loc["2024-12-25", "IsHoliday"] and df.loc["2024-12-25", "y"] == 50


def test_build_series_masks_zero_counts_on_holidays():
    counts = synthetic_counts()
    counts.loc[counts["TravelDate"] == "2024-05-27", "EntryTapCount"] = 0
    df = tfl.build_series(counts, "A").set_index("Date")
    assert df.loc["2024-05-27", "IsHoliday"] and np.isnan(df.loc["2024-05-27", "y"])


def test_build_series_unknown_station():
    with pytest.raises(KeyError):
        tfl.build_series(synthetic_counts(), "B")


@pytest.mark.parametrize("k", [1, 2, 4])
def test_calendar_features_shape(k):
    dates = pd.date_range("2025-01-06", periods=14, freq="D")      # starts on a Monday
    X = tfl.calendar_features(dates, np.zeros(14), np.zeros(14), k)
    assert X.shape == (14, 6 + 2 * k + 2)
    assert X.filter(like="dow_").iloc[0].sum() == 0                 # Monday is the reference
    assert X.filter(like="dow_").sum(axis=1).iloc[1:7].eq(1).all()


def test_seasonal_naive_returns_weekday_means():
    dates = pd.date_range("2025-01-06", periods=56, freq="D")
    train = pd.DataFrame({"Date": dates, "y": 100 * WEEKLY[dates.dayofweek]})
    target = pd.date_range("2025-03-03", periods=7, freq="D")
    np.testing.assert_allclose(tfl.seasonal_naive(train, target), 100 * WEEKLY)


def test_poisson_recovers_weekly_pattern():
    df = tfl.build_series(synthetic_counts(), "A")
    origin = pd.Timestamp("2025-06-30")
    target = pd.date_range("2025-07-07", periods=7, freq="D")
    pred = tfl.forecast(df, origin, target, models=["poisson"])
    np.testing.assert_allclose(pred["poisson"], 1000 * WEEKLY, rtol=0.05)


def test_backtest_origins_respect_gap_and_training_length():
    df = tfl.build_series(synthetic_counts(), "A")
    bt = tfl.rolling_backtest(df, gap_days=58, n_origins=10, min_train_days=240,
                              models=["poisson"])
    origins = pd.to_datetime(bt["origin"].unique())
    assert ((origins - df["Date"].min()).days >= 240).all()
    assert origins.max() + pd.Timedelta(days=58 + 30) <= df["Date"].max() + pd.Timedelta(days=1)
    assert set(bt["model"]) == {"naive", "poisson"}


def test_name_clusters_ignores_label_order():
    centres = pd.DataFrame({"LogMeanEntryCount": [0.7, -0.1, -0.6],
                            "WeekdayWeekendBalance": [-0.6, 1.5, 0.2]})
    names = tfl.name_clusters(centres)
    assert names == {1: "Weekday commuter stations", 2: "Low volume stations",
                     0: "Consistent high-volume stations"}
    shuffled = centres.iloc[[2, 0, 1]].reset_index(drop=True)
    assert tfl.name_clusters(shuffled)[2] == "Weekday commuter stations"
