"""Shared code for the clustering and forecasting notebooks."""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Three Elizabeth line stations appear under two names in the raw file.
RENAME = {
    "Woolwich Elizabeth Line":     "Woolwich EL",
    "Canary Wharf Elizabeth Line": "Canary Wharf EL",
    "Custom House Elizabeth Line": "Custom House EL",
}

# RMT strike, September 2025. Underground services were largely suspended from
# Monday 8 to Thursday 11 September and partly restored on Friday 12. The DLR
# struck separately on 9 and 11 September. The Overground and Elizabeth line ran
# but carried displaced demand, so all stations are affected in one direction or
# the other.
STRIKE = (pd.Timestamp("2025-09-05"), pd.Timestamp("2025-09-12"))

BANK_HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
    "2024-08-26", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26",
    "2025-08-25", "2025-12-25", "2025-12-26",
])

# Engineering works suppress demand at closed stations and inflate it at stations
# used as replacement hubs, so the disruption filter is two-sided.
DISRUPT_LO, DISRUPT_HI = 0.25, 1.75

FOURIER_K = 2
MODEL_NAMES = ["naive", "poisson", "lognormal", "xgb"]


CLUSTER_ORDER = ["Weekday commuter stations", "Low volume stations",
                 "Consistent high-volume stations"]


def name_clusters(centres):
    """Map k = 3 cluster labels to names using their centroids.

    k-means label numbers are arbitrary and can change between library versions,
    so names are assigned from the centroid profile: the cluster with the highest
    weekday-to-weekend balance is the commuter group, and of the other two the
    larger by volume is the high-volume group.
    """
    commuter = centres["WeekdayWeekendBalance"].idxmax()
    rest = centres.drop(commuter)
    high = rest["LogMeanEntryCount"].idxmax()
    low = rest.index.drop(high)[0]
    return {commuter: CLUSTER_ORDER[0], low: CLUSTER_ORDER[1], high: CLUSTER_ORDER[2]}


def load_counts(path):
    """Processed daily counts written by notebook 1."""
    return pd.read_csv(path, parse_dates=["TravelDate"])


def build_series(counts, station, extend_to=None):
    """One station as a gap-free daily frame with flags and a modelling column y.

    Missing dates stay as missing values so that the series is never silently
    shortened. Entries keeps the observed record; y is the masked series that is
    modelled.
    """
    end = extend_to if extend_to is not None else counts["TravelDate"].max()
    dates = pd.date_range(counts["TravelDate"].min(), end, freq="D")

    one = counts[counts["Station"] == station].set_index("TravelDate").sort_index()
    if one.empty:
        raise KeyError(f"no rows for station {station!r}")
    df = pd.DataFrame({"Date": dates})
    df["Entries"] = one["EntryTapCount"].reindex(dates).values

    df["IsStrike"] = df["Date"].between(*STRIKE)
    df["IsHoliday"] = df["Date"].isin(BANK_HOLIDAYS)
    df["IsXmas"] = (
        ((df["Date"].dt.month == 12) & (df["Date"].dt.day >= 23))
        | ((df["Date"].dt.month == 1) & (df["Date"].dt.day <= 2))
    )
    df = mask_disruptions(df)
    return df.reset_index(drop=True)


def mask_disruptions(df, fit_end=None):
    """Flag disrupted days and set y, using weekday medians from data up to fit_end.

    Restricting the medians to data up to the forecast origin keeps the training
    mask free of information from after the origin.
    """
    df = df.copy()
    ref = df if fit_end is None else df[df["Date"] <= fit_end]
    # The median, unlike the mean, is not pulled towards the disrupted days
    # that the ratio is meant to find.
    dow_median = ref.groupby(ref["Date"].dt.dayofweek)["Entries"].median()
    df["DowRatio"] = df["Entries"] / df["Date"].dt.dayofweek.map(dow_median).values

    # Bank holidays are exempt from the ratio test because their low demand is
    # real, but a day with no entries at all is a closure whatever the date.
    df["IsDisrupted"] = (
        (~df["DowRatio"].between(DISRUPT_LO, DISRUPT_HI) & ~df["IsHoliday"])
        | (df["Entries"] == 0)
    ).fillna(False)

    df["y"] = df["Entries"]
    df.loc[df["IsStrike"] | df["IsDisrupted"], "y"] = np.nan
    return df


def calendar_features(dates, holiday, xmas, k=FOURIER_K):
    """Day-of-week indicators (Monday is the reference), k annual harmonics and holiday flags."""
    cols = {}
    for d in range(1, 7):
        cols[f"dow_{d}"] = (dates.dayofweek == d).astype(float)

    for j in range(1, k + 1):
        ang = 2 * np.pi * j * dates.dayofyear.values / 365.25
        cols[f"sin{j}"] = np.sin(ang)
        cols[f"cos{j}"] = np.cos(ang)

    cols["holiday"] = np.asarray(holiday, dtype=float)
    cols["xmas"] = np.asarray(xmas, dtype=float)
    return pd.DataFrame(cols)


def features_for(df, k=FOURIER_K):
    return calendar_features(pd.DatetimeIndex(df["Date"]),
                             df["IsHoliday"], df["IsXmas"], k)


def fit_predict(X_train, y_train, X_future, model):
    if model == "poisson":
        # Unpenalised: with daily counts in the thousands, an L2 penalty of order 1
        # is negligible against the deviance. Scaling only helps the solver.
        m = make_pipeline(StandardScaler(),
                          PoissonRegressor(alpha=0.0, max_iter=5000))
        m.fit(X_train, y_train)
        return m.predict(X_future)

    if model == "lognormal":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(X_train, np.log(y_train))
        smear = np.mean(np.exp(np.log(y_train) - m.predict(X_train)))   # Duan (1983)
        return np.exp(m.predict(X_future)) * smear

    if model == "xgb":
        m = XGBRegressor(objective="count:poisson", max_delta_step=0.7,
                         n_estimators=400, learning_rate=0.05, max_depth=4,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=0, verbosity=0)
        m.fit(X_train, y_train)
        return m.predict(X_future)

    raise ValueError(f"unknown model {model!r}")


def seasonal_naive(train, target_dates, weeks=8):
    """Day-of-week means over the last `weeks` weeks of training data."""
    recent = train[train["Date"] > train["Date"].max() - pd.Timedelta(weeks=weeks)]
    by_dow = recent.groupby(recent["Date"].dt.dayofweek)["y"].mean()
    return pd.Series(target_dates.dayofweek).map(by_dow).values


def forecast(df, origin, target_dates, k=FOURIER_K, models=MODEL_NAMES[1:]):
    """Seasonal naive plus `models`, trained up to origin.

    Calendar features need one fit for every horizon.
    """
    df = mask_disruptions(df, origin)
    X = features_for(df, k)
    is_train = (df["Date"] <= origin).values & df["y"].notna().values
    is_future = df["Date"].isin(target_dates).values

    out = pd.DataFrame({"Date": df.loc[is_future, "Date"].values})
    out["naive"] = seasonal_naive(df[is_train], pd.DatetimeIndex(out["Date"]))
    for m in models:
        out[m] = fit_predict(X[is_train], df["y"].values[is_train],
                             X[is_future], m)
    return out


def rolling_backtest(df, gap_days, n_origins=10, step_days=30, horizon=30,
                     min_train_days=240, k=FOURIER_K, models=MODEL_NAMES[1:]):
    """Walk-forward origins, each forecasting `horizon` days starting `gap_days` ahead."""
    last, first = df["Date"].max(), df["Date"].min()
    latest = last - pd.Timedelta(days=gap_days + horizon)
    origins = [latest - pd.Timedelta(days=step_days * i) for i in range(n_origins)]
    origins = [o for o in origins if (o - first).days >= min_train_days][::-1]

    # MASE scale: in-sample seasonal naive (lag 7) error
    scale = df.set_index("Date")["y"].diff(7).abs().mean()

    rows = []
    for o in origins:
        tgt = pd.date_range(o + pd.Timedelta(days=gap_days), periods=horizon, freq="D")
        pred = forecast(df, o, tgt, k, models)
        actual = mask_disruptions(df, o).set_index("Date")["y"].reindex(tgt).values

        for m in ["naive", *models]:
            e = actual - pred[m].values
            ok = ~np.isnan(e)
            rows.append({
                "origin": o, "model": m,
                "mae": np.abs(e[ok]).mean(),
                "mase": np.abs(e[ok]).mean() / scale,
                "bias": e[ok].mean() / actual[ok].mean(),
                "log_resid": np.log(actual[ok] / pred[m].values[ok]),
                "total_ratio": actual[ok].sum() / pred[m].values[ok].sum(),
            })
    return pd.DataFrame(rows)


def lag_features(df, h):
    """Lags for a forecast h days ahead.

    Masked days are filled forwards with the last observed value. Interpolation
    would borrow values from after a masked day, which can lie beyond the origin.
    """
    logy = np.log(df["y"]).ffill()
    same_dow = 7 * int(np.ceil(h / 7))          # most recent matching weekday

    return pd.DataFrame({
        "last_obs": logy.shift(h),
        "same_dow": logy.shift(same_dow),
        "recent_mean": logy.shift(h).rolling(28, min_periods=14).mean(),
    })


def forecast_with_lags(df, origin, target_dates, model="poisson", k=FOURIER_K):
    """Direct multi-step: one fit per horizon, using only lags observed at the origin."""
    df = mask_disruptions(df, origin)
    Xcal = features_for(df, k)
    y = df["y"].values
    is_train = (df["Date"] <= origin).values & df["y"].notna().values

    rows = []
    for d in target_dates:
        h = (d - origin).days
        idx = df.index[df["Date"] == d]
        if len(idx) == 0:
            continue
        X = pd.concat([Xcal, lag_features(df, h)], axis=1)
        ok = is_train & X.notna().all(axis=1).values
        rows.append({"Date": d, "h": h,
                     model: fit_predict(X[ok], y[ok], X.loc[idx], model)[0]})
    return pd.DataFrame(rows)
