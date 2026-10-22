"""
Retail Sales Forecasting: Time Series Analysis & Machine Learning
================================================================

Forecasts daily revenue for a UK online retailer 13 weeks ahead (covering the
Sep–Dec peak season) and compares five forecasting approaches.

Data : Online Retail II, UCI Machine Learning Repository (CC BY 4.0)
       https://archive.ics.uci.edu/dataset/502/online+retail+ii
       1M+ real transactions, Dec 2009 – Dec 2011

Steps
  1. Clean raw transactions and aggregate to daily revenue
  2. Exploratory analysis: trend, weekly / yearly seasonality, peak drivers
  3. Leakage-free calendar features
  4. Models: seasonal-naive baseline, Holt-Winters, Linear Regression,
     Random Forest and Gradient Boosting (tuned with time-series CV), ensemble
  5. Evaluate on a 13-week hold-out (MAE, RMSE, WAPE, weekly WAPE)
  6. Retrain on all data and forecast the next two weeks

Usage
  pip install -r requirements.txt
  python sales_forecasting.py

  Uses daily_sales.csv if it is present. Otherwise it builds it from
  online_retail_II.xlsx (download and unzip it from the UCI link above).
  Charts are saved to ./images, the forecast to forecast_next_2_weeks.csv.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (12, 4.5), "figure.dpi": 110, "axes.titleweight": "bold"})

RAW_FILE = Path("online_retail_II.xlsx")
DAILY_FILE = Path("daily_sales.csv")
IMG = Path("images")
TEST_START = pd.Timestamp("2011-09-09")    # last 13 weeks are held out for testing
GBP = mtick.FuncFormatter(lambda x, _: f"£{x / 1000:,.0f}k")

# Stock codes that are fees, postage or manual adjustments, not products
NON_PRODUCT_CODES = {"POST", "DOT", "M", "D", "C2", "BANK CHARGES", "AMAZONFEE",
                     "CRUK", "PADS", "ADJUST", "ADJUST2", "TEST001", "TEST002", "S", "B"}

# Days the shop took no orders (besides Saturdays and the 24 Dec – 3 Jan shutdown)
UK_HOLIDAYS = pd.to_datetime([
    "2010-04-02", "2010-04-04", "2010-04-05", "2010-05-03", "2010-05-31", "2010-08-30",
    "2011-04-22", "2011-04-24", "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29",
])


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMG / name)
    plt.close(fig)
    print(f"  chart saved: images/{name}")


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
def build_daily_sales(raw_path: Path) -> pd.DataFrame:
    """Clean raw transactions and aggregate them to one row per calendar day."""
    sheets = pd.read_excel(raw_path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    y1, y2 = sheets["Year 2009-2010"], sheets["Year 2010-2011"]
    y1 = y1[y1["InvoiceDate"] < y2["InvoiceDate"].min()]          # sheets overlap 1–9 Dec 2010
    tx = pd.concat([y1, y2], ignore_index=True)

    n0 = len(tx)
    tx = tx[~tx["Invoice"].str.startswith("C")]                    # cancelled invoices
    tx = tx[(tx["Quantity"] > 0) & (tx["Price"] > 0)]               # returns / bad prices
    tx = tx[~tx["StockCode"].str.upper().isin(NON_PRODUCT_CODES)]   # fees, postage
    tx = tx.drop_duplicates()
    tx = tx.assign(Revenue=tx["Quantity"] * tx["Price"], date=tx["InvoiceDate"].dt.normalize())
    print(f"Transactions: {n0:,} raw -> {len(tx):,} after cleaning ({1 - len(tx) / n0:.1%} removed)")

    daily = tx.groupby("date").agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"),
                                   customers=("Customer ID", "nunique"), units=("Quantity", "sum"))
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0).rename_axis("date").reset_index()   # closed days = 0
    daily["revenue"] = daily["revenue"].round(2)
    return daily


def load_data() -> pd.DataFrame:
    section("1. LOAD DATA")
    if not DAILY_FILE.exists():
        print(f"{DAILY_FILE} not found, building it from {RAW_FILE} (takes a few minutes)...")
        build_daily_sales(RAW_FILE).to_csv(DAILY_FILE, index=False)
    df = pd.read_csv(DAILY_FILE, parse_dates=["date"]).set_index("date")
    print(f"{len(df)} days | {df.index.min():%d %b %Y} -> {df.index.max():%d %b %Y} "
          f"| total revenue £{df.revenue.sum():,.0f}")
    print(df.describe().round(1))
    return df


def is_closed(dates: pd.DatetimeIndex) -> np.ndarray:
    """Business calendar: no orders on Saturdays, UK bank holidays and 24 Dec – 3 Jan."""
    xmas = ((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 3))
    return np.asarray((dates.dayofweek == 5) | dates.isin(UK_HOLIDAYS) | xmas)


# ---------------------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------------------
def explore(df: pd.DataFrame) -> None:
    section("2. EXPLORATORY ANALYSIS")
    print("Closed-day rule vs actual sales (every zero-sales day should be 'closed'):")
    print(pd.crosstab(df["closed"], df["revenue"] > 0, rownames=["closed (rule)"], colnames=["had sales"]))
    open_df = df[~df.closed]

    # Daily revenue with rolling mean
    fig, ax = plt.subplots()
    ax.plot(open_df.index, open_df.revenue, lw=0.6, alpha=0.45, label="Daily revenue (open days)")
    ax.plot(open_df.index, open_df.revenue.rolling(28, min_periods=10).mean(), lw=2.2, label="28-day rolling mean")
    ax.yaxis.set_major_formatter(GBP)
    ax.set_title("Daily revenue, Dec 2009 – Dec 2011")
    ax.legend(loc="upper left")
    save(fig, "daily_revenue.png")

    # Monthly revenue, year over year
    monthly = df.revenue.resample("MS").sum().to_frame("revenue")
    monthly["year"], monthly["month"] = monthly.index.year, monthly.index.month
    pivot = monthly[monthly.index >= "2010-01-01"].pivot(index="month", columns="year", values="revenue")
    pivot.index = pd.to_datetime(pivot.index, format="%m").strftime("%b")
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", width=0.75, ax=ax)
    ax.yaxis.set_major_formatter(GBP)
    ax.set_xlabel("")
    ax.set_title("Monthly revenue: 2010 vs 2011")
    ax.tick_params(axis="x", rotation=0)
    save(fig, "monthly_yoy.png")

    peak = lambda y: monthly.loc[f"{y}-09":f"{y}-11", "revenue"].sum()
    print(f"Sep–Nov revenue: 2010 £{peak(2010):,.0f} | 2011 £{peak(2011):,.0f} | growth {peak(2011) / peak(2010) - 1:+.1%}")
    print(f"Share of 2010 revenue earned in Sep–Nov: {peak(2010) / monthly.loc['2010', 'revenue'].sum():.1%}")

    # Weekday pattern
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
    by_day = [open_df.revenue[open_df.index.day_name() == d] for d in order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(by_day, showfliers=False, patch_artist=True, boxprops={"facecolor": "#4C72B0", "alpha": 0.7})
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.yaxis.set_major_formatter(GBP)
    ax.set_title("Revenue by weekday (open days, outliers hidden)")
    save(fig, "weekday.png")
    print("Median revenue by weekday:")
    print(pd.Series([s.median() for s in by_day], index=order).map("£{:,.0f}".format).to_string())

    # STL decomposition: trend + weekly seasonality + residual
    stl = STL(df.revenue, period=7, robust=True).fit()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, (name, s) in zip(axes, [("Trend", stl.trend), ("Weekly seasonality", stl.seasonal), ("Residual", stl.resid)]):
        ax.plot(s, lw=0.8)
        ax.set_title(name, loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(GBP)
    save(fig, "stl_decomposition.png")

    # Peak driver: number of orders vs average order value
    m = df[df.index >= "2010-01-01"].resample("MS")[["revenue", "orders"]].sum()
    m["avg_order_value"] = m.revenue / m.orders
    fig, ax1 = plt.subplots()
    ax1.bar(m.index, m.orders, width=20, alpha=0.6, label="Orders")
    ax2 = ax1.twinx()
    ax2.plot(m.index, m.avg_order_value, color="C3", marker="o", label="Avg order value (£)")
    ax2.grid(False)
    ax1.set_ylabel("Orders / month")
    ax2.set_ylabel("Avg order value (£)")
    ax1.set_title("What drives the peak: mostly more orders (+80% Jul→Nov 2010), order value +25%")
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.88))
    save(fig, "orders_vs_aov.png")
    print("Orders and average order value, Jul–Nov 2010:")
    print(m.loc["2010-07":"2010-11"].round(0).to_string())


# ---------------------------------------------------------------------------
# 3. Features (calendar only, so nothing from the forecast window leaks in)
# ---------------------------------------------------------------------------
def make_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    X = pd.DataFrame(index=idx)
    X["trend"] = (idx - pd.Timestamp("2009-12-01")).days
    X["dayofweek"] = idx.dayofweek
    X["month"] = idx.month
    X["weekofyear"] = idx.isocalendar().week.astype(int).values
    X["dayofyear"] = idx.dayofyear
    X["dayofmonth"] = idx.day
    xmas = pd.to_datetime([f"{y}-12-25" for y in idx.year])
    X["days_to_xmas"] = np.clip((xmas - idx).days, -10, 120)        # >120 days away treated the same
    for k in (1, 2, 3):                                               # smooth yearly seasonality
        X[f"sin{k}"] = np.sin(2 * np.pi * k * idx.dayofyear / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * idx.dayofyear / 365.25)
    for d in range(7):                                                # one-hot weekday (linear model)
        X[f"dow_{d}"] = (idx.dayofweek == d).astype(int)
    return X


LIN_COLS = (["trend", "days_to_xmas"] + [f"{f}{k}" for k in (1, 2, 3) for f in ("sin", "cos")]
            + [f"dow_{d}" for d in (0, 1, 2, 3, 4, 6)])               # Saturday dropped (always closed)
TREE_COLS = ["trend", "dayofweek", "month", "weekofyear", "dayofyear", "dayofmonth", "days_to_xmas"]


def evaluate(name: str, pred, y: pd.Series) -> dict:
    """MAE / RMSE in £, WAPE = sum|error| / sum(actual), plus the same at weekly level."""
    pred = pd.Series(np.asarray(pred), index=y.index)
    wk_a, wk_p = y.resample("W").sum(), pred.resample("W").sum()
    return {"model": name,
            "MAE (£)": mean_absolute_error(y, pred),
            "RMSE (£)": mean_squared_error(y, pred) ** 0.5,
            "WAPE": (y - pred).abs().sum() / y.sum(),
            "Weekly WAPE": (wk_a - wk_p).abs().sum() / wk_a.sum(),
            "Total bias": pred.sum() / y.sum() - 1}


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
def seasonal_naive(df: pd.DataFrame, test_index: pd.DatetimeIndex) -> pd.Series:
    """Same weekday 52 weeks ago, scaled by recent year-on-year growth (a spreadsheet-style rule)."""
    full = df.revenue
    last_year = full.shift(364).reindex(test_index)
    last_year = last_year.where(last_year > 0, full.shift(357).reindex(test_index))   # holiday fallback
    day = pd.Timedelta(days=1)
    recent = full[TEST_START - 56 * day:TEST_START - day].sum()
    prior = full[TEST_START - (56 + 364) * day:TEST_START - (1 + 364) * day].sum()
    print(f"  baseline YoY growth factor: {recent / prior:.3f}")
    return last_year * (recent / prior)


def train_and_evaluate(df: pd.DataFrame):
    section("3-4. FEATURES, TRAIN/TEST SPLIT & MODELS")
    data = df[~df.closed]
    X_all, y_all = make_features(data.index), data.revenue
    train, test = X_all.index < TEST_START, X_all.index >= TEST_START
    X_train, y_train, X_test, y_test = X_all[train], y_all[train], X_all[test], y_all[test]
    print(f"Train: {len(X_train)} open days ({X_train.index.min():%d %b %Y} – {X_train.index.max():%d %b %Y})")
    print(f"Test : {len(X_test)} open days ({X_test.index.min():%d %b %Y} – {X_test.index.max():%d %b %Y})")

    preds = {}
    print("Seasonal naive baseline...")
    preds["Seasonal naive (baseline)"] = seasonal_naive(df, y_test.index)

    print("Holt-Winters (damped trend, weekly seasonality of 6 open days)...")
    hw = ExponentialSmoothing(y_train.values, trend="add", damped_trend=True,
                              seasonal="mul", seasonal_periods=6).fit()
    preds["Holt-Winters"] = hw.forecast(len(y_test))

    # Target is log revenue: weekday and seasonal effects are multiplicative
    print("Linear Regression (log target)...")
    lr = LinearRegression().fit(X_train[LIN_COLS], np.log1p(y_train))
    preds["Linear Regression"] = np.expm1(lr.predict(X_test[LIN_COLS]))

    tscv = TimeSeriesSplit(n_splits=4, test_size=60)    # every validation fold comes after its training data
    print("Random Forest, tuning with time-series cross-validation...")
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [300], "max_depth": [6, 10, None], "min_samples_leaf": [1, 3, 5], "max_features": [0.5, 1.0]},
        cv=tscv, scoring="neg_mean_absolute_error").fit(X_train[TREE_COLS], np.log1p(y_train))
    print(f"  best params: {rf_search.best_params_}")
    preds["Random Forest (tuned)"] = np.expm1(rf_search.predict(X_test[TREE_COLS]))

    print("Gradient Boosting, tuning with time-series cross-validation...")
    gb_search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        {"learning_rate": [0.03, 0.1], "max_depth": [3, 5], "max_iter": [200, 500], "min_samples_leaf": [10, 20]},
        cv=tscv, scoring="neg_mean_absolute_error").fit(X_train[TREE_COLS], np.log1p(y_train))
    print(f"  best params: {gb_search.best_params_}")
    preds["Gradient Boosting (tuned)"] = np.expm1(gb_search.predict(X_test[TREE_COLS]))

    preds["Ensemble (LR + RF)"] = (np.asarray(preds["Linear Regression"]) + np.asarray(preds["Random Forest (tuned)"])) / 2

    preds = {k: pd.Series(np.asarray(v), index=y_test.index) for k, v in preds.items()}
    models = {"rf_params": rf_search.best_params_, "gb_params": gb_search.best_params_,
              "rf": rf_search.best_estimator_}
    return X_all, y_all, y_test, preds, models


# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------
def report(y_all, y_test, preds, models) -> str:
    section("5. RESULTS ON THE 13-WEEK HOLD-OUT (9 Sep – 9 Dec 2011)")
    res = pd.DataFrame([evaluate(k, v, y_test) for k, v in preds.items()]).set_index("model").sort_values("WAPE")
    res["WAPE vs baseline"] = res["WAPE"] / res.loc["Seasonal naive (baseline)", "WAPE"] - 1
    shown = res.copy()
    for c in ["MAE (£)", "RMSE (£)"]:
        shown[c] = shown[c].map("£{:,.0f}".format)
    for c in ["WAPE", "Weekly WAPE"]:
        shown[c] = shown[c].map("{:.1%}".format)
    for c in ["Total bias", "WAPE vs baseline"]:
        shown[c] = shown[c].map("{:+.1%}".format)
    print(shown.to_string())
    res.to_csv("model_results.csv")

    best = res.index[0]
    best_pred, base_pred = preds[best], preds["Seasonal naive (baseline)"]
    print(f"\nBest model: {best}")

    # Forecast vs actual, daily
    fig, ax = plt.subplots()
    hist = y_all[y_all.index >= "2011-06-01"]
    ax.plot(hist.index, hist, color="grey", lw=0.8, alpha=0.7, label="Actual")
    ax.plot(best_pred.index, best_pred, color="C3", lw=1.8, label=f"Forecast: {best}")
    ax.plot(base_pred.index, base_pred, color="C0", lw=1, ls="--", alpha=0.8, label="Baseline")
    ax.axvline(TEST_START, color="black", ls=":", lw=1)
    ax.text(TEST_START, ax.get_ylim()[1] * 0.95, "  test period ->", fontsize=9)
    ax.yaxis.set_major_formatter(GBP)
    ax.set_title("13-week hold-out: forecast vs actual daily revenue")
    ax.legend(loc="upper left")
    save(fig, "forecast_vs_actual.png")

    # Weekly totals
    wk = pd.DataFrame({"Actual": y_test.resample("W").sum(), best: best_pred.resample("W").sum(),
                       "Baseline": base_pred.resample("W").sum()})
    fig, ax = plt.subplots()
    wk.plot(kind="bar", width=0.8, ax=ax, color=["grey", "C3", "C0"])
    ax.set_xticklabels(wk.index.strftime("%d %b"), rotation=0)
    ax.yaxis.set_major_formatter(GBP)
    ax.set_xlabel("Week ending")
    ax.set_title("Weekly totals: forecast vs actual")
    save(fig, "weekly_forecast.png")

    # Residuals
    resid = y_test - best_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lim = [0, max(y_test.max(), best_pred.max()) * 1.05]
    axes[0].scatter(best_pred, y_test, alpha=0.6)
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set(xlabel="Forecast", ylabel="Actual", title="Actual vs forecast")
    axes[0].xaxis.set_major_formatter(GBP)
    axes[0].yaxis.set_major_formatter(GBP)
    axes[1].hist(resid, bins=25, alpha=0.8)
    axes[1].set_title("Residual distribution (actual − forecast)")
    axes[1].xaxis.set_major_formatter(GBP)
    save(fig, "residuals.png")
    print("Largest misses:", resid.abs().sort_values(ascending=False).head(5).index.strftime("%a %d %b %Y").tolist())

    # Feature importance
    imp = pd.Series(models["rf"].feature_importances_, index=TREE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    imp.plot(kind="barh", ax=ax, color="C2")
    ax.set_title("Random Forest feature importance")
    save(fig, "feature_importance.png")
    return best


# ---------------------------------------------------------------------------
# 6. Forecast the next two weeks with the best approach, retrained on all data
# ---------------------------------------------------------------------------
def forecast_future(X_all, y_all, best: str, models) -> None:
    section("6. FORECAST: 10 – 23 DEC 2011")
    future_idx = pd.date_range("2011-12-10", "2011-12-23", freq="D")
    future_open = future_idx[~is_closed(future_idx)]
    X_f = make_features(future_open)
    y_log = np.log1p(y_all)

    def lr():
        return np.expm1(LinearRegression().fit(X_all[LIN_COLS], y_log).predict(X_f[LIN_COLS]))

    def rf():
        m = RandomForestRegressor(random_state=42, n_jobs=-1, **models["rf_params"])
        return np.expm1(m.fit(X_all[TREE_COLS], y_log).predict(X_f[TREE_COLS]))

    def gb():
        m = HistGradientBoostingRegressor(random_state=42, **models["gb_params"])
        return np.expm1(m.fit(X_all[TREE_COLS], y_log).predict(X_f[TREE_COLS]))

    builders = {"Linear Regression": lr, "Random Forest (tuned)": rf, "Gradient Boosting (tuned)": gb,
                "Ensemble (LR + RF)": lambda: (lr() + rf()) / 2}
    model_used = best if best in builders else "Ensemble (LR + RF)"   # baseline / HW can't use the ML pipeline

    forecast = pd.Series(0.0, index=future_idx, name="forecast_revenue")
    forecast[future_open] = builders[model_used]()
    out = forecast.round(0).to_frame()
    out["weekday"] = out.index.day_name()
    out["status"] = np.where(is_closed(future_idx), "closed", "open")
    out.to_csv("forecast_next_2_weeks.csv", index_label="date")
    print(f"Model used: {model_used}")
    print(out.to_string())
    print(f"Two-week total forecast: £{forecast.sum():,.0f}  (saved to forecast_next_2_weeks.csv)")


def main() -> None:
    IMG.mkdir(exist_ok=True)
    df = load_data()
    df["closed"] = is_closed(df.index)
    explore(df)
    X_all, y_all, y_test, preds, models = train_and_evaluate(df)
    best = report(y_all, y_test, preds, models)
    forecast_future(X_all, y_all, best, models)
    print("\nDone.")


if __name__ == "__main__":
    main()
