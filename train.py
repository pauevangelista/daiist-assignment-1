"""Assignment 1 - training pipeline: hotel booking cancellations."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/hotel_bookings.csv"
TARGET = "is_canceled"
SAMPLE_SIZE = 18_000
SEED = 42

# Split by arrival date: simulates predicting future bookings from the past
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_END = pd.Timestamp("2017-03-31")

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}

# Not known at booking time, or directly encode the outcome
LEAKY_COLS = [
    "reservation_status",           # literally the outcome
    "reservation_status_date",      # date the outcome happened
    "assigned_room_type",           # assigned near check-in
    "booking_changes",              # accumulates after booking
    "required_car_parking_spaces",  # requested after booking
    "total_of_special_requests",    # can be added after booking
]

CATEGORICALS = ["hotel", "meal", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type",
                "country", "agent"]

# Continuous columns that get standardised (dummies/flags are left alone)
SCALE_COLS = ["lead_time_log", "total_nights_log", "weekend_share", "guests",
              "adr_log", "adr_per_person_log", "prev_cancel_log", "prev_kept_log"]


# ---------- loading and splitting ----------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"]))
    return df


def sample_data(df, n=SAMPLE_SIZE, seed=SEED):
    df = df.sample(n=n, random_state=seed)
    return df.sort_values("arrival_date").reset_index(drop=True)


def time_split(df):
    train = df[df["arrival_date"] <= TRAIN_END]
    val = df[(df["arrival_date"] > TRAIN_END) & (df["arrival_date"] <= VAL_END)]
    test = df[df["arrival_date"] > VAL_END]
    return train.copy(), val.copy(), test.copy()


# ---------- cleaning: fixed rules, nothing learned from the data ----------
def clean(df):
    df = df.copy()
    df["children"] = df["children"].fillna(0)
    df["guests"] = df["adults"] + df["children"] + df["babies"]
    df = df[df["guests"] > 0]                      # zero-guest rows are not real stays
    df["country"] = df["country"].fillna("UNK")
    df["agent"] = df["agent"].apply(lambda a: "none" if pd.isna(a) else str(int(a)))
    df["has_company"] = df["company"].notna().astype(int)
    df["adr"] = df["adr"].clip(lower=0)            # one negative rate is an error
    return df.drop(columns=["company"])


# ---------- feature engineering: anything "learned" is fit on train only ----------
def _group_cats(df, enc):
    cats = df[CATEGORICALS].astype(str).copy()
    cats["country"] = cats["country"].where(cats["country"].isin(enc["countries"]), "OTHER")
    cats["agent"] = cats["agent"].where(cats["agent"].isin(enc["agents"]), "OTHER")
    return cats


def fit_encoders(train):
    enc = {
        "countries": train["country"].value_counts().head(10).index.tolist(),
        "agents": train["agent"].value_counts().head(10).index.tolist(),
        "adr_cap": float(train["adr"].quantile(0.99)),
    }
    cats = _group_cats(train, enc)
    enc["levels"] = {c: sorted(cats[c].unique()) for c in CATEGORICALS}
    return enc


def build_features(df, enc):
    X = pd.DataFrame(index=df.index)
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    adr = df["adr"].clip(upper=enc["adr_cap"])
    month = df["arrival_date"].dt.month
    dow = df["arrival_date"].dt.dayofweek

    X["lead_time_log"] = np.log1p(df["lead_time"])
    X["total_nights_log"] = np.log1p(nights)
    X["weekend_share"] = (df["stays_in_weekend_nights"] / nights.replace(0, np.nan)).fillna(0)
    X["guests"] = df["guests"]
    X["has_children"] = ((df["children"] + df["babies"]) > 0).astype(int)
    X["adr_log"] = np.log1p(adr)
    X["adr_per_person_log"] = np.log1p(adr / df["guests"])
    X["adr_zero"] = (df["adr"] == 0).astype(int)
    X["prev_cancel_log"] = np.log1p(df["previous_cancellations"])
    X["prev_kept_log"] = np.log1p(df["previous_bookings_not_canceled"])
    X["is_repeated_guest"] = df["is_repeated_guest"]
    X["on_waiting_list"] = (df["days_in_waiting_list"] > 0).astype(int)
    X["has_company"] = df["has_company"]
    # cyclical encoding: December and January end up next to each other
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # one-hot with levels fixed from train, so every split has identical columns;
    # a level never seen in train becomes an all-zero row
    cats = _group_cats(df, enc)
    dummies = [pd.get_dummies(
        pd.Series(pd.Categorical(cats[c], categories=enc["levels"][c]), index=df.index),
        prefix=c) for c in CATEGORICALS]
    return pd.concat([X] + dummies, axis=1).astype(float)


def main():
    df = load_data()
    print("full data:", df.shape, "| date range:",
          df["arrival_date"].min().date(), "to", df["arrival_date"].max().date())

    df = clean(sample_data(df).drop(columns=LEAKY_COLS))
    train, val, test = time_split(df)

    enc = fit_encoders(train)
    X_train = build_features(train, enc)
    X_val = build_features(val, enc)
    X_test = build_features(test, enc)
    y_train, y_val, y_test = (p[TARGET].to_numpy() for p in (train, val, test))

    scaler = StandardScaler().fit(X_train[SCALE_COLS])
    for X in (X_train, X_val, X_test):
        X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val),
                       ("test", X_test, y_test)]:
        print(f"{name:5s} X={X.shape}  cancel_rate={y.mean():.3f}  NaNs={int(X.isna().sum().sum())}")

    corr = X_train.corrwith(pd.Series(y_train, index=X_train.index))
    print("\nstrongest correlations with cancellation (train):")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(10).round(3))


# No __main__ guard on purpose: I haven't seen main.py, so this runs
# whether it executes the file as a script or imports it.
main()
