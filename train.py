"""Assignment 1 - training pipeline: hotel booking cancellations."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score

DATA_PATH = "data/hotel_bookings.csv"
TARGET = "is_canceled"
SAMPLE_SIZE = 18_000
SEED = 42
MODELS_DIR = Path("models")

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

def evaluate(y, p, threshold=0.5):
    """Metrics for predicted probabilities p against true labels y."""
    pred = (p >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "accuracy": float((pred == y).mean()),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def print_table(results):
    print(f"\n{'model':<22}{'split':<6}{'auc':>7}{'logloss':>9}{'acc':>7}{'prec':>7}{'recall':>8}")
    for (name, split), m in results.items():
        print(f"{name:<22}{split:<6}{m['auc']:>7.3f}{m['logloss']:>9.3f}"
              f"{m['accuracy']:>7.3f}{m['precision']:>7.3f}{m['recall']:>8.3f}")


def main():
    df = load_data()
    print("full data:", df.shape)

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
    print("shapes:", X_train.shape, X_val.shape, X_test.shape)

    Xtr, Xva, Xte = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()
    results = {}

    # --- naive baseline: always predict the training cancellation rate ---
    for split, y in [("val", y_val), ("test", y_test)]:
        results[("baseline", split)] = evaluate(y, np.full(len(y), y_train.mean()))

    # --- scikit-learn: pick C on validation log loss (the only knob we tune) ---
    best_C, best_ll = None, np.inf
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_train)
        ll = log_loss(y_val, m.predict_proba(Xva)[:, 1])
        print(f"sklearn C={C:<6} val logloss={ll:.4f}")
        if ll < best_ll:
            best_C, best_ll = C, ll
    print("chosen C =", best_C)

    sk = LogisticRegression(C=best_C, max_iter=2000).fit(Xtr, y_train)
    for split, X, y in [("val", Xva, y_val), ("test", Xte, y_test)]:
        results[("sklearn", split)] = evaluate(y, sk.predict_proba(X)[:, 1])

    print_table(results)

    # --- save everything the app will need (it must never retrain) ---
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(sk, MODELS_DIR / "sklearn_logreg.joblib")
    joblib.dump({"enc": enc, "scaler": scaler, "scale_cols": SCALE_COLS,
                 "columns": list(X_train.columns), "best_C": best_C},
                MODELS_DIR / "preprocessing.joblib")
    np.savez_compressed(MODELS_DIR / "splits.npz",
                        X_train=Xtr.astype("float32"), y_train=y_train,
                        X_val=Xva.astype("float32"), y_val=y_val,
                        X_test=Xte.astype("float32"), y_test=y_test)
    print("\nsaved to", MODELS_DIR.resolve())


# No __main__ guard on purpose: I haven't seen main.py, so this runs
# whether it executes the file as a script or imports it.
main()
