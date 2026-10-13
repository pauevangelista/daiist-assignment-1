"""Assignment 1 - training pipeline: hotel booking cancellations.

Business problem: at booking time, predict which bookings will be cancelled so a
revenue manager can decide how many rooms to overbook.

Run via:  uv run python main.py train
Loads the committed CSV, cleans it, engineers features, splits by arrival date,
trains logistic regression three ways on the same split (scikit-learn, manual
PyTorch loop, nn.Module + torch.optim), compares them with a naive baseline, and
saves everything app.py needs into models/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/hotel_bookings.csv"
MODELS_DIR = Path("models")
TARGET = "is_canceled"
SAMPLE_SIZE = 18_000
SEED = 42

# Split by arrival date: simulates predicting future bookings from the past
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_END = pd.Timestamp("2017-03-31")

# Business costs, in units of one night's room rate (ADR).
# Positive class = "will cancel", so flagging a booking means overbooking against it.
COST_FN = 1.0   # missed cancellation -> room sits empty -> ~1 night of lost revenue
COST_FP = 1.5   # false alarm -> guest shows up, no room -> compensation + goodwill

# Optimisation settings shared by the two PyTorch versions
C_GRID = [0.01, 0.1, 1.0, 10.0]      # sklearn C (inverse regularisation strength)
LR_GRID = [0.3, 0.6, 1.0]            # learning rates tried for the PyTorch loops
EPOCHS = 20_000                      # full-batch gradient steps

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


# ---------- evaluation ----------
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


def booking_cost(y, p, threshold):
    """Average business cost per booking (in ADR-nights) when flagging p >= threshold."""
    flagged = p >= threshold
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return (fp * COST_FP + fn * COST_FN) / len(y)


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmin([booking_cost(y, p, t) for t in grid]))])


# ---------- model 2: logistic regression by hand (Session 5 style) ----------
def train_manual(X, y, C, lr, epochs):
    """Raw tensors, autograd for the gradient, manual gradient-descent update."""
    n, d = X.shape
    lam = 1.0 / (C * n)                       # converts sklearn's C to a mean-loss penalty
    w = torch.zeros(d, requires_grad=True)    # weights (convex problem, so zeros is fine)
    b = torch.zeros(1, requires_grad=True)    # intercept

    for _ in range(epochs):
        z = X @ w + b                                                   # 1. logits
        p = torch.sigmoid(z).clamp(1e-7, 1 - 1e-7)                      # 2. probabilities
        bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()   # 3. loss by hand
        loss = bce + 0.5 * lam * (w ** 2).sum()                         # 4. L2 on w only
        loss.backward()                                                 # 5. autograd
        with torch.no_grad():                                           # 6. manual update
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                                                  # 7. reset grads
        b.grad.zero_()
    return w.detach(), b.detach()


def predict_manual(X, w, b):
    with torch.no_grad():
        return torch.sigmoid(X @ w + b).numpy()


def penalized_loss(X, y, w, b, C):
    """The objective sklearn and the manual loop both minimise (lower = closer to optimum)."""
    n = X.shape[0]
    p = torch.sigmoid(X @ w + b).clamp(1e-7, 1 - 1e-7)
    bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    return float(bce + 0.5 / (C * n) * (w ** 2).sum())


# ---------- model 3: standard PyTorch workflow ----------
def train_nn(X, y, C, lr, epochs):
    """nn.Module + loss object + torch.optim."""
    n, d = X.shape
    lam = 1.0 / (C * n)                        # same C -> penalty conversion as manual
    model = nn.Linear(d, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    loss_fn = nn.BCEWithLogitsLoss()           # sigmoid + BCE in one stable op (mean)
    opt = torch.optim.SGD(                     # weight_decay=lam adds lam*w to the gradient,
        [{"params": [model.weight], "weight_decay": lam},   # equal to the 0.5*lam*||w||^2 penalty
         {"params": [model.bias], "weight_decay": 0.0}],    # bias not penalised, like sklearn
        lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X).squeeze(1), y)
        loss.backward()
        opt.step()
    return model


def predict_nn(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(X).squeeze(1)).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ----- data, cleaning, split, features -----
    df = load_data()
    print("full data:", df.shape, "| date range:",
          df["arrival_date"].min().date(), "to", df["arrival_date"].max().date())

    df = clean(sample_data(df).drop(columns=LEAKY_COLS))
    train, val, test = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s} rows={len(part):6d} cancel_rate={part[TARGET].mean():.3f} "
              f"{part['arrival_date'].min().date()} -> {part['arrival_date'].max().date()}")

    enc = fit_encoders(train)
    X_train = build_features(train, enc)
    X_val = build_features(val, enc)
    X_test = build_features(test, enc)
    y_train, y_val, y_test = (p[TARGET].to_numpy() for p in (train, val, test))

    scaler = StandardScaler().fit(X_train[SCALE_COLS])      # fit on train only
    for X in (X_train, X_val, X_test):
        X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])
    print("feature shapes:", X_train.shape, X_val.shape, X_test.shape)

    columns = list(X_train.columns)
    Xtr, Xva, Xte = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()
    results = {}

    # ----- naive baseline: always predict the training cancellation rate -----
    for split, y in [("val", y_val), ("test", y_test)]:
        results[("baseline", split)] = evaluate(y, np.full(len(y), y_train.mean()))

    # ----- model 1: scikit-learn (the only tuned knob is C, picked on val logloss) -----
    best_C, best_ll = None, np.inf
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_train)
        ll = log_loss(y_val, m.predict_proba(Xva)[:, 1])
        print(f"sklearn C={C:<6} val logloss={ll:.4f}")
        if ll < best_ll:
            best_C, best_ll = C, ll
    print("chosen C =", best_C)

    sk = LogisticRegression(C=best_C, max_iter=2000).fit(Xtr, y_train)
    for split, X, y in [("val", Xva, y_val), ("test", Xte, y_test)]:
        results[("sklearn", split)] = evaluate(y, sk.predict_proba(X)[:, 1])

    # ----- tensors shared by both PyTorch versions -----
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)

    # ----- model 2: manual PyTorch, full batch, C matched to sklearn -----
    best_lr, best_ll, best_wb = None, np.inf, None
    for lr in LR_GRID:                          # learning rate is the only knob tuned
        w, b = train_manual(Xtr_t, ytr_t, C=best_C, lr=lr, epochs=EPOCHS)
        ll = log_loss(y_val, predict_manual(Xva_t, w, b))
        print(f"manual lr={lr:<5} val logloss={ll:.4f}")
        if ll < best_ll:
            best_lr, best_ll, best_wb = lr, ll, (w, b)
    w_man, b_man = best_wb
    print("chosen manual lr =", best_lr)
    for split, X, y in [("val", Xva_t, y_val), ("test", Xte_t, y_test)]:
        results[("manual_torch", split)] = evaluate(y, predict_manual(X, w_man, b_man))

    # ----- model 3: nn.Module + torch.optim, same lr / epochs / C as manual -----
    nn_model = train_nn(Xtr_t, ytr_t, C=best_C, lr=best_lr, epochs=EPOCHS)
    for split, X, y in [("val", Xva_t, y_val), ("test", Xte_t, y_test)]:
        results[("nn_torch", split)] = evaluate(y, predict_nn(nn_model, X))

    # ----- do the three agree? -----
    p_sk = sk.predict_proba(Xte)[:, 1]
    p_man = predict_manual(Xte_t, w_man, b_man)
    p_nn = predict_nn(nn_model, Xte_t)
    print("\nagreement on test probabilities (max abs difference):")
    print("  manual vs sklearn:", float(np.abs(p_man - p_sk).max()))
    print("  nn     vs manual :", float(np.abs(p_nn - p_man).max()))
    print("  nn     vs sklearn:", float(np.abs(p_nn - p_sk).max()))
    print("max |w_manual - w_sklearn| =", float(np.abs(w_man.numpy() - sk.coef_[0]).max()))

    # Both methods minimise the same penalised loss; the lower value is closer to the optimum
    w_sk = torch.tensor(sk.coef_[0], dtype=torch.float32)
    b_sk = torch.tensor(sk.intercept_, dtype=torch.float32)
    w_nn = nn_model.weight.detach().squeeze(0)
    b_nn = nn_model.bias.detach()
    print("objective sklearn:", penalized_loss(Xtr_t, ytr_t, w_sk, b_sk, best_C))
    print("objective manual :", penalized_loss(Xtr_t, ytr_t, w_man, b_man, best_C))
    print("objective nn     :", penalized_loss(Xtr_t, ytr_t, w_nn, b_nn, best_C))

    print_table(results)

    # ----- ablation: is the suspicious deposit_type quirk doing the work? -----
    keep = [i for i, c in enumerate(columns) if not c.startswith("deposit_type_")]
    sk_nodep = LogisticRegression(C=best_C, max_iter=2000).fit(Xtr[:, keep], y_train)
    ablation = {}
    print("\nablation: sklearn without deposit_type")
    for split, X, y in [("val", Xva, y_val), ("test", Xte, y_test)]:
        m = evaluate(y, sk_nodep.predict_proba(X[:, keep])[:, 1])
        ablation[split] = m
        print(f"  {split:<5} auc={m['auc']:.3f} logloss={m['logloss']:.3f} "
              f"recall={m['recall']:.3f} (with deposit_type: "
              f"auc={results[('sklearn', split)]['auc']:.3f})")

    # ----- business cost: threshold chosen on val, judged on test -----
    p_val = sk.predict_proba(Xva)[:, 1]
    t_star = best_threshold(y_val, p_val)
    cost = {
        "never_flag_test": booking_cost(y_test, np.zeros(len(y_test)), 0.5),
        "threshold_0.5_test": booking_cost(y_test, p_sk, 0.5),
        "val_best_threshold": t_star,
        "val_best_threshold_test_cost": booking_cost(y_test, p_sk, t_star),
    }
    print(f"\nbusiness cost per booking, in ADR-nights (FN={COST_FN}, FP={COST_FP}):")
    print(f"  never flag anyone      : {cost['never_flag_test']:.3f}")
    print(f"  sklearn @ threshold 0.5: {cost['threshold_0.5_test']:.3f}")
    print(f"  sklearn @ val-best {t_star:.2f} : {cost['val_best_threshold_test_cost']:.3f}")

    # ----- save everything the app needs (it must never retrain) -----
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(sk, MODELS_DIR / "sklearn_logreg.joblib")
    torch.save({"w": w_man, "b": b_man, "lr": best_lr}, MODELS_DIR / "manual_torch.pt")
    torch.save(nn_model.state_dict(), MODELS_DIR / "nn_torch.pt")
    joblib.dump({"enc": enc, "scaler": scaler, "scale_cols": SCALE_COLS,
                 "columns": columns, "best_C": best_C, "best_lr": best_lr},
                MODELS_DIR / "preprocessing.joblib")
    np.savez_compressed(MODELS_DIR / "splits.npz",
                        X_train=Xtr.astype("float32"), y_train=y_train,
                        X_val=Xva.astype("float32"), y_val=y_val,
                        X_test=Xte.astype("float32"), y_test=y_test)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({
            "results": {f"{k[0]}|{k[1]}": {m: float(v) for m, v in d.items()}
                        for k, d in results.items()},
            "ablation_no_deposit": {s: {m: float(v) for m, v in d.items()}
                                    for s, d in ablation.items()},
            "cost": {"cost_fn": COST_FN, "cost_fp": COST_FP, **cost},
            "config": {"best_C": best_C, "best_lr": best_lr, "epochs": EPOCHS,
                       "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test)},
        }, f, indent=2)
    print("\nsaved to", MODELS_DIR.resolve())


# No __main__ guard on purpose: this runs whether main.py executes the file as a
# script or imports it.
main()
