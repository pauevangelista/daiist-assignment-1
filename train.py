"""Assignment 1 - training pipeline: hotel booking cancellations."""
import pandas as pd

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


def main():
    df = load_data()
    print("full data:", df.shape, "| date range:",
          df["arrival_date"].min().date(), "to", df["arrival_date"].max().date())

    df = sample_data(df).drop(columns=LEAKY_COLS)
    train, val, test = time_split(df)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s} rows={len(part):6d}  cancel_rate={part[TARGET].mean():.3f}  "
              f"{part['arrival_date'].min().date()} -> {part['arrival_date'].max().date()}")


# No __main__ guard on purpose: I haven't seen main.py, so this runs
# whether it executes the file as a script or imports it.
main()
