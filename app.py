"""Assignment 1 - Gradio dashboard: hotel booking cancellations.

Loads only what train.py saved into models/ (it never retrains anything):
  sklearn_logreg.joblib, manual_torch.pt, nn_torch.pt, preprocessing.joblib,
  splits.npz, metrics.json
Run via:  uv run python main.py app
"""
import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

MODELS_DIR = Path("models")
MODEL_NAMES = ["sklearn", "manual_torch", "nn_torch"]
SPLITS = ["train", "val", "test"]
COLORS = {"sklearn": "#1f77b4", "manual_torch": "#ff7f0e", "nn_torch": "#2ca02c"}


# ---------- load saved artifacts (inference only) ----------
def load_artifacts():
    needed = ["sklearn_logreg.joblib", "manual_torch.pt", "nn_torch.pt",
              "preprocessing.joblib", "splits.npz", "metrics.json"]
    missing = [f for f in needed if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {MODELS_DIR}/. Run `uv run python main.py train` first.")

    sk = joblib.load(MODELS_DIR / "sklearn_logreg.joblib")
    prep = joblib.load(MODELS_DIR / "preprocessing.joblib")
    manual = torch.load(MODELS_DIR / "manual_torch.pt")
    nn_state = torch.load(MODELS_DIR / "nn_torch.pt")
    data = np.load(MODELS_DIR / "splits.npz")
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    X = {s: data[f"X_{s}"].astype("float32") for s in SPLITS}
    y = {s: data[f"y_{s}"] for s in SPLITS}

    nn_model = nn.Linear(X["train"].shape[1], 1)
    nn_model.load_state_dict(nn_state)
    nn_model.eval()

    probs = {}
    for s in SPLITS:
        Xt = torch.tensor(X[s])
        with torch.no_grad():
            probs[("sklearn", s)] = sk.predict_proba(X[s])[:, 1]
            probs[("manual_torch", s)] = torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy()
            probs[("nn_torch", s)] = torch.sigmoid(nn_model(Xt).squeeze(1)).numpy()

    # undo standardisation for readable feature histograms
    columns = prep["columns"]
    idx = [columns.index(c) for c in prep["scale_cols"]]
    raw = {}
    for s in SPLITS:
        Xr = X[s].astype("float64").copy()
        Xr[:, idx] = prep["scaler"].inverse_transform(Xr[:, idx])
        raw[s] = pd.DataFrame(Xr, columns=columns)
    return columns, y, probs, raw, metrics


COLUMNS, Y, PROBS, RAW, METRICS = load_artifacts()
COST_FN = METRICS["cost"]["cost_fn"]
COST_FP = METRICS["cost"]["cost_fp"]


# ---------- tab 1: compare the three models ----------
def metrics_table():
    rows = []
    for key, m in METRICS["results"].items():
        name, split = key.split("|")
        rows.append({"model": name, "split": split, **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)


def calibration_plot(split):
    """Predicted probability vs actual cancellation rate, per model (10 equal-size bins)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="gray")))
    y = Y[split]
    for name in MODEL_NAMES:
        p = PROBS[(name, split)]
        bins = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=g["p"], y=g["y"], mode="lines+markers", name=name,
                                 line=dict(color=COLORS[name])))
    fig.update_layout(title=f"Predicted probability vs actual cancellation rate ({split})",
                      xaxis_title="mean predicted probability",
                      yaxis_title="actual cancellation rate", height=420)
    return fig


def agreement_plot(split):
    """Do the three implementations predict the same probabilities?"""
    rng = np.random.default_rng(0)
    n = len(Y[split])
    sel = rng.choice(n, size=min(1500, n), replace=False)
    ref = PROBS[("sklearn", split)][sel]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(dash="dash", color="gray")))
    for name in ["manual_torch", "nn_torch"]:
        fig.add_trace(go.Scatter(x=ref, y=PROBS[(name, split)][sel], mode="markers",
                                 name=name, marker=dict(size=4, color=COLORS[name], opacity=0.5)))
    fig.update_layout(title=f"Each PyTorch model vs scikit-learn ({split}, 1,500 bookings)",
                      xaxis_title="sklearn probability", yaxis_title="PyTorch probability",
                      height=420)
    return fig


def compare(split):
    return calibration_plot(split), agreement_plot(split)


# ---------- tab 2: distributions ----------
def target_plot():
    rates = [float(Y[s].mean()) for s in SPLITS]
    fig = go.Figure(go.Bar(x=SPLITS, y=rates, text=[f"{r:.1%}" for r in rates],
                           textposition="outside", marker_color="#636efa"))
    fig.update_layout(title="Cancellation rate by split (time-based split: seasonal drift)",
                      yaxis_title="share cancelled", yaxis_range=[0, 0.6], height=360)
    return fig


def feature_plot(feature, split):
    df, y = RAW[split], Y[split]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df.loc[y == 0, feature], name="kept", opacity=0.6,
                               marker_color="#2ca02c"))
    fig.add_trace(go.Histogram(x=df.loc[y == 1, feature], name="cancelled", opacity=0.6,
                               marker_color="#d62728"))
    fig.update_layout(barmode="overlay", title=f"{feature} ({split}, original scale)",
                      xaxis_title=feature, yaxis_title="bookings", height=420)
    return fig


# ---------- tab 3: threshold, confusion matrix, business cost ----------
def cost_per_booking(y, p, t):
    flagged = p >= t
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return (fp * COST_FP + fn * COST_FN) / len(y)


def threshold_view(model, split, t):
    y, p = Y[split], PROBS[(model, split)]
    flagged = p >= t
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())

    cm = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=["predicted kept", "predicted cancel"],
        y=["actually kept", "actually cancelled"], text=[[tn, fp], [fn, tp]],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix: {model}, {split}, threshold {t:.2f}",
                     yaxis_autorange="reversed", height=380)

    grid = np.linspace(0.05, 0.95, 91)
    costs = [cost_per_booking(y, p, g) for g in grid]
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name="cost"))
    curve.add_trace(go.Scatter(x=[t], y=[cost_per_booking(y, p, t)], mode="markers",
                               marker=dict(size=12, color="red"), name="current"))
    curve.update_layout(title="Business cost per booking vs threshold",
                        xaxis_title="threshold", yaxis_title="cost (ADR-nights)", height=380)

    cost = cost_per_booking(y, p, t)
    never = float(y.mean()) * COST_FN
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    best = grid[int(np.argmin(costs))]
    md = (f"### Business cost: **{cost:.3f} ADR-nights per booking**\n"
          f"Total on {len(y):,} bookings: **{cost * len(y):,.0f} ADR-nights** "
          f"({fp} false alarms x {COST_FP} + {fn} missed cancellations x {COST_FN}).\n\n"
          f"- Never flagging anyone costs {never:.3f} per booking.\n"
          f"- Precision {prec:.3f}, recall {rec:.3f}.\n"
          f"- Cheapest threshold on this split: {best:.2f}.\n\n"
          f"*Costs: missed cancellation = {COST_FN} night (empty room); "
          f"false alarm = {COST_FP} nights (guest walked).*")
    return cm, curve, md


# ---------- layout ----------
with gr.Blocks(title="Hotel cancellation models") as demo:
    gr.Markdown(
        "# Hotel booking cancellations\n"
        "Same logistic regression trained three ways (scikit-learn, manual PyTorch loop, "
        "`nn.Module` + `torch.optim`) on the same time-based split. "
        "This dashboard only loads the saved models; nothing is retrained.")

    with gr.Tab("Model comparison"):
        split_c = gr.Radio(["val", "test"], value="test", label="Split")
        gr.Dataframe(value=metrics_table(), label="Metrics (threshold 0.5)", interactive=False)
        plot_cal = gr.Plot()
        plot_agree = gr.Plot()
        split_c.change(compare, split_c, [plot_cal, plot_agree])
        demo.load(compare, split_c, [plot_cal, plot_agree])

    with gr.Tab("Distributions"):
        gr.Plot(value=target_plot())
        with gr.Row():
            feat = gr.Dropdown(COLUMNS, value="lead_time_log", label="Feature")
            split_d = gr.Radio(SPLITS, value="train", label="Split")
        plot_feat = gr.Plot()
        feat.change(feature_plot, [feat, split_d], plot_feat)
        split_d.change(feature_plot, [feat, split_d], plot_feat)
        demo.load(feature_plot, [feat, split_d], plot_feat)

    with gr.Tab("Threshold and cost"):
        with gr.Row():
            model_t = gr.Dropdown(MODEL_NAMES, value="sklearn", label="Model")
            split_t = gr.Radio(["val", "test"], value="test", label="Split")
        thr = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
        with gr.Row():
            plot_cm = gr.Plot()
            plot_curve = gr.Plot()
        summary = gr.Markdown()
        for comp in (model_t, split_t, thr):
            comp.change(threshold_view, [model_t, split_t, thr], [plot_cm, plot_curve, summary])
        demo.load(threshold_view, [model_t, split_t, thr], [plot_cm, plot_curve, summary])

demo.launch()
