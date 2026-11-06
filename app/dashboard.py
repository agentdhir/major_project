"""
dashboard.py — FORESIGHT live Security-Operations-Centre (SOC) dashboard.

Run from the project root:
    streamlit run app/dashboard.py

What the panel sees
-------------------
1. Pick an attack scenario (or a harmless decoy scan) and an attacker
   stealth level, then press  ▶ Play.
2. Real KDDTest+ connections stream past, as seen at our gateway.
3. FORESIGHT forecasts — at every connection — the probability that an
   attack will hit within the next 10/25/50/100 connections, the attack
   type, an ETA, and explains WHICH past connections and features triggered
   the warning.  The ground-truth phases (hidden from the model) are shaded
   so you can see the warning arrive BEFORE the attack.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.config import CATEGORIES, MODEL_DIR  # noqa: E402
from foresight.inference import SCENARIOS, Foresight  # noqa: E402

st.set_page_config(page_title="FORESIGHT — Predictive IDS", page_icon="🛡️",
                   layout="wide")

CAT_COLORS = {"Normal": "#9AA5B1", "DoS": "#D64545", "Probe": "#F0A202",
              "R2L": "#8E44AD", "U2R": "#1B1F3B"}
PHASE_COLORS = {1: ("Reconnaissance", "rgba(240,162,2,0.15)"),
                2: ("Attack", "rgba(214,69,69,0.18)"),
                3: ("Decoy scan", "rgba(52,152,219,0.15)")}
LEVELS = {0: ("CLEAR", "#2E7D32", "No attack expected"),
          1: ("EARLY WARNING", "#F0A202", "Attack being prepared"),
          2: ("ATTACK", "#C62828", "Attack in progress")}


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading FORESIGHT models…")
def load_system():
    return Foresight.load()


@st.cache_data(show_spinner="Generating traffic and running FORESIGHT…")
def make_run(scenario, stealth, seed):
    fs = load_system()
    stream = fs.simulate(scenario, stealth=stealth, seed=seed)
    a = fs.analyse(stream)
    recs = fs.source_records(stream)
    return dict(stream=stream, a=a,
                labels=recs["label"].values, services=recs["service"].values,
                protocols=recs["protocol_type"].values)


if not (MODEL_DIR / "foresight_net.pt").exists():
    st.error("No trained model found. Run `python run_pipeline.py` first.")
    st.stop()

fs = load_system()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ FORESIGHT")
st.sidebar.caption("Deep-learning early warning for network attacks")
# Optional URL parameters, e.g. ?scenario=recon_dos&stealth=0.5&seed=3&t=520
# (handy for demo links and screenshots; only used on the first load).
qp = st.query_params
first_load = "key" not in st.session_state
scen_keys = list(SCENARIOS)
scen0 = qp.get("scenario", "recon_dos") if first_load else "recon_dos"
scen = st.sidebar.selectbox("Attack scenario", scen_keys,
                            index=scen_keys.index(scen0) if scen0 in scen_keys else 0,
                            format_func=lambda k: SCENARIOS[k])
stealth = st.sidebar.slider("Attacker stealth (recon intensity ×)", 0.25, 1.0,
                            float(qp.get("stealth", 1.0)) if first_load else 1.0, 0.05,
                            help="Lower = slower, sparser reconnaissance "
                                 "(harder to notice).")
seed = st.sidebar.number_input("Random seed", 0, 9999,
                               int(qp.get("seed", 7)) if first_load else 7)
speed = st.sidebar.slider("Playback speed (connections / tick)", 1, 20, 6)

run = make_run(scen, float(stealth), int(seed))
stream, a = run["stream"], run["a"]
T = len(stream)
W = fs.wcfg.window

key = (scen, float(stealth), int(seed))
if st.session_state.get("key") != key:          # new traffic -> start over
    t0 = int(qp.get("t", W)) if first_load else W
    st.session_state.key = key
    st.session_state.pending_t = min(max(t0, W), T - 1)
    st.session_state.playing = False
# A widget's value may only be changed BEFORE the widget is created.
if "pending_t" in st.session_state:
    st.session_state.t_slider = st.session_state.pop("pending_t")

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Play"):
    st.session_state.playing = True
if c2.button("⏸ Pause"):
    st.session_state.playing = False
    st.session_state.pending_t = st.session_state.get("cur_t", W)
    st.rerun()
if c3.button("⏮ Reset"):
    st.session_state.playing = False
    st.session_state.pending_t = W
    st.rerun()
st.sidebar.slider("Jump to connection", W, T - 1, key="t_slider")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Alarm threshold:** {a.threshold:.3f}  \n"
    "(site-calibrated: 1 % alert budget on unlabelled burn-in traffic)")
st.sidebar.markdown(
    "Shaded regions show the **true** phases — the model never sees them.")

# ------------------------------------------------------------------ header
st.title("FORESIGHT — Predicting Network Attacks Before They Strike")
st.caption("Real KDDTest+ connections replayed as a kill-chain campaign at our "
           "gateway. Each forecast uses only the last 64 connections.")

kpi = st.empty()
chart = st.empty()
lower = st.empty()


def campaign_info():
    c = stream.campaigns.iloc[0]
    return c


def render(t):
    c = campaign_info()
    lvl = int(a.level[t])
    name, color, desc = LEVELS[lvl]
    p100 = float(a.warn[t])
    # ---------------- KPI row
    with kpi.container():
        k1, k2, k3, k4, k5 = st.columns([1.3, 1, 1.2, 1.2, 1])
        k1.markdown(
            f"<div style='background:{color};padding:14px;border-radius:10px;"
            f"color:white;text-align:center'><div style='font-size:13px'>THREAT LEVEL"
            f"</div><div style='font-size:26px;font-weight:700'>{name}</div>"
            f"<div style='font-size:12px'>{desc}</div></div>",
            unsafe_allow_html=True)
        k2.metric("P(attack ≤ 100 conns)", f"{p100:.0%}",
                  delta=f"{p100 - a.threshold:+.0%} vs threshold",
                  delta_color="inverse")
        k3.metric("Expected attack type",
                  fs.attack_type_text(a.type_prob[t]) if lvl >= 1 else "—")
        k4.metric("ETA", fs.eta_text(int(a.eta[t])) if lvl >= 1 else "—")
        now = CATEGORIES[int(np.argmax(a.det_prob[t]))]
        k5.metric("Current connection", now,
                  delta=f"zero-day score {a.anomaly[t]:.0%}", delta_color="off")
    # ---------------- timeline chart
    x = np.arange(T)
    fig = go.Figure()
    for ph, (label, fill) in PHASE_COLORS.items():
        idx = np.flatnonzero(stream.phase == ph)
        if len(idx):
            fig.add_vrect(x0=idx[0], x1=idx[-1], fillcolor=fill, line_width=0,
                          annotation_text=label, annotation_position="top left")
    fig.add_trace(go.Scatter(x=x[:t + 1], y=a.warn[:t + 1], mode="lines",
                             name="FORESIGHT: P(attack within 100)",
                             line=dict(color="#1B4F72", width=2.5)))
    fig.add_trace(go.Scatter(x=x[:t + 1], y=a.prob[:t + 1, 1], mode="lines",
                             name="P(attack within 25)",
                             line=dict(color="#5DADE2", width=1.2, dash="dot")))
    is_att = np.isin(stream.cat_id[:t + 1], [1, 3, 4])
    fig.add_trace(go.Scatter(x=x[:t + 1][is_att], y=np.full(is_att.sum(), 1.02),
                             mode="markers", name="attack connection",
                             marker=dict(color="#D64545", size=5, symbol="line-ns-open")))
    is_pr = stream.cat_id[:t + 1] == 2
    fig.add_trace(go.Scatter(x=x[:t + 1][is_pr], y=np.full(is_pr.sum(), -0.04),
                             mode="markers", name="scan connection",
                             marker=dict(color="#F0A202", size=5, symbol="line-ns-open")))
    fig.add_hline(y=a.threshold, line_dash="dash", line_color="#C0392B",
                  annotation_text="alarm threshold")
    # First alarm + lead time annotation
    al = np.flatnonzero(a.warn[:t + 1] > a.threshold)
    z0 = c.recon_start if c.has_recon else max(0, c.onset - 200)
    pre = al[(al >= z0) & ((al < c.onset) if c.onset >= 0 else True)] if len(al) else al
    if len(pre):
        fa = int(pre[0])
        fig.add_vline(x=fa, line_color="#F0A202", line_width=2)
        if c.onset >= 0 and t >= c.onset:
            fig.add_annotation(x=(fa + c.onset) / 2, y=0.93,
                               text=f"⏱ warned {c.onset - fa} connections early",
                               showarrow=False, font=dict(size=14, color="#1B4F72"),
                               bgcolor="white")
    fig.add_vline(x=t, line_color="black", line_width=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(range=[-0.08, 1.08], title="probability"),
                      xaxis=dict(range=[0, T], title="connection number"),
                      legend=dict(orientation="h", y=-0.25))
    chart.plotly_chart(fig, width="stretch", key=f"tl{t}")
    # ---------------- what is the model looking at?
    with lower.container():
        l1, l2 = st.columns([1.6, 1])
        lo = t - W + 1
        cats = [CATEGORIES[i] for i in stream.cat_id[lo:t + 1]]
        att = a.attn[t] if a.attn is not None else np.ones(W) / W
        fig2 = go.Figure(go.Bar(
            x=np.arange(lo, t + 1), y=att,
            marker_color=[CAT_COLORS[c_] for c_ in cats],
            hovertext=[f"{lab} / {svc}" for lab, svc in
                       zip(run["labels"][lo:t + 1], run["services"][lo:t + 1])]))
        fig2.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10),
                           title="Attention over the last 64 connections "
                                 "(colour = true type: grey normal, "
                                 "orange scan, red/purple attack)",
                           title_font_size=13, yaxis_title="attention",
                           xaxis_title="connection number")
        l1.plotly_chart(fig2, width="stretch", key=f"att{t}")
        with l2:
            st.markdown("**Why? — top features behind this forecast**")
            if lvl >= 1 and not st.session_state.playing:
                e = fs.explain(stream, t)
                df = pd.DataFrame(e["top_features"], columns=["feature", "share"])[::-1]
                fig3 = go.Figure(go.Bar(x=df.share * 100, y=df.feature, orientation="h",
                                        marker_color="#2a78d6"))
                fig3.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                                   xaxis_title="% of the warning explained")
                st.plotly_chart(fig3, width="stretch", key=f"why{t}")
            elif lvl >= 1:
                st.caption("Pause playback to compute the feature explanation.")
            else:
                st.caption("No warning at this moment.")
        # Alert log
        log_rows = []
        ev = np.flatnonzero(np.diff(np.r_[0, (a.level[:t + 1] > 0).astype(int)]) == 1)
        for s in ev[-6:]:
            log_rows.append(dict(connection=int(s), level=LEVELS[int(a.level[s])][0],
                                 forecast=f"{a.warn[s]:.0%}",
                                 expected=fs.attack_type_text(a.type_prob[s])))
        st.markdown("**Alert log**")
        st.dataframe(pd.DataFrame(log_rows) if log_rows else
                     pd.DataFrame(columns=["connection", "level", "forecast", "expected"]),
                     width="stretch", hide_index=True)


if st.session_state.playing:
    t = int(st.session_state.t_slider)
    while t < T - 1:
        render(t)
        st.session_state.cur_t = t            # remembered if Pause is pressed
        t = min(T - 1, t + speed)
        time.sleep(0.12)
    # finished: park the slider at the end and show the final state
    st.session_state.playing = False
    st.session_state.pending_t = T - 1
    st.rerun()
else:
    render(int(st.session_state.t_slider))

with st.expander("ℹ️ How FORESIGHT works"):
    st.markdown("""
* **Input:** the last 64 connections seen at the gateway (41 NSL-KDD features each).
* **Model:** FORESIGHT-Net = connection embedding → causal dilated CNN (bursts)
  → LSTM (slow escalation) → temporal attention (focus + explanation).
* **Outputs:** P(attack within 10/25/50/100 connections) via a survival-style
  head (monotone in time), the attack type, and the current connection class.
* **Alarm:** threshold calibrated on the new network with an unlabelled
  burn-in period (1 % alert budget) — no attack labels needed on site.
* **Zero-day guard:** an autoencoder trained only on normal traffic scores how
  unusual each connection is (percentile vs. known-normal traffic).
""")
