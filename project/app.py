from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")


@st.cache_resource
def load_bundle():
    model_path = Path(__file__).resolve().parent / "shopper_model.joblib"
    return joblib.load(model_path)


bundle = load_bundle()
defaults, ranges, cats = bundle["defaults"], bundle["numeric_ranges"], bundle["categories"]

st.title("Will this visitor buy?")
st.caption("Change the session details on the left. The model estimates the chance that this web session ends in a purchase.")

# ---------- sidebar: model choice ----------
mode = st.sidebar.radio(
    "Model version",
    ["Full model", "Early-session model"],
    help="The early-session model ignores PageValues, BounceRates and ExitRates, which are "
         "only fully known when a session ends. It is the one to use for real-time decisions.",
)
key = "full" if mode == "Full model" else "early"
model, threshold = bundle[key]["model"], bundle[key]["threshold"]
dropped = bundle["early"]["dropped"] if key == "early" else []

st.sidebar.markdown(f"**Decision threshold:** {threshold:.2f}")
st.sidebar.caption("Sessions at or above this probability are flagged as likely buyers.")

# ---------- inputs ----------
COUNTS = ["Administrative", "Informational", "ProductRelated"]
DURATIONS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
LABELS = {
    "Administrative": "Account / admin pages viewed",
    "Administrative_Duration": "Time on admin pages (sec)",
    "Informational": "Info pages viewed",
    "Informational_Duration": "Time on info pages (sec)",
    "ProductRelated": "Product pages viewed",
    "ProductRelated_Duration": "Time on product pages (sec)",
    "BounceRates": "Bounce rate",
    "ExitRates": "Exit rate",
    "PageValues": "Page value",
    "SpecialDay": "Closeness to a special day (0-1)",
}

values = {}
left, right = st.columns(2)

with left:
    st.subheader("Browsing behaviour")
    for col in COUNTS:
        values[col] = st.slider(LABELS[col], min_value=0, max_value=int(ranges[col][1]),
                                value=int(defaults[col]), step=1)
    for col in DURATIONS:
        values[col] = st.slider(LABELS[col], min_value=0.0, max_value=ranges[col][1],
                                value=float(round(defaults[col], 1)), step=10.0)

with right:
    st.subheader("Session quality")
    for col in ["BounceRates", "ExitRates"]:
        if col in dropped:
            values[col] = float(defaults[col])
            st.slider(LABELS[col] + " (not used by this model)", 0.0, 0.2,
                      value=float(round(defaults[col], 3)), disabled=True)
        else:
            values[col] = st.slider(LABELS[col], min_value=0.0, max_value=0.2,
                                    value=float(round(defaults[col], 3)), step=0.005)
    if "PageValues" in dropped:
        values["PageValues"] = float(defaults["PageValues"])
        st.slider(LABELS["PageValues"] + " (not used by this model)", 0.0, ranges["PageValues"][1],
                  value=float(defaults["PageValues"]), disabled=True)
    else:
        values["PageValues"] = st.slider(LABELS["PageValues"], min_value=0.0, max_value=ranges["PageValues"][1],
                                         value=float(defaults["PageValues"]), step=1.0)
    values["SpecialDay"] = st.select_slider(LABELS["SpecialDay"], options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

    st.subheader("Visitor and context")
    c1, c2 = st.columns(2)
    values["Month"] = c1.selectbox("Month", cats["Month"], index=cats["Month"].index("Nov") if "Nov" in cats["Month"] else 0)
    values["VisitorType"] = c2.selectbox("Visitor type", cats["VisitorType"])
    values["Weekend"] = int(c1.checkbox("Weekend", value=False))
    values["OperatingSystems"] = c2.selectbox("Operating system (ID)", cats["OperatingSystems"],
                                              index=cats["OperatingSystems"].index(2) if 2 in cats["OperatingSystems"] else 0)
    values["Browser"] = c1.selectbox("Browser (ID)", cats["Browser"],
                                     index=cats["Browser"].index(2) if 2 in cats["Browser"] else 0)
    values["Region"] = c2.selectbox("Region (ID)", cats["Region"],
                                    index=cats["Region"].index(1) if 1 in cats["Region"] else 0)
    values["TrafficType"] = c1.selectbox("Traffic type (ID)", cats["TrafficType"],
                                         index=cats["TrafficType"].index(2) if 2 in cats["TrafficType"] else 0)

# ---------- prediction ----------
row = pd.DataFrame([values])[bundle["columns"]]
if key == "early":
    row = row.drop(columns=dropped)

prob = float(model.predict_proba(row)[0, 1])
will_buy = prob >= threshold

st.divider()
st.subheader("Prediction")
m1, m2 = st.columns([1, 2])
m1.metric("Purchase probability", f"{prob:.1%}")
with m2:
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Flag threshold: {threshold:.0%}")

if will_buy:
    st.success("**Likely buyer.** Consider an immediate nudge such as a discount banner or free shipping to close the sale.")
else:
    st.warning("**Likely browser.** Avoid spending incentive budget now; re-target later with ads or email.")
