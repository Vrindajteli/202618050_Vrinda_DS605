from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Purchase Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Refined, subtle color theme (Enterprise SaaS / Analytics aesthetic)
st.markdown(
    """
    <style>
    /* Global font adjustments & header styling */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }
    
    /* Top accent line */
    .title-accent {
        width: 48px;
        height: 4px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        border-radius: 2px;
        margin-bottom: 1rem;
    }

    /* Status Pills */
    .pill {
        display: inline-block;
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.22);
        color: #1e40af;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
        letter-spacing: 0.01em;
    }
    
    /* Input Container Cards */
    .section-card {
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(248, 250, 252, 0.5);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* Prediction Card - Buyer */
    .prediction-card-buyer {
        background: rgba(16, 185, 129, 0.04);
        border: 1.5px solid #10b981;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
    }
    .badge-buyer {
        background-color: #059669;
        color: #ffffff;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
    }

    /* Prediction Card - Browser */
    .prediction-card-browser {
        background: rgba(245, 158, 11, 0.04);
        border: 1.5px solid #d97706;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);
    }
    .badge-browser {
        background-color: #d97706;
        color: #ffffff;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
    }

    /* Metric Probability Display */
    .big-prob {
        font-size: 2.85rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0.4rem 0 0.2rem 0;
        letter-spacing: -0.02em;
    }
    .prob-caption {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    /* Recommendation Callout */
    .recommendation-buyer {
        background: rgba(16, 185, 129, 0.06);
        border-left: 3.5px solid #059669;
        border-radius: 0 6px 6px 0;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        font-size: 0.88rem;
    }
    .recommendation-browser {
        background: rgba(245, 158, 11, 0.06);
        border-left: 3.5px solid #d97706;
        border-radius: 0 6px 6px 0;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        font-size: 0.88rem;
    }

    /* Signal item styling */
    .signal-item {
        font-size: 0.86rem;
        padding: 0.3rem 0;
        color: #334155;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model & Bundle Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    model_path = Path(__file__).resolve().parent / "shopper_model.joblib"
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent / "shopper_model_bundle.joblib"
    return joblib.load(model_path)


bundle = load_bundle()
defaults = bundle["defaults"]
ranges = bundle["numeric_ranges"]
cats = bundle["categories"]
columns = bundle["columns"]


# -----------------------------------------------------------------------------
# Sidebar: Model Configuration & Quick Presets
# -----------------------------------------------------------------------------
st.sidebar.title("Model Settings")

mode = st.sidebar.radio(
    "Evaluation Scope",
    ["Full Model (Post-Session)", "Early-Session Model (Real-Time)"],
    help=(
        "Full Model: Uses all session metrics including PageValues, BounceRates, and ExitRates.\n"
        "Early Model: Excludes session-end metrics to predict in real-time mid-session without data leakage."
    ),
)
key = "full" if "Full" in mode else "early"
model_info = bundle[key]
model, threshold = model_info["model"], float(model_info["threshold"])
dropped = bundle["early"].get("dropped", ["PageValues", "BounceRates", "ExitRates"]) if key == "early" else []

# Model metadata display in sidebar
st.sidebar.markdown(f"**Selected Model:** `{model_info.get('name', 'XGBoost')}`")
st.sidebar.markdown(f"**Decision Threshold:** `{threshold:.1%}`")
st.sidebar.caption("Sessions scoring at or above this threshold are classified as likely buyers.")

st.sidebar.divider()
st.sidebar.subheader("Quick Presets")
st.sidebar.caption("Load pre-configured visitor behaviors:")

# Preset definitions without emojis
PRESETS = {
    "High-Intent Buyer": {
        "Administrative": 3,
        "Administrative_Duration": 85.0,
        "Informational": 1,
        "Informational_Duration": 40.0,
        "ProductRelated": 45,
        "ProductRelated_Duration": 1650.0,
        "BounceRates": 0.002,
        "ExitRates": 0.015,
        "PageValues": 28.5,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "VisitorType": "Returning_Visitor",
        "Weekend": 1,
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
    },
    "Casual Browser (Bounce)": {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 2,
        "ProductRelated_Duration": 32.0,
        "BounceRates": 0.12,
        "ExitRates": 0.15,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "May",
        "VisitorType": "Returning_Visitor",
        "Weekend": 0,
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
    },
    "Typical Median Session": {
        "Administrative": int(defaults["Administrative"]),
        "Administrative_Duration": float(defaults["Administrative_Duration"]),
        "Informational": int(defaults["Informational"]),
        "Informational_Duration": float(defaults["Informational_Duration"]),
        "ProductRelated": int(defaults["ProductRelated"]),
        "ProductRelated_Duration": float(defaults["ProductRelated_Duration"]),
        "BounceRates": float(defaults["BounceRates"]),
        "ExitRates": float(defaults["ExitRates"]),
        "PageValues": float(defaults["PageValues"]),
        "SpecialDay": float(defaults["SpecialDay"]),
        "Month": "Nov" if "Nov" in cats["Month"] else cats["Month"][0],
        "VisitorType": "Returning_Visitor" if "Returning_Visitor" in cats["VisitorType"] else cats["VisitorType"][0],
        "Weekend": 0,
        "OperatingSystems": 2 if 2 in cats["OperatingSystems"] else cats["OperatingSystems"][0],
        "Browser": 2 if 2 in cats["Browser"] else cats["Browser"][0],
        "Region": 3 if 3 in cats["Region"] else cats["Region"][0],
        "TrafficType": 2 if 2 in cats["TrafficType"] else cats["TrafficType"][0],
    },
}

col_p1, col_p2, col_p3 = st.sidebar.columns(3)
if col_p1.button("High Intent", use_container_width=True):
    for k, v in PRESETS["High-Intent Buyer"].items():
        st.session_state[f"val_{k}"] = v
if col_p2.button("Casual", use_container_width=True):
    for k, v in PRESETS["Casual Browser (Bounce)"].items():
        st.session_state[f"val_{k}"] = v
if col_p3.button("Median", use_container_width=True):
    for k, v in PRESETS["Typical Median Session"].items():
        st.session_state[f"val_{k}"] = v

# Helper to retrieve state or default
def get_val(key_name, default_val):
    state_key = f"val_{key_name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_val
    return st.session_state[state_key]


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="title-accent"></div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Shopper Purchase Intention Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predict whether an online session ends in a purchase based on '
    'browsing behavior, engagement quality, and visitor context.</div>',
    unsafe_allow_html=True,
)

# Metadata pills
st.markdown(
    f"""
    <div>
        <span class="pill">Model: <b>{model_info.get('name', 'XGBoost')}</b></span>
        <span class="pill">Target: <b>Revenue</b> (~15.5% baseline)</span>
        <span class="pill">Decision Threshold: <b>{threshold:.1%}</b></span>
        <span class="pill">Evaluation Mode: <b>{'Real-Time Early Session' if key == 'early' else 'Post-Session Full'}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------------------------------------------------------
# Two-Column Layout: Inputs (Left) and Live Prediction (Right)
# -----------------------------------------------------------------------------
input_col, pred_col = st.columns([1.4, 1.0], gap="large")

values = {}

with input_col:
    tab_browse, tab_quality, tab_visitor = st.tabs([
        "Browsing Behavior",
        "Session Quality & Timing",
        "Visitor & Context",
    ])

    # --- TAB 1: Browsing Behavior ---
    with tab_browse:
        st.caption("Page counts and time spent (in seconds) across site sections.")

        # Product Related
        st.markdown("##### Product Pages *(Primary shopping activity)*")
        p_c1, p_c2 = st.columns(2)
        prod_count = p_c1.slider(
            "Product pages viewed",
            min_value=0,
            max_value=int(ranges["ProductRelated"][1]),
            value=int(get_val("ProductRelated", defaults["ProductRelated"])),
            step=1,
            key="slider_ProductRelated",
        )
        values["ProductRelated"] = prod_count
        st.session_state["val_ProductRelated"] = prod_count

        prod_dur = p_c2.slider(
            "Time on product pages (sec)",
            min_value=0.0,
            max_value=float(ranges["ProductRelated_Duration"][1]),
            value=float(round(get_val("ProductRelated_Duration", defaults["ProductRelated_Duration"]), 1)),
            step=10.0,
            help=f"~{int(get_val('ProductRelated_Duration', defaults['ProductRelated_Duration']) / 60)} minutes",
            key="slider_ProductRelated_Duration",
        )
        values["ProductRelated_Duration"] = prod_dur
        st.session_state["val_ProductRelated_Duration"] = prod_dur

        st.divider()

        # Administrative
        st.markdown("##### Account & Administrative Pages")
        a_c1, a_c2 = st.columns(2)
        admin_count = a_c1.slider(
            "Admin / account pages viewed",
            min_value=0,
            max_value=int(ranges["Administrative"][1]),
            value=int(get_val("Administrative", defaults["Administrative"])),
            step=1,
            key="slider_Administrative",
        )
        values["Administrative"] = admin_count
        st.session_state["val_Administrative"] = admin_count

        admin_dur = a_c2.slider(
            "Time on admin pages (sec)",
            min_value=0.0,
            max_value=float(ranges["Administrative_Duration"][1]),
            value=float(round(get_val("Administrative_Duration", defaults["Administrative_Duration"]), 1)),
            step=10.0,
            key="slider_Administrative_Duration",
        )
        values["Administrative_Duration"] = admin_dur
        st.session_state["val_Administrative_Duration"] = admin_dur

        st.divider()

        # Informational
        st.markdown("##### Informational Pages *(FAQ, About, Policies)*")
        i_c1, i_c2 = st.columns(2)
        info_count = i_c1.slider(
            "Info pages viewed",
            min_value=0,
            max_value=int(ranges["Informational"][1]),
            value=int(get_val("Informational", defaults["Informational"])),
            step=1,
            key="slider_Informational",
        )
        values["Informational"] = info_count
        st.session_state["val_Informational"] = info_count

        info_dur = i_c2.slider(
            "Time on info pages (sec)",
            min_value=0.0,
            max_value=float(ranges["Informational_Duration"][1]),
            value=float(round(get_val("Informational_Duration", defaults["Informational_Duration"]), 1)),
            step=10.0,
            key="slider_Informational_Duration",
        )
        values["Informational_Duration"] = info_dur
        st.session_state["val_Informational_Duration"] = info_dur

    # --- TAB 2: Session Quality & Timing ---
    with tab_quality:
        if key == "early":
            st.info(
                "**Real-Time Guardrail Active:** PageValues, BounceRates, and ExitRates are session-end "
                "analytics. They are excluded from this early-session model to prevent data leakage during live sessions."
            )

        # Page Values
        st.markdown("##### Page Value")
        st.caption("Average value of pages visited prior to completing a transaction.")
        if "PageValues" in dropped:
            pv_val = float(defaults["PageValues"])
            values["PageValues"] = pv_val
            st.slider(
                "Page Value (Locked in Early Mode)",
                min_value=0.0,
                max_value=float(ranges["PageValues"][1]),
                value=pv_val,
                disabled=True,
                help="Excluded in Early-Session model to prevent leakage.",
            )
        else:
            pv_val = st.slider(
                "Page Value",
                min_value=0.0,
                max_value=float(ranges["PageValues"][1]),
                value=float(get_val("PageValues", defaults["PageValues"])),
                step=1.0,
                key="slider_PageValues",
            )
            values["PageValues"] = pv_val
            st.session_state["val_PageValues"] = pv_val

        st.divider()

        # Bounce & Exit Rates
        st.markdown("##### User Engagement & Drop-off")
        r_c1, r_c2 = st.columns(2)
        if "BounceRates" in dropped:
            br_val = float(defaults["BounceRates"])
            values["BounceRates"] = br_val
            r_c1.slider("Bounce Rate (Locked)", 0.0, 0.2, value=br_val, disabled=True)
        else:
            br_val = r_c1.slider(
                "Bounce Rate",
                min_value=0.0,
                max_value=0.2,
                value=float(round(get_val("BounceRates", defaults["BounceRates"]), 3)),
                step=0.005,
                key="slider_BounceRates",
            )
            values["BounceRates"] = br_val
            st.session_state["val_BounceRates"] = br_val

        if "ExitRates" in dropped:
            er_val = float(defaults["ExitRates"])
            values["ExitRates"] = er_val
            r_c2.slider("Exit Rate (Locked)", 0.0, 0.2, value=er_val, disabled=True)
        else:
            er_val = r_c2.slider(
                "Exit Rate",
                min_value=0.0,
                max_value=0.2,
                value=float(round(get_val("ExitRates", defaults["ExitRates"]), 3)),
                step=0.005,
                key="slider_ExitRates",
            )
            values["ExitRates"] = er_val
            st.session_state["val_ExitRates"] = er_val

        st.divider()

        # Special Day
        st.markdown("##### Holiday & Event Proximity")
        st.caption("Closeness of the session date to a major holiday or retail event.")
        sd_val = st.select_slider(
            "Special Day Closeness",
            options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            value=float(get_val("SpecialDay", defaults["SpecialDay"])),
            key="slider_SpecialDay",
        )
        values["SpecialDay"] = sd_val
        st.session_state["val_SpecialDay"] = sd_val

    # --- TAB 3: Visitor & Context ---
    with tab_visitor:
        st.markdown("##### Demographic & Calendar Attributes")
        v_c1, v_c2, v_c3 = st.columns(3)

        month_default = get_val("Month", "Nov" if "Nov" in cats["Month"] else cats["Month"][0])
        month_idx = cats["Month"].index(month_default) if month_default in cats["Month"] else 0
        v_month = v_c1.selectbox("Calendar Month", cats["Month"], index=month_idx, key="sb_Month")
        values["Month"] = v_month
        st.session_state["val_Month"] = v_month

        vt_default = get_val("VisitorType", "Returning_Visitor" if "Returning_Visitor" in cats["VisitorType"] else cats["VisitorType"][0])
        vt_idx = cats["VisitorType"].index(vt_default) if vt_default in cats["VisitorType"] else 0
        v_type = v_c2.selectbox("Visitor Profile", cats["VisitorType"], index=vt_idx, key="sb_VisitorType")
        values["VisitorType"] = v_type
        st.session_state["val_VisitorType"] = v_type

        wknd_val = v_c3.checkbox("Weekend Session", value=bool(get_val("Weekend", 0)), key="cb_Weekend")
        values["Weekend"] = int(wknd_val)
        st.session_state["val_Weekend"] = int(wknd_val)

        st.divider()

        st.markdown("##### Technical & Acquisition Channel")
        t_c1, t_c2 = st.columns(2)

        os_default = get_val("OperatingSystems", 2 if 2 in cats["OperatingSystems"] else cats["OperatingSystems"][0])
        os_idx = cats["OperatingSystems"].index(os_default) if os_default in cats["OperatingSystems"] else 0
        v_os = t_c1.selectbox("Operating System ID", cats["OperatingSystems"], index=os_idx, key="sb_OS")
        values["OperatingSystems"] = v_os
        st.session_state["val_OperatingSystems"] = v_os

        br_default = get_val("Browser", 2 if 2 in cats["Browser"] else cats["Browser"][0])
        br_idx = cats["Browser"].index(br_default) if br_default in cats["Browser"] else 0
        v_br = t_c2.selectbox("Browser ID", cats["Browser"], index=br_idx, key="sb_Browser")
        values["Browser"] = v_br
        st.session_state["val_Browser"] = v_br

        reg_default = get_val("Region", 3 if 3 in cats["Region"] else cats["Region"][0])
        reg_idx = cats["Region"].index(reg_default) if reg_default in cats["Region"] else 0
        v_reg = t_c1.selectbox("Geographic Region ID", cats["Region"], index=reg_idx, key="sb_Region")
        values["Region"] = v_reg
        st.session_state["val_Region"] = v_reg

        traf_default = get_val("TrafficType", 2 if 2 in cats["TrafficType"] else cats["TrafficType"][0])
        traf_idx = cats["TrafficType"].index(traf_default) if traf_default in cats["TrafficType"] else 0
        v_traf = t_c2.selectbox("Traffic Source Type ID", cats["TrafficType"], index=traf_idx, key="sb_Traffic")
        values["TrafficType"] = v_traf
        st.session_state["val_TrafficType"] = v_traf


# -----------------------------------------------------------------------------
# Live Model Prediction & Business Intelligence Panel (Right Column)
# -----------------------------------------------------------------------------
with pred_col:
    # Prepare inference vector matching training columns
    row = pd.DataFrame([values])[columns]
    if key == "early":
        row = row.drop(columns=dropped)

    # Compute probability
    prob = float(model.predict_proba(row)[0, 1])
    will_buy = prob >= threshold
    delta_from_thresh = prob - threshold

    # Prediction Hero Card
    card_class = "prediction-card-buyer" if will_buy else "prediction-card-browser"
    badge_class = "badge-buyer" if will_buy else "badge-browser"
    badge_label = "HIGH INTENT (BUYER)" if will_buy else "CASUAL VISITOR (BROWSER)"
    metric_color = "#059669" if will_buy else "#d97706"

    st.markdown(
        f"""
        <div class="{card_class}">
            <span class="{badge_class}">{badge_label}</span>
            <div style="margin-top: 10px; font-size: 0.85rem; font-weight: 500; color: #64748b;">
                Purchase Probability
            </div>
            <div class="big-prob" style="color: {metric_color};">
                {prob:.1%}
            </div>
            <div class="prob-caption">
                Decision Cutoff: <b>{threshold:.1%}</b> ({'+' if delta_from_thresh >= 0 else ''}{delta_from_thresh:.1%} delta)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar indicator
    st.progress(min(max(prob, 0.0), 1.0))

    # Actionable Business Recommendation
    if will_buy:
        st.markdown(
            """
            <div class="recommendation-buyer">
                <b>Recommended Action: Immediate Checkout Nudge</b><br>
                High conversion probability. Present an immediate checkout incentive 
                (e.g., free shipping threshold banner or limited-time checkout timer) 
                to accelerate purchase completion.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="recommendation-browser">
                <b>Recommended Action: Preserve Promotion Budget</b><br>
                Low conversion probability. Avoid expending margin-reducing discounts. 
                Focus on organic product recommendations or email remarketing list growth.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Key Session Highlights
    st.markdown("##### Key Session Signals")
    signals = []
    if key == "full" and values["PageValues"] > 0:
        signals.append(f"<b>Page Value:</b> <code>{values['PageValues']:.1f}</code> (Positive indicator)")
    if values["ProductRelated"] >= 30:
        signals.append(f"<b>Product Engagement:</b> <code>{values['ProductRelated']}</code> pages viewed")
    if values["Month"] == "Nov":
        signals.append("<b>Holiday Season:</b> November sessions feature elevated conversion rates")
    if values["VisitorType"] == "New_Visitor":
        signals.append("<b>New Visitor:</b> Statistically higher conversion likelihood on initial visit")
    if values["BounceRates"] > 0.05 and key == "full":
        signals.append(f"<b>Elevated Bounce Rate:</b> <code>{values['BounceRates']:.3f}</code>")

    if signals:
        for s in signals:
            st.markdown(f'<div class="signal-item">• {s}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-item">• Baseline browsing pattern with standard conversion likelihood.</div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("Model Architecture & Technical Details"):
        st.markdown(
            f"""
            - **Algorithm:** `{model_info.get('name', 'XGBoost')}`
            - **Cross-Validation:** 5-Fold Stratified CV on model-fitting set
            - **Threshold Tuning:** Calibrated on separate stratified validation set to maximize $F_1$ score
            - **Imbalance Handling:** SMOTENC / algorithmic class weighting
            - **Active Evaluation Mode:** `{'Early-Session (Real-Time)' if key == 'early' else 'Full-Session'}`
            """
        )
