import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ShopperIQ | Purchase Intention",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME / CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #101827;
    --navy-soft: #17243A;
    --blue: #2563EB;
    --teal: #14B8A6;
    --green: #16A34A;
    --amber: #D97706;
    --ink: #172033;
    --muted: #64748B;
    --line: #E2E8F0;
    --surface: #FFFFFF;
    --canvas: #F4F7FB;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--canvas);
    color: var(--ink);
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101827 0%, #17243A 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #E5EDF8 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.045);
    border-radius: 10px;
    padding: 8px 10px;
    margin: 4px 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(20,184,166,0.16);
}
.sidebar-brand {
    padding: 12px 4px 18px 4px;
}
.sidebar-brand-title {
    color: white;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.sidebar-brand-sub {
    color: #9FB0C8;
    font-size: 0.72rem;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-top: 3px;
}

.hero {
    background: linear-gradient(115deg, #101827 0%, #1D3557 62%, #155E75 100%);
    border-radius: 22px;
    padding: 32px 34px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 14px 35px rgba(16, 24, 39, 0.13);
}
.hero-eyebrow {
    color: #8FE5D9;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero h1 {
    color: white !important;
    font-size: clamp(1.8rem, 4vw, 2.65rem) !important;
    font-weight: 800 !important;
    letter-spacing: -1.2px;
    line-height: 1.15;
    margin: 0 0 10px 0;
}
.hero p {
    color: #D5E1EF !important;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 780px;
    margin: 0;
}
.page-heading {
    color: #14213A;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.7px;
    margin: 0 0 5px 0;
}
.page-subheading {
    color: #64748B;
    font-size: 0.98rem;
    margin-bottom: 22px;
}
.section-heading {
    color: #172033;
    font-size: 1.12rem;
    font-weight: 750;
    margin: 5px 0 3px 0;
}
.section-caption {
    color: #718096;
    font-size: 0.86rem;
    margin-bottom: 14px;
}
.info-card {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 16px;
    padding: 19px 20px;
    min-height: 116px;
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.035);
}
.info-card .label {
    color: #718096;
    font-size: 0.78rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}
.info-card .value {
    color: #14213A;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 7px;
}
.info-card .detail {
    color: #8492A6;
    font-size: 0.8rem;
    margin-top: 2px;
}
.panel {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 17px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
}
.result-panel {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 9px 28px rgba(15, 23, 42, 0.06);
}
.result-status {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.4px;
    margin: 5px 0 4px 0;
}
.result-note {
    color: #64748B;
    font-size: 0.9rem;
    line-height: 1.55;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 14px;
    padding: 15px 17px;
    box-shadow: 0 4px 14px rgba(15,23,42,0.035);
}
div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 650;
}
div[data-testid="stMetricValue"] {
    color: #14213A;
    font-weight: 800;
}
.stButton > button, .stFormSubmitButton > button {
    border: 0 !important;
    border-radius: 11px !important;
    min-height: 2.8rem;
    font-weight: 750 !important;
    background: linear-gradient(100deg, #2563EB 0%, #0D9488 100%) !important;
    color: white !important;
    box-shadow: 0 6px 15px rgba(37,99,235,0.18);
    transition: all 0.2s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}
div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 17px;
    padding: 22px 22px 14px 22px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.035);
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 9px;
    border-color: #D9E2EF;
    background: #FBFCFE;
}
.stAlert {
    border-radius: 12px;
}
hr {
    border-color: #E2E8F0 !important;
}
.footer {
    color: #8A97A9;
    font-size: 0.78rem;
    text-align: center;
    padding: 22px 0 4px 0;
}
.small-pill {
    display: inline-block;
    background: rgba(20,184,166,0.13);
    color: #0F766E;
    border: 1px solid rgba(20,184,166,0.2);
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.4px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
MODEL_PATH = Path(__file__).parent / "shopper_model_bundle.joblib"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return cloudpickle.load(file)


try:
    bundle = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Place shopper_model_bundle.joblib "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# =========================================================
# SHARED HELPERS
# =========================================================
def page_header(title, subtitle):
    st.markdown(f'<div class="page-heading">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subheading">{subtitle}</div>', unsafe_allow_html=True)


def info_card(label, value, detail=""):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, caption=None):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def make_gauge(probability, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {"size": 43, "color": "#14213A"}
            },
            title={
                "text": "Estimated purchase probability",
                "font": {"size": 15, "color": "#64748B"}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": "%",
                    "tickfont": {"color": "#718096", "size": 10}
                },
                "bar": {"color": "#14B8A6", "thickness": 0.26},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100], "color": "#EAF0F6"}
                ],
                "threshold": {
                    "line": {"color": "#2563EB", "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100
                }
            }
        )
    )
    fig.update_layout(
        height=285,
        margin=dict(l=22, r=22, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif"}
    )
    return fig


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">ShopperIQ</div>
            <div class="sidebar-brand-sub">Purchase Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Home", "Purchase Prediction", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**PROJECT**")
    st.caption("Online Shoppers Purchasing Intention")
    st.markdown(
        '<span class="small-pill">MACHINE LEARNING</span>',
        unsafe_allow_html=True
    )


# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Machine learning · Customer analytics</div>
            <h1>Understand shopping behaviour.<br>Predict purchase intention.</h1>
            <p>
                Estimate whether an online visitor is likely to make a purchase
                using browsing activity, session characteristics and visitor information.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_title("Platform overview", "A machine-learning application for analysing online shopping sessions.")

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Prediction task", "Classification", "Purchase or no-purchase outcome")
    with c2:
        info_card("Available features", str(len(bundle.get("columns", []))), "Visitor and session attributes")
    with c3:
        info_card("Model options", "2", "Full and reduced-feature models")

    st.write("")
    section_title("How it works", "Three simple steps from visitor information to model output.")

    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown(
            '<div class="panel"><div style="font-size:1.5rem;">1</div>'
            '<b>Enter visitor details</b><p style="color:#64748B;font-size:.88rem;">'
            'Provide browsing activity, session metrics and visitor characteristics.</p></div>',
            unsafe_allow_html=True
        )
    with step2:
        st.markdown(
            '<div class="panel"><div style="font-size:1.5rem;">2</div>'
            '<b>Run the prediction</b><p style="color:#64748B;font-size:.88rem;">'
            'The selected trained pipeline processes the input and estimates purchase probability.</p></div>',
            unsafe_allow_html=True
        )
    with step3:
        st.markdown(
            '<div class="panel"><div style="font-size:1.5rem;">3</div>'
            '<b>Review the result</b><p style="color:#64748B;font-size:.88rem;">'
            'See the predicted class, probability and model decision threshold.</p></div>',
            unsafe_allow_html=True
        )

    st.write("")
    st.info("Choose **Purchase Prediction** from the sidebar to analyse a visitor session.")


# =========================================================
# PURCHASE PREDICTION PAGE
# =========================================================
elif page == "Purchase Prediction":
    page_header(
        "Purchase intention prediction",
        "Enter the visitor's session information and generate a prediction using your trained model."
    )

    model_choice = st.radio(
        "Choose prediction model",
        ["Full Model", "Reduced Feature Model"],
        horizontal=True,
        help="The reduced-feature model uses the feature subset saved in your model bundle."
    )

    if model_choice == "Full Model":
        model_info = bundle["full"]
    else:
        model_info = bundle["early"]

    model = model_info["model"]
    threshold = float(model_info["threshold"])
    features = list(model_info["features"])

    defaults = bundle.get("defaults", bundle.get("numeric_defaults", {}))
    categories = bundle.get("categories", bundle.get("categorical_values", {}))
    numeric_ranges = bundle.get("numeric_ranges", {})

    descriptions = {
        "Administrative": "Number of administrative pages visited during the session.",
        "Administrative_Duration": "Time spent on administrative pages.",
        "Informational": "Number of informational pages visited during the session.",
        "Informational_Duration": "Time spent on informational pages.",
        "ProductRelated": "Number of product-related pages visited.",
        "ProductRelated_Duration": "Time spent on product-related pages.",
        "BounceRates": "Average bounce rate of pages visited.",
        "ExitRates": "Average exit rate of pages visited.",
        "PageValues": "Average page value associated with the session.",
        "SpecialDay": "Closeness of the visit to a special day.",
        "Month": "Month in which the session occurred.",
        "OperatingSystems": "Operating system category.",
        "Browser": "Browser category.",
        "Region": "Geographical region category.",
        "TrafficType": "Traffic source category.",
        "VisitorType": "Visitor classification.",
        "Weekend": "Whether the visit occurred on a weekend."
    }

    st.markdown(
        f'<span class="small-pill">SELECTED MODEL: {model_info.get("name", model_choice)}</span>',
        unsafe_allow_html=True
    )
    st.caption(
        f"{len(features)} model input features · Decision threshold: {threshold:.3f}"
    )

    user_inputs = {}

    with st.form("prediction_form", clear_on_submit=False):
        section_title("01 · Visitor behaviour", "Browsing activity and time spent on different page types.")

        behaviour = [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration"
        ]

        cols = st.columns(2)
        for i, feature in enumerate(behaviour):
            if feature not in features:
                continue

            with cols[i % 2]:
                default = float(defaults.get(feature, 0) or 0)
                if feature in numeric_ranges:
                    low, high = numeric_ranges[feature]
                    low, high = float(low), float(high)
                    if high <= low:
                        high = low + 1
                    default = min(max(default, low), high)
                    user_inputs[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=low,
                        max_value=high,
                        value=default,
                        help=descriptions.get(feature, "")
                    )
                else:
                    user_inputs[feature] = st.number_input(
                        feature.replace("_", " "),
                        min_value=0.0,
                        value=max(0.0, default),
                        help=descriptions.get(feature, "")
                    )

        st.markdown("---")
        section_title("02 · Session characteristics", "Session-level engagement and page-value measures.")

        session_features = ["BounceRates", "ExitRates", "PageValues", "SpecialDay"]
        cols = st.columns(2)

        for i, feature in enumerate(session_features):
            if feature not in features:
                continue

            with cols[i % 2]:
                default = float(defaults.get(feature, 0) or 0)
                if feature in numeric_ranges:
                    low, high = numeric_ranges[feature]
                    low, high = float(low), float(high)
                    if high <= low:
                        high = low + 1
                    default = min(max(default, low), high)
                    user_inputs[feature] = st.number_input(
                        feature,
                        min_value=low,
                        max_value=high,
                        value=default,
                        help=descriptions.get(feature, "")
                    )
                else:
                    user_inputs[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=max(0.0, default),
                        help=descriptions.get(feature, "")
                    )

        st.markdown("---")
        section_title("03 · Visitor and traffic information", "Select the visitor, device and traffic characteristics.")

        categorical_features = [
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend"
        ]

        cols = st.columns(2)
        for i, feature in enumerate(categorical_features):
            if feature not in features:
                continue

            with cols[i % 2]:
                default = defaults.get(feature, None)

                if feature in categories and len(categories[feature]) > 0:
                    options = list(categories[feature])
                    selected_index = 0
                    for j, option in enumerate(options):
                        if option == default:
                            selected_index = j
                            break

                    user_inputs[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options,
                        index=selected_index,
                        help=descriptions.get(feature, "")
                    )
                elif feature == "Weekend":
                    weekend_default = int(default) if default in [0, 1] else 0
                    user_inputs[feature] = st.selectbox(
                        "Weekend",
                        [0, 1],
                        index=weekend_default,
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        help=descriptions.get(feature, "")
                    )
                else:
                    # Fallback for bundles that do not store category options.
                    user_inputs[feature] = st.text_input(
                        feature.replace("_", " "),
                        value=str(default if default is not None else ""),
                        help=descriptions.get(feature, "")
                    )

        st.markdown("")
        submitted = st.form_submit_button(
            "Predict purchase intention",
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREDICTION OUTPUT
    # -----------------------------------------------------
    if submitted:
        try:
            input_df = pd.DataFrame([user_inputs]).reindex(columns=features)

            # Convert fallback text fields to numeric only when the saved defaults
            # or category metadata indicate numeric values.
            for col in input_df.columns:
                if input_df[col].dtype == object and col not in categories:
                    if col in defaults and isinstance(defaults[col], (int, float, np.number)):
                        input_df[col] = pd.to_numeric(input_df[col], errors="raise")

            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= threshold)

            st.markdown("")
            section_title("Prediction result", "The result below is generated by the selected trained model.")

            result_col, gauge_col = st.columns([1.05, 1.15], gap="large")

            with result_col:
                st.markdown('<div class="result-panel">', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(
                        '<div class="small-pill">MODEL PREDICTION</div>'
                        '<div class="result-status" style="color:#15803D;">Purchase likely</div>'
                        '<div class="result-note">The estimated purchase probability meets or exceeds the selected model threshold.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="small-pill">MODEL PREDICTION</div>'
                        '<div class="result-status" style="color:#B45309;">No purchase predicted</div>'
                        '<div class="result-note">The estimated purchase probability is below the selected model threshold.</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                metric1, metric2 = st.columns(2)
                metric1.metric("Purchase probability", f"{probability * 100:.2f}%")
                metric2.metric("Decision threshold", f"{threshold * 100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            with gauge_col:
                st.markdown('<div class="result-panel">', unsafe_allow_html=True)
                st.plotly_chart(
                    make_gauge(probability, threshold),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
                st.caption("Blue marker = selected decision threshold.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown("**Prediction details**")
            detail1, detail2, detail3 = st.columns(3)
            with detail1:
                info_card("Predicted class", "Purchase" if prediction == 1 else "No purchase", "Threshold-based classification")
            with detail2:
                info_card("Model selected", "Full" if model_choice == "Full Model" else "Reduced", model_info.get("name", model_choice))
            with detail3:
                info_card("Input features", str(len(features)), "Features supplied to this model")

            with st.expander("Review submitted visitor information"):
                display_df = input_df.copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.caption(
                "Note: The probability is the model's estimated score. The final class is determined "
                "by comparing that score with the threshold selected during model development."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info(
                "Please check that all selected values match the feature types and categories "
                "used when the model was trained."
            )


# =========================================================
# ABOUT PAGE
# =========================================================
elif page == "About":
    page_header(
        "About this project",
        "An overview of the objective, model options and prediction approach."
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Project overview</div>
            <h1>Online Shoppers Purchasing Intention</h1>
            <p>
                This machine-learning project analyses online shopping session information
                to estimate whether a visitor is likely to generate a purchase.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    section_title("Project objective")
    st.markdown(
        """
        <div class="panel">
            The objective is to use browsing behaviour, visitor characteristics and
            session information to predict the Revenue outcome. The application
            presents the estimated purchase probability and the classification
            produced using the threshold stored with the selected model.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    section_title("Models available in this application")
    full_col, reduced_col = st.columns(2)

    with full_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="small-pill">FULL MODEL</div>
                <h3 style="color:#172033;margin-bottom:5px;">{bundle["full"].get("name", "Full Model")}</h3>
                <p style="color:#64748B;font-size:.9rem;">Uses the full feature set saved in the model bundle.</p>
                <b>Features:</b> {len(bundle["full"].get("features", []))}<br>
                <b>Threshold:</b> {float(bundle["full"]["threshold"]):.4f}
            </div>
            """,
            unsafe_allow_html=True
        )

    with reduced_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="small-pill">REDUCED FEATURE MODEL</div>
                <h3 style="color:#172033;margin-bottom:5px;">{bundle["early"].get("name", "Reduced Feature Model")}</h3>
                <p style="color:#64748B;font-size:.9rem;">Uses the reduced feature set saved in the model bundle.</p>
                <b>Features:</b> {len(bundle["early"].get("features", []))}<br>
                <b>Threshold:</b> {float(bundle["early"]["threshold"]):.4f}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    section_title("How the prediction is made")
    st.markdown(
        """
        <div class="panel">
            1. Visitor and session inputs are arranged in the feature order expected by the selected model.<br><br>
            2. The saved preprocessing and classification pipeline generates a purchase probability.<br><br>
            3. The probability is compared with the model-specific threshold stored in the bundle.<br><br>
            4. The application displays the resulting predicted class and probability.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "The reduced-feature model is a feature-reduced comparison model. "
        "It should not automatically be interpreted as a real-time prediction model."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        ShopperIQ · Online Shoppers Purchasing Intention · Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
