import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Airbnb Price Predictor", layout="centered")

MODEL_DIR = Path(__file__).resolve().parent / "model"
DATA_PATH = Path(__file__).resolve().parent / "AB_NYC_2019.csv"

# ---------- Load the saved pipeline and metadata ----------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "airbnb_price_model.pkl")
    metadata = joblib.load(MODEL_DIR / "model_metadata.pkl")
    return model, metadata

@st.cache_data
def load_centroids():
    df = pd.read_csv(DATA_PATH)
    # Average latitude and longitude per neighbourhood
    centroids = df.groupby("neighbourhood")[["latitude", "longitude"]].mean().to_dict("index")
    default_lat = df["latitude"].mean()
    default_lon = df["longitude"].mean()
    return centroids, default_lat, default_lon

model, metadata = load_model()
centroids_lookup, default_lat, default_lon = load_centroids()

st.title("NYC Airbnb Price Predictor")
st.write(
    "Fill in the listing details below to get an estimated nightly price. "
)

# ---------- Input form ----------
with st.form("listing_form"):
    col1, col2 = st.columns(2)

    with col1:
        neighbourhood_group = st.selectbox(
            "Neighbourhood Group", metadata["neighbourhood_groups"]
        )
        neighbourhood = st.selectbox(
            "Neighbourhood", metadata["neighbourhoods"],
            help="Only the most common neighbourhoods are listed. Pick 'Other' if yours isn't here."
        )
        room_type = st.selectbox("Room Type", metadata["room_types"])
        minimum_nights = st.number_input(
            "Minimum Nights", min_value=1, max_value=int(metadata["min_nights_cap"]), value=2
        )

    with col2:
        availability_365 = st.slider("Available Days per Year", 0, 365, 180)
        calculated_host_listings_count = st.number_input(
            "Host's Total Listings", min_value=1, value=1
        )
        has_reviews = st.checkbox(
            "This listing has reviews", value=True, key="has_reviews"
        )

        if has_reviews:
            number_of_reviews = st.number_input(
                "Number of Reviews", min_value=1, value=10, step=1, format="%d", key="number_of_reviews"
            )
            reviews_per_month = st.number_input(
                "Reviews per Month", min_value=0.0, value=1.0, step=0.1, key="reviews_per_month"
            )
            days_since_last_review = st.number_input(
                "Days Since Last Review", min_value=0, value=30, step=1, format="%d", key="days_since_last_review"
            )
        else:
            number_of_reviews = 0
            reviews_per_month = 0.0
            days_since_last_review = metadata["no_review_fill_value"]

    submitted = st.form_submit_button("Predict Price")

# ---------- Prediction ----------
if submitted:
    # Dynamically retrieve coordinates for the selected neighbourhood
    coords = centroids_lookup.get(neighbourhood, {})
    lat = coords.get("latitude", default_lat)
    lon = coords.get("longitude", default_lon)

    input_data = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": lat,
        "longitude": lon,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "days_since_last_review": days_since_last_review
    }])

    predicted_price = model.predict(input_data)[0]
    predicted_price = max(predicted_price, 0)

    st.success(f"### Estimated Price: ${predicted_price:.2f} per night")

    if predicted_price >= metadata["price_limit"]:
        st.info(
            "This estimate is near the top of what the model was trained on, "
            "so treat it as a lower bound rather than an exact number."
        )
