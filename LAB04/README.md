# NYC Airbnb Price Prediction

DS605: Fundamentals of Machine Learning — Lab Assignment 4

This project builds a full machine learning workflow that predicts the nightly
price of an Airbnb listing in New York City, and wraps the final model in a
simple Streamlit app.

## Dataset

[Kaggle - New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
(`AB_NYC_2019.csv`). Not included in this repo — download it from Kaggle and
place it in the project folder before running the notebook.

## Project structure

```
├── DS605_Lab04_202618050_optimized.ipynb   # Full workflow: cleaning, models, tuning
├── app.py                                  # Streamlit app for price prediction
├── requirements.txt                        # Python packages needed
├── model/                                  # Created after running the notebook
│   ├── airbnb_price_model.pkl              # Saved pipeline (preprocessing + model)
│   └── model_metadata.pkl                  # Dropdown values and preprocessing limits used by the app
└── README.md
```

## How to run

**1. Set up the environment**
```
pip install -r requirements.txt
```

**2. Run the notebook**

Open `DS605_Lab04_202618050_optimized.ipynb` and run it top to bottom with
`AB_NYC_2019.csv` in the same folder. This cleans the data, trains and
compares models, tunes the best one, and saves the final pipeline into the
`model/` folder.

**3. Run the app**
```
streamlit run app.py
```
Fill in the listing details in the browser tab that opens, and click
**Predict Price**.

## What the notebook does

**Data cleaning and feature engineering**
- Removed listings with a price of $0 and the top 1% most expensive listings (outliers)
- Filled missing `reviews_per_month` with 0 (no reviews yet)
- Capped `minimum_nights` at its 99th percentile to remove unrealistic values
- Turned `last_review` into a new feature, `days_since_last_review`
- Grouped rare neighbourhoods (outside the top 20 by listing count) into `"Other"`
  to keep the model simpler and faster to train

**Models compared**
- Linear Regression and Polynomial Regression (simple baselines)
- Random Forest
- Gradient Boosting
- XGBoost

All models predict `log(price)` internally (since price is right-skewed) and
convert back to normal dollars automatically. Random Forest, Gradient
Boosting, and XGBoost are tuned with `RandomizedSearchCV`, and whichever has
the lowest test RMSE is saved as the final model.

**Final model:** *XG Boost*

## Deployed app

[https://airbnb-202618050.streamlit.app/]

