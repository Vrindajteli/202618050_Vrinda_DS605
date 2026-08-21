# Assignment title : 
Scikit-learn: Data Preprocessing and Model Performance Evaluation
# Student ID : 
202618050
# Dataset link : 
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
# Preprocessing choices :
* **Feature Removal for High Missingness Rate**: Removed the feature `company` since its missing value rate was too high (>94%).
* **Target Leakage Protection**: Removed `reservation_status` and `reservation_status_date` because those features revealed directly the result of the booking, thus, preventing data leakage during training phase.
* **Outliers Removal**: Removed extreme outliers in numerical features where $ (\text{adr} < 0)$ or $(\text{adr} \ge 1000)$, unrealistic number of people in the hotel room $(\text{adults} > 10)$.
* **Filled missing values** using `KNNImputer(n_neighbors=5)`, meaning that it takes weighted averages from k nearest neighbors.
* **Two scaling techniques** were considered: **Pipeline A**: `StandardScaler`, z-score normalization and **Pipeline B**: `MinMaxScaler`, scales data from 0 to 1.
* **Categorical Variables**: Used `SimpleImputer(strategy="most_frequent")` to fill the missing values in categorical features and encode those features using `OneHotEncoder(handle_unknown="ignore")`.
* **Train-test-split & Data Leaking**: Used `train_test_split(test_size=0.2, stratify=y, random_state=42)` and wrapped all the preprocessing into `ColumnTransformer` and `Pipeline` so that standardizers and imputers are fit only on training data.
# Final observations :
1. Best Overall Result
The DecisionTreeClassifier with Pipeline A (StandardScaler) has the best overall results – the highest testing accuracy (~84.15%) and F1-score (~0.7873). The recall (~0.7912) of the DecisionTreeClassifier with Pipeline A (StandardScaler) is substantially higher than in the case of Logistic Regression, capturing true positives much better.
2. Impact of Scaling on Logistic RegressionStandardScaler slightly positively affects Logistic Regression in comparison with MinMaxScaler. In logistic regression, the model uses coefficients that are optimized by the process of gradient descent. It means that by standardizing unbounded or skewed features such as lead_time and adr, one makes Logistic Regression better in converging than in squashing all features into a bounded $[0, 1]$ range.
3. Impact of Scaling on Decision TreesNo, scaling makes no substantial difference for the Decision Tree. The performance does not vary at all: the test accuracy is the same for Pipeline A (84.15%) and Pipeline B (84.12%). Tree-based algorithms are used to find the best single variable split point $X_j \le t$, which makes them indifferent to monotonic transformation.
4. The Decision Tree has an advantage in both test accuracy (~84.15% vs. ~80.45%) and F1-score (~0.7873 vs. ~0.7102) because it captures nonlinear feature            interactions (interaction between deposit_type, lead_time and country).
5. The unconstrained DecisionTreeClassifier can fit training data perfectly (~99.98% accuracy) but performs poorly (~84.15%) on testing data, which means             severe overfitting, that needs to be controlled by hyperparameter tuning (e.g., max_depth or min_samples_leaf).
6. Logistic Regression is very stable, which is shown by very low variance; moreover, its performance is quite similar in terms of accuracy (training                 ~81.24%, testing ~80.45%).Error Distribution
7. It is clear from confusion matrices that Logistic Regression generates a lot of False Negatives (misses a lot of true positives), that is why recall (~0.6510) is low.
