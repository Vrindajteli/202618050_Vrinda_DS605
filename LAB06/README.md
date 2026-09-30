# Lab 6: Feature Extraction and Machine Learning with Image and Text Data
Student ID : 202618050

Raw images and raw text are converted into numerical features and classified.

- **Image task:** Asphalt Crack Dataset (Mendeley Data), 400 images (200 crack / 200 non-crack), crack vs non-crack.
- **Text task:** Email Spam Classification Dataset (Kaggle), 5,172 emails, spam vs ham.


## How to run

```bash
pip install -r requirements.txt
jupyter notebook lab6_notebook.ipynb    
```

## Part A: Image features and classification

**Pipeline:** read with OpenCV -> resize to 128x128 -> grayscale -> NumPy intensity statistics + Canny edge features -> one row per image with label -> train/test split (80/20, stratified) -> classifiers.

**Baseline features (17):** mean brightness, contrast (std), median, min, max, percentiles (5/25/75/95), IQR, skewness, kurtosis, entropy, dark-pixel ratio (<60), bright-pixel ratio (>200), Canny edge count and edge density (thresholds 100/200).

**Classifiers:** Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest. Metrics: accuracy, precision, recall, F1, confusion matrix, training time and prediction time (full table in `outputs/image_results_baseline.csv`).

## Part B: Text vectorization and spam classification

**Pipeline:** load CSV -> inspect class distribution -> basic cleaning (lowercase, remove URLs/HTML/non-letters) -> CountVectorizer and TF-IDF -> Naive Bayes, Logistic Regression, Linear SVM.

**Note on the dataset:** the Kaggle file is already vectorized (3,000 word-count columns). Each email was rebuilt as a bag-of-words document by repeating every word by its count, so the vectorizers could be applied. Word order is lost, so **n-grams were not used**.

**Count vs TF-IDF (F1, 3,000 features each):**

| Model | CountVectorizer | TF-IDF |
|---|---|---|
| Multinomial NB | 0.904 | 0.751 |
| Logistic Regression | 0.970 | 0.913 |
| Linear SVM | 0.967 | **0.974** |

Full table with precision, recall, timings and non-zero entries: `outputs/text_comparison.csv`.

## Part C: Improving the representation

### Image: blur + automatic Canny thresholds + extra features

Change: Gaussian blur before Canny, thresholds set from the median intensity (0.67x / 1.33x), plus grid-based edge statistics (4x4 blocks) and Laplacian variance (17 -> 20 features). Justification: fixed thresholds (100/200) react to asphalt texture; blur and adaptive thresholds keep the strong crack edges, and the grid features capture that cracks are localized.

| Features | Best model | # features | Test accuracy | Recall | F1 | 5-fold CV accuracy |
|---|---|---|---|---|---|---|
| Baseline | Logistic Regression | 17 | 0.950 | 0.975 | 0.951 | 0.9475 |
| Improved | SVM (RBF) | 20 | 0.950 | 0.975 | 0.951 | **0.9650** |

### Text: limiting the vocabulary (TF-IDF + Linear SVM)

| Variant | # features | Accuracy | F1 | Vectorize (s) | Train (s) | Predict (s) |
|---|---|---|---|---|---|---|
| Baseline (all words) | 3000 | 0.9845 | 0.9736 | 4.94 | 0.131 | 0.0013 |
| max_features=1000 | 1000 | 0.9797 | 0.9656 | 4.87 | 0.101 | 0.0009 |
| max_features=500 | 500 | 0.9700 | 0.9498 | 4.79 | 0.087 | 0.0008 |
| stop_words='english' | 2762 | 0.9807 | 0.9670 | 4.80 | 0.101 | 0.0010 |
| min_df=5, max_df=0.9 | 2958 | 0.9855 | 0.9752 | 4.85 | 0.088 | 0.0012 |
| stop + max_features=1000 | 1000 | 0.9739 | 0.9560 | 4.86 | 0.072 | 0.0008 |

## Observations

**Image**
- Simple intensity and edge statistics are enough for about 95% accuracy and 97.5% recall, so most cracks are caught without any deep learning.
- The improved features give the same test F1 (0.951) but a higher 5-fold CV accuracy (0.965 vs 0.9475) using only 3 extra features. The test set has just 80 images (one image is 1.25%), so the CV score is the more reliable comparison, and the gain is small.
- All feature extraction and training is very fast (training under 0.02 s), so the extra features cost almost nothing in computation.

**Text**
- The best baseline is TF-IDF + Linear SVM (F1 0.974). Neither representation wins overall: Count is better for Naive Bayes and Logistic Regression, TF-IDF is better for the linear SVM. TF-IDF weights hurt Multinomial NB, which expects raw counts.
- Limiting the vocabulary trades a little accuracy for a much smaller model. `max_features=1000` uses **3x fewer features**, loses about 0.8 points of F1 (0.9736 -> 0.9656) and trains about 23% faster. Going down to 500 features costs about 2.4 points of F1, so the curve bends there.
- `min_df=5, max_df=0.9` gave the highest F1 (0.9752) and the fastest training among the mild variants, but its gain over the baseline (0.16 points) is within noise for a single split.
- Vectorization (about 4.8 s) dominates the total time and is almost unchanged by `max_features`, because the time is spent tokenizing the text, not counting features. Fewer features mainly help training, prediction and memory.




