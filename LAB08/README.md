# Lab 8: Decision Trees and Support Vector Machines

Dataset: UCI Banknote Authentication (1,372 samples, 4 features, 2 classes). Split: 80/20 stratified (1,097 train / 275 test), `random_state=42`, used for every experiment.

## Files
- `Lab8_Decision_Tree_SVM.ipynb` - full solution with outputs and plots

## How to run
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook Lab8_Decision_Tree_SVM.ipynb
```

## What is implemented
- **Part A:** Gini and entropy, best-split search, recursive decision tree with stopping rules, prediction, tree plot with split rules, decision boundaries (variance vs skewness), criterion and depth study against `DecisionTreeClassifier`.
- **Part B:** Linear soft-margin SVM (hinge loss, subgradient descent) in NumPy, training-set scaling, margin and violator plot, study of C, learning rate and iterations against `SVC(kernel="linear")`.
- **Part C:** `make_moons` with linear, polynomial and RBF SVMs.
- **Part D:** All four features, same split, accuracy / precision / recall / F1 and timing for five models.

## Results

### Part A: Decision tree (two features)
- Manual and Scikit-learn trees give identical accuracy and 100% prediction agreement up to depth 5. At deeper settings agreement stays at 98.9% or higher; small differences come from tie-breaking between equal splits.
- Depth 5 (Gini): train 0.948, test 0.931 for both trees.
- Gini and entropy are very close. Best test accuracy is about 0.935 (Gini depth 4, entropy depth 5).
- Unlimited depth reaches 1.00 train accuracy but test accuracy stays around 0.92 to 0.93, so deeper trees only overfit.

### Part B: Linear SVM (two features)
| | w | b | Test acc |
|---|---|---|---|
| Manual | [-1.98, -1.24] | -0.231 | 0.876 |
| Scikit-learn | [-2.01, -1.26] | -0.221 | 0.876 |

- Margin violators (manual): 316. Scikit-learn support vectors: 317.
- **C:** smaller C gives a smaller `||w||` (1.25 at C=0.01 vs 2.39 at C=100), a wider margin and more violators (515 vs about 312). Test accuracy barely changes (0.887 to 0.876).
- **Learning rate:** 0.001 is too slow (final loss 0.4625). From 0.1 upward the loss reaches 0.2844.
- **Iterations:** loss stops improving after about 500 iterations (0.2846 at 500, 0.2844 at 1000 and 5000).
- Variance and skewness alone are not linearly separable, so accuracy is limited to about 0.88.

### Part C: Kernel SVM (moons)
| Kernel | Train acc | Test acc |
|---|---|---|
| Linear | 0.813 | 0.920 |
| Polynomial (degree 3) | 0.940 | 0.930 |
| RBF | 0.933 | 0.950 |

The moons cannot be split by a straight line. Polynomial and RBF kernels act like a higher-dimensional mapping and give curved boundaries; RBF follows the shape best.

### Part D: All four features
| Model | Train acc | Accuracy | Precision | Recall | F1 | Train time (s) | Predict time (s) |
|---|---|---|---|---|---|---|---|
| Manual Tree | 0.9945 | 0.9927 | 0.9839 | 1.0 | 0.9919 | 0.3235 | 0.0003 |
| Sklearn Tree | 0.9945 | 0.9927 | 0.9839 | 1.0 | 0.9919 | 0.0074 | 0.0004 |
| Manual Linear SVM | 0.9854 | 0.9709 | 0.9385 | 1.0 | 0.9683 | 0.0806 | 0.0001 |
| Sklearn Linear SVM | 0.9863 | 0.9745 | 0.9457 | 1.0 | 0.9721 | 0.0089 | 0.0006 |
| RBF SVM | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.0000 | 0.0063 | 0.0025 |

## Observations
- RBF SVM is the best model (perfect test score), then the trees (99.3%), then the linear SVMs (97%). All models have recall 1.0, so differences come from precision.
- Using all four features lifts every model above 97%, compared with 88% to 93% using two features.
- **Interpretability:** the tree is easiest to explain (if/else rules). The linear SVM gives one weight per feature. The RBF SVM is a black box.
- **Scaling:** SVMs need scaled features; trees do not.
- **Overfitting and complexity:** deep trees memorize the training set; `max_depth` controls this. Linear SVMs are simple and stable but limited. RBF is flexible and generalizes well here.
- **Speed:** the manual tree is slowest to train because it searches every threshold in Python loops. Scikit-learn is faster for both models. Prediction is fast for all models at this data size.
