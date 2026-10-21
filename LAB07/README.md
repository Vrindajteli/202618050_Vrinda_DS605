# Lab 7: Image Representation, PCA, Eigenfaces and Face Recognition

#### Dataset: AT&T Database of Faces (ORL)
#### Student ID: 202618050

This project turns face photos into numbers, shrinks those numbers with PCA, and then checks how well people can be recognized from the shrunken version. Everything is in one notebook: `DS605_Lab7_PCA_Eigenfaces.ipynb`.

## The dataset

- 40 people, 10 photos each, 400 images in total
- Each photo is 112 x 92 pixels, grayscale, so 10,304 pixel values per image
- Pixel values run from 0 to 251

## How to run it

1. Download the AT&T faces and place them in a folder named `data` next to the notebook. It should contain folders `s1` to `s40`, each with `1.pgm` to `10.pgm`.
2. Install the libraries: `pip install numpy pandas matplotlib pillow scikit-learn jupyter`
3. Open the notebook and run all cells from top to bottom.

All plots appear inside the notebook, so the saved outputs show the figures.

## What was done

**Part A: Images as numbers.** Each image is read as a grayscale grid and flattened into one row of 10,304 values. That gives a 400 x 10,304 matrix with a label for each row. The data is split 70/30 with a fixed random seed (42), keeping the people balanced: 7 photos per person for training (280 images) and 3 per person for testing (120 images). All 40 people appear in both sets.

**Part B: PCA and eigenfaces.**
- The mean face is computed from the training images and subtracted from every image.
- The principal directions are found two ways using NumPy: an eigendecomposition (done on the small 280 x 280 matrix, which is much cheaper than the full 10,304 x 10,304 covariance matrix) and a direct SVD.
- The cumulative explained variance is plotted, and the smallest number of components reaching 90% is kept.
- The top 16 eigenfaces are reshaped back into 112 x 92 images.
- Test faces are projected into the reduced space, rebuilt from different numbers of components, and checked for quality (PSNR) and storage.

**Part C: Recognition by closest match.** Each test face is compared to every training face using Euclidean distance, and it takes the label of the closest one. This is done on raw pixels and on the reduced PCA values.

**Part D: Traditional classifiers.** KNN (k=1 and k=3), Logistic Regression, a linear SVM and an RBF SVM are trained on the PCA values and compared to the closest-match method. Then the number of components is varied from 2 to 250 to see how accuracy changes. Only the number of components is varied; the classifiers keep scikit-learn's default settings.

## Results

**Eigendecomposition vs SVD**
- Both methods gave the same directions (cosine similarity of 1.0 for the top 10) and the same eigenvalues (largest difference among the top 50 was about 1.9e-09).
- On my machine the small-matrix eigendecomposition took 0.22 s and the SVD took 3.58 s.

**Components needed for 90% variance:** 91 out of 280.

**Compression:** each face goes from 10,304 numbers to 91, about 113 times fewer. This counts only the per-face codes. The mean face and the eigenfaces also have to be stored once and shared by all faces.

**Reconstruction:** with 5 to 15 components, faces are blurry and only the general head shape and lighting show. At 91 components the face is clear and recognizable. At 150 and above, the result is very close to the original.

**Closest-face recognition**

| Features | Test accuracy | Matching time |
|---|---|---|
| Raw pixels (10,304 values) | 95.00% | 4171.87 ms |
| PCA (91 values) | 95.83% | 30.99 ms |

PCA matching was slightly more accurate and about 135 times faster. In PCA space, 115 of 120 test faces were matched correctly and 5 were wrong.

**Classifiers at 91 components**

| Classifier | Test accuracy |
|---|---|
| Closest face (1-NN, my own code) | 95.83% |
| KNN (k=1) | 95.83% |
| KNN (k=3) | 89.17% |
| Logistic Regression | 91.67% |
| Linear SVM | 96.67% |
| RBF SVM | 95.83% |

**Changing the number of components**

| Model | Best number of components | Best accuracy | Accuracy at 91 |
|---|---|---|---|
| 1-NN | 30 | 95.83% | 95.83% |
| KNN (k=3) | 30 | 90.00% | 89.17% |
| Logistic Regression | 120 | 97.50% | 91.67% |
| Linear SVM | 40 | 97.50% | 96.67% |
| RBF SVM | 40 | 96.67% | 95.83% |

The best number of components in this table is read off the test accuracy, so it shows the trend rather than a fully fair estimate.

## Observations

- A small number of components goes a long way. Accuracy rises quickly up to about 30 components, and for most models it does not get better after 40. The 90% variance point (91 components) is a good safe choice but is not the sharpest one for recognition.
- The simple closest-face method holds up well. It matched or beat Logistic Regression and KNN with k=3 at 91 components, and was only about one image behind the linear SVM.
- The linear SVM was the most reliable classifier, reaching 97.50% with only 40 components, which is a very compact representation.
- Logistic Regression needed many more components (120) to reach its best result, and it printed convergence warnings, so it may do better with scaled features or more iterations.
- With only 120 test images, each image is worth about 0.83 percentage points. Differences of one or two images between methods are small and should not be over-read.
- The wrong matches probably come from large changes in expression, head angle, glasses or lighting, where a face can look closer to another person than to the same person in different conditions.


