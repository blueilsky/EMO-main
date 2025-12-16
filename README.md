
# EMO: Elastic Oversampling with Boundary Skeletons from Minimal Enclosing Balls

This repository contains the reference MATLAB implementation of **EMO**, an elastic modeling-based oversampling method for imbalanced classification.

## 1. Datasets

The benchmark datasets used in the paper can be downloaded from:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- KEEL Repository: https://sci2s.ugr.es/keel/
- PROMISE Repository: http://promisedata.org/repository

Please place the processed datasets into the `Dataset/` folder, e.g.

```text
Dataset/
    dataset1.csv
    dataset2.csv
    ...
```

## 2. Main Files

- `meb_plot.m`  
  Main implementation of the EMO oversampling algorithm.

- `LS_Test.m`  
  Support pair extraction between majority and minority classes.

- `LARGEfacetIntersection.m`, `LARGElineSearch.m`, `LARGEupdateS.m`  
  Utilities used in the linear separability test, following:
  > Shuiming Zhong and Huan Lyu, "A New Sufficient & Necessary Condition for Testing Linear Separability between Two Sets," TPAMI, 2024.

- `experiment.m`  
  Main experimental script that runs EMO and baselines on the datasets.

## 3. Basic Usage

1. Add the project folder to the MATLAB path:

```matlab
addpath(genpath(pwd));
```

2. Run the main experiment:

```matlab
experiment;
```

3. Apply EMO to your own data:

```matlab
% X: N x d features, y: N x 1 labels (0 = majority, 1 = minority)
[X_res, y_res] = meb_plot(X, y);
```

## 4. Citation

```text
S. Zhong and H. Lyu,
"A New Sufficient & Necessary Condition for Testing Linear Separability between Two Sets,"
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
```
