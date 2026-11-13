# SF Airbnb Price Regression

Course project (AI & ML, section 2): predict the nightly `price` of San Francisco Airbnb listings from tabular listing data (a regression task).

**Notebook:** [`notebooks/sf_airbnb_price.ipynb`](notebooks/sf_airbnb_price.ipynb) holds the whole analysis: setup, loading, EDA, cleaning, feature engineering, train/test split and models.

## Dataset

Kaggle "San Francisco Airbnb open data" (by kabhishm), file `listings.csv` (7,535 listings, 18 columns). The data is not included in this repository. Download it from Kaggle and put `listings.csv` in `data/raw/`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sf_airbnb_price.ipynb
```

Then use **Restart & Run All**.

## Approach

- Cleaning: drop identifier and free-text columns, fill missing `reviews_per_month` for listings with no reviews, cap extreme `minimum_nights` values at 365.
- Features: days since last review, distance to downtown, multi-listing host flag.
- Split first, then preprocess: imputing, scaling and one-hot encoding are fitted inside a scikit-learn `Pipeline` on the training set only, to avoid data leakage.
- Models are compared with 5-fold cross-validation on the training set, starting from baselines and moving up to linear models and tree-based models. The test set is used once, at the end.

## Status

Work in progress. Models so far: baselines, Linear Regression, Ridge, Lasso. Tree-based models and the final test-set evaluation are still to come.
