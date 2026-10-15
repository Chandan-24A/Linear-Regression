# 📈 Machine Learning: Linear Regression Projects

A comprehensive machine learning repository demonstrating the end-to-end implementation of **Simple Linear Regression (SLR)**, **Multiple Linear Regression (MLR)**, and **Polynomial & Regularized Regression (Ridge, Lasso, ElasticNet)** using `scikit-learn` and `statsmodels`.

This Project showcases real-world machine learning practices: exploratory data analysis (EDA), statistical significance testing (p-values & t-statistics), polynomial feature engineering, handling bias-variance tradeoff, L1/L2 regularization, automated hyperparameter tuning via Cross-Validation, and model inference.

---

## 📁 Project Structure

```text
Linear_Regresion/
├── Car_fuel_efficiency_predictor.ipynb  # Polynomial, Ridge, Lasso & ElasticNet Regression
├── House_price_prediction.ipynb         # Multiple Linear Regression & OLS Hypothesis Testing
├── student_marks_prediction.ipynb       # Simple Linear Regression & Inference
├── auto-mpg.csv                         # Auto MPG Dataset (398 rows)
├── house_price_regression_dataset.csv   # Dataset for House Price Prediction (1,000 rows)
├── StudentMarksDataset.csv              # Dataset for Student Marks Prediction (100 rows)
├── requirements.txt                     # Project dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 📊 Projects Overview & Comparison

| Metric / Aspect | 🎓 Student Marks Prediction | 🏡 House Price Prediction | 🚗 Car Fuel Efficiency Prediction |
| :--- | :--- | :--- | :--- |
| **Model Type** | Simple Linear Regression (1D) | Multiple Linear Regression (Multi-D) | Polynomial & Regularized Regression |
| **Notebook** | `student_marks_prediction.ipynb` | `House_price_prediction.ipynb` | `Car_fuel_efficiency_predictor.ipynb` |
| **Dataset** | `StudentMarksDataset.csv` (100 rows) | `house_price_regression_dataset.csv` (1,000 rows) | `auto-mpg.csv` (398 rows) |
| **Target Variable ($y$)** | `Std_Marks` (Exam Marks) | `House_Price` (USD $) | `mpg` (Miles Per Gallon) |
| **Predictor Count ($X$)** | 1 Feature | 6 Features (Selected via OLS) | 8 Features $\rightarrow$ 44 (Poly Deg-2) |
| **Train / Test Split** | 75% Train / 25% Test (`random_state=45`) | 75% Train / 25% Test (`random_state=42`) | 75% Train / 25% Test (`random_state=42`) |
| **Preprocessing** | Mean Imputation + `StandardScaler` | OLS Selection + `StandardScaler` | One-Hot Encoding + Mean Imp + `StandardScaler` |
| **Best Model Algorithm**| Ordinary Least Squares (OLS) | Multiple Linear Regression (OLS) | **Ridge Regression (Degree 2, $\alpha \approx 1.53$)** |
| **$R^2$ Score** | **`0.9523` (95.2%)** | **`0.9984` (99.8%)** | **`0.9076` (90.8%)** |
| **Adjusted $R^2$** | **`0.9513`** | **`0.9983`** | **`0.8998`** |
| **Mean Absolute Error (MAE)**| **`0.9052`** marks | **`$8,152.90`** | **`1.7027`** mpg |
| **Root Mean Squared Error (RMSE)**| **`1.0998`** marks | **`$10,183.28`** (~1% error) | **`2.2908`** mpg |

---

## 🎓 Project 1: Student Marks Prediction

### 1. Problem Statement
Predict a student's final marks based on the number of study hours dedicated.

### 2. Feature Workflow
- **Raw Columns**: `Std_RollNo`, `Std_Name`, `Std_Branch`, `Std_Course`, `Std_StudyHours`, `Std_Marks`.
- **Feature Selection**: Dropped non-predictive administrative identifiers (`Std_RollNo`, `Std_Name`, `Std_Branch`, `Std_Course`).
- **Data Cleaning**: Handled missing values in `Std_StudyHours` using mean imputation (`fillna()`).
- **Data Leakage Prevention**: Scaled features using `StandardScaler` by fitting strictly on `X_train` and transforming `X_test`.

### 3. Model Parameters & Interpretation
$$\text{Marks} = 78.23 + 4.73 \times (\text{Standardized Study Hours})$$
- **Intercept ($\beta_0$)**: `78.23` (baseline marks for an average student).
- **Slope ($\beta_1$)**: `+4.73` (each standard deviation increase in study hours yields an additional ~4.7 marks).

### 4. Sample Inference
- Studying **`7.34 hours`** $\rightarrow$ Predicted Marks: **`79.30`**
- Studying **`5.64 hours`** $\rightarrow$ Predicted Marks: **`72.81`**

---

## 🏡 Project 2: House Price Prediction

### 1. Problem Statement
Predict property valuation from multi-dimensional architectural and location characteristics using Multiple Linear Regression.

### 2. In-Depth Feature Selection & Hypothesis Testing
Raw features evaluated: `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`.

During Exploratory Data Analysis, Pearson correlation indicated near-zero univariate correlations for `Num_Bedrooms` (`0.014`) and `Num_Bathrooms` (`-0.001`). Rather than blindly dropping them, an **Ordinary Least Squares (OLS) hypothesis test** was performed:

| Feature | OLS Coefficient ($\beta$) | t-statistic | p-value | Decision | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Square_Footage`** | +$199.75 / sq.ft. | 804.0 | **`0.0000`** | **Keep** | Primary linear driver of property cost. |
| **`Num_Bedrooms`** | +$10,160 / room | 46.4 | **`0.0000`** | **Keep** | Statistically significant ($p < 0.001$) when controlling for area. |
| **`Num_Bathrooms`** | +$8,250 / bath | 21.8 | **`0.0000`** | **Keep** | Highly significant ($p < 0.001$). |
| **`Year_Built`** | +$991.38 / year | 65.8 | **`0.0000`** | **Keep** | Premium for newer construction. |
| **`Lot_Size`** | +$14,930 / unit | 62.1 | **`0.0000`** | **Keep** | Second strongest driver ($r = 0.160$). |
| **`Garage_Size`** | +$5,156 / car | 13.5 | **`0.0000`** | **Keep** | Significant amenity premium. |
| **`Neighborhood_Quality`**| +$80.60 | 0.749 | **`0.4540`** | **DROP** | **Insignificant ($p > 0.05$)**; pure noise. Removing it improved model AIC. |

### 3. Final Model Evaluation
- **$R^2$ Score**: **`0.9984`**
- **Adjusted $R^2$**: **`0.9983`**
- **Root Mean Squared Error (RMSE)**: **`$10,183.28`**

---

## 🚗 Project 3: Car Fuel Efficiency Prediction (Polynomial & Regularization)

### 1. Problem Statement
Predict vehicle fuel efficiency (`mpg` — Miles Per Gallon) from engine, weight, and geographical origin attributes using **Polynomial Feature Engineering** and **Regularized Linear Models**.

### 2. End-to-End Workflow & Preprocessing
- **Data Cleaning**: Converted `horsepower` from object to numeric, handling missing values with average imputation.
- **Categorical Feature Encoding**: The `origin` feature (1: USA, 2: Europe, 3: Asia) was mapped and **One-Hot Encoded** using `pd.get_dummies(..., drop_first=True)` to eliminate false numerical hierarchy assumptions.
- **Pipeline Architecture**: Structured transformations using Scikit-Learn `Pipeline` (`PolynomialFeatures` $\rightarrow$ `StandardScaler` $\rightarrow$ Estimator) to eliminate data leakage.

### 3. Polynomial Degree Progression & Bias-Variance Diagnosis
- **Degree 1 (Linear Regression, 8 features)**: $R^2 = 0.8430$, $\text{RMSE} = 2.986$. Misses non-linear curvature between vehicle weight and fuel efficiency.
- **Degree 2 (Polynomial OLS, 44 features)**: $R^2 = 0.8983$, $\text{RMSE} = 2.403$. Captures interaction and quadratic effects.
- **Degree 3 (Polynomial OLS, 164 features)**: $R^2 = -0.0406$, $\text{RMSE} = 7.687$. **Catastrophic Overfitting** due to extreme parameter explosion relative to sample size ($N=298$).

### 4. Regularization & Cross-Validation Benchmarking
To resolve the variance explosion and multicollinearity of polynomial features, regularized models were trained at **Degree 2** with 5-Fold Cross-Validation:

| Model | Regularization Penalty | Optimal Hyperparameters | Test $R^2$ | Test RMSE | Active Features |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Ridge Regression** | L2 ($\sum w_j^2$) | $\alpha = 1.5264$ | **`0.9076`** | **`2.2908`** | 44 / 44 (Smooth shrinkage) |
| **Lasso Regression** | L1 ($\sum \|w_j\|$) | $\alpha = 0.0173$ | **`0.9013`** | **`2.3671`** | **16 / 44 (Feature Selection)** |
| **ElasticNet Regression** | L1 + L2 Blend | $\alpha = 0.0158$,$\text{l1\_ratio} = 0.95$ | **`0.9017`** | **`2.3626`** | 16 / 44 (Sparse grouping) |

### 5. Key Insights
- **Ridge (L2)** delivered the **highest predictive accuracy** ($R^2 = 90.76\%$) by shrinking correlated polynomial weights.
- **Lasso (L1)** successfully performed **automated feature elimination**, driving 28 redundant polynomial interactions to strictly zero while preserving 90% accuracy with only 16 active terms.
- **ElasticNet** verified that sparse feature selection (L1 penalty) dominates this dataset (`l1_ratio = 0.95`).

---

## 🔬 Key Statistical Concepts Demonstrated

1. **Univariate vs. Multivariate Significance**:
   - Demonstrated why low Pearson correlation does not imply a feature is useless. Multivariate regression isolates individual feature contributions while holding others constant.
2. **Preventing Data Leakage**:
   - Scalers and encoders are fitted **strictly** on training partitions (`fit_transform(X_train)`) and applied to test partitions (`transform(X_test)`), embedded within Scikit-Learn `Pipeline` objects.
3. **The Bias-Variance Tradeoff in Polynomial Spaces**:
   - Observed the progression from high bias (Degree 1 linear model) to optimal balance (Degree 2 + Ridge) to extreme high variance (Degree 3 OLS).
4. **L1 vs. L2 Regularization Mechanics**:
   - Analyzed how Ridge stabilizes collinear feature spaces via smooth coefficient shrinkage, while Lasso produces sparse, interpretable models by zeroing out non-essential interaction terms.
5. **Hyperparameter Tuning via Cross-Validation**:
   - Implemented `RidgeCV`, `LassoCV`, and `ElasticNetCV` with 5-fold cross-validation to prevent test-set overfitting during parameter searches.
6. **OLS Regression Diagnostics**:
   - Utilized `statsmodels.api.OLS` for p-values, t-statistics, condition numbers, and AIC comparison.

---

## 🛠️ Tech Stack
- **Language**: Python 3
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (`LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `PolynomialFeatures`, `Pipeline`, `StandardScaler`)
- **Statistical Modeling**: `statsmodels`
