# 📈 Machine Learning: Linear Regression Projects

A comprehensive machine learning repository demonstrating the end-to-end implementation of **Simple Linear Regression (SLR)** and **Multiple Linear Regression (MLR)** using `scikit-learn` and `statsmodels`.

This repository showcases real-world machine learning practices: exploratory data analysis (EDA), statistical significance testing (p-values & t-statistics), feature selection, data leakage prevention during scaling, model evaluation, and inference.

---

## 📁 Repository Structure

```text
Linear_Regresion/         
├── student_marks_prediction.ipynb       # Simple Linear Regression & Inference
├── house_price_regression_dataset.csv   # Multiple Linear Regression & OLS Analysis
├── StudentMarksDataset.csv              # Dataset for Student Marks Prediction (100 rows)
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

---

## 📊 Projects Overview & Comparison

| Metric / Aspect | 🎓 Student Marks Prediction | 🏡 House Price Prediction |
| :--- | :--- | :--- |
| **Model Type** | Simple Linear Regression (1D) | Multiple Linear Regression (Multi-D) |
| **Notebook** | `student_marks_prediction.ipynb` | `House_price_rediction.ipynb` |
| **Dataset** | `StudentMarksDataset.csv` (100 samples) | `house_price_regression_dataset.csv` (1,000 samples) |
| **Target Variable ($y$)** | `Std_Marks` (Exam Marks) | `House_Price` (USD $) |
| **Predictor Count ($X$)** | 1 Feature | 6 Features (Selected via OLS) |
| **Train / Test Split** | 75% Train / 25% Test (`random_state=45`) | 75% Train / 25% Test (`random_state=42`) |
| **Preprocessing** | Mean Imputation + `StandardScaler` | Feature Selection + `StandardScaler` |
| **$R^2$ Score** | **`0.9523` (95.2%)** | **`0.9984` (99.8%)** |
| **Adjusted $R^2$** | **`0.9513`** | **`0.9983`** |
| **Mean Absolute Error (MAE)**| **`0.9052`** marks | **`$8,152.90`** |
| **Root Mean Squared Error (RMSE)**| **`1.0998`** marks | **`$10,183.28`** (~1% error) |

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

## 🔬 Key Engineering & Statistical Concepts Demonstrated

1. **Univariate vs. Multivariate Significance**:
   - Demonstrated why low Pearson correlation does not imply a feature is useless. Multivariate regression isolates individual feature contributions while holding others constant.
2. **Preventing Data Leakage**:
   - Scalers are fitted **only** on training partitions (`scaler.fit_transform(X_train)`), and applied downstream to test partitions (`scaler.transform(X_test)`).
3. **$R^2$ vs. Adjusted $R^2$**:
   - Tracked Adjusted $R^2$ across both projects to penalize redundant parameters and avoid artificial inflation of model fit.
4. **OLS Regression Diagnostics**:
   - Utilized `statsmodels.api.OLS` for p-values, standard errors, condition numbers, and AIC comparison.

---

## 🛠️ Tech Stack
- **Language**: Python 3
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Statistical Modeling**: `statsmodels`
