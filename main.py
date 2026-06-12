import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pd.read_csv("data/train.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ==========================================
# 2. DATASET INFORMATION
# ==========================================

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)

print(df.info())


# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

print(missing_values.sort_values(ascending=False))


# ==========================================
# 4. EDA - HOUSE PRICE DISTRIBUTION
# ==========================================

print("\nGenerating Price Distribution Plot...")

plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True)

plt.title("Sale Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Count")

plt.savefig("outputs/price_distribution.png")
plt.close()


# ==========================================
# 5. CORRELATION HEATMAP
# ==========================================

print("Generating Correlation Heatmap...")

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr())

plt.title("Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")
plt.close()


# ==========================================
# 6. TOP CORRELATED FEATURES
# ==========================================

print("\n" + "=" * 50)
print("TOP FEATURES CORRELATED WITH SALE PRICE")
print("=" * 50)

correlation = numeric_df.corr()

top_features = (
    correlation["SalePrice"]
    .sort_values(ascending=False)
)

print(top_features.head(15))


# ==========================================
# 7. HANDLE MISSING VALUES
# ==========================================

print("\nHandling Missing Values...")

numerical_columns = df.select_dtypes(
    include=np.number
).columns

for col in numerical_columns:
    df[col] = df[col].fillna(
        df[col].median()
    )

categorical_columns = df.select_dtypes(
    include="object"
).columns

for col in categorical_columns:
    df[col] = df[col].fillna(
        df[col].mode()[0]
    )


# ==========================================
# 8. ENCODE CATEGORICAL VARIABLES
# ==========================================

print("Encoding Categorical Variables...")

df = pd.get_dummies(
    df,
    drop_first=True
)

print("New Dataset Shape:", df.shape)


# ==========================================
# 9. FEATURES AND TARGET
# ==========================================

X = df.drop(
    "SalePrice",
    axis=1
)

y = df["SalePrice"]


# ==========================================
# 10. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 11. LINEAR REGRESSION
# ==========================================

print("\nTraining Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


# ==========================================
# 12. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    linear_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

print("\nLinear Regression Results")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))


# ==========================================
# 13. RIDGE REGRESSION
# ==========================================

print("\nTraining Ridge Regression...")

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train,
    y_train
)

ridge_predictions = ridge_model.predict(
    X_test
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ridge_predictions
    )
)

print("Ridge RMSE:", round(ridge_rmse, 2))


# ==========================================
# 14. LASSO REGRESSION
# ==========================================

print("\nTraining Lasso Regression...")

lasso_model = Lasso(alpha=100)

lasso_model.fit(
    X_train,
    y_train
)

lasso_predictions = lasso_model.predict(
    X_test
)

lasso_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lasso_predictions
    )
)

print("Lasso RMSE:", round(lasso_rmse, 2))


# ==========================================
# 15. CROSS VALIDATION
# ==========================================

print("\nRunning 5-Fold Cross Validation...")

scores = cross_val_score(
    linear_model,
    X,
    y,
    cv=5,
    scoring="neg_mean_squared_error"
)

cv_rmse = np.sqrt(-scores)

print("Cross Validation RMSE Scores:")
print(cv_rmse)

print(
    "Average CV RMSE:",
    round(cv_rmse.mean(), 2)
)


# ==========================================
# 16. FEATURE IMPORTANCE
# ==========================================

print("\nFinding Important Features...")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nTop 10 Positive Features")
print(importance.head(10))

print("\nTop 10 Negative Features")
print(importance.tail(10))


# ==========================================
# 17. FEATURE IMPORTANCE PLOT
# ==========================================

top10 = importance.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=top10
)

plt.title(
    "Top 10 Important Features"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()


# ==========================================
# 18. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    linear_predictions
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted Prices"
)

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.close()


# ==========================================
# 19. SAVE MODEL
# ==========================================

joblib.dump(
    linear_model,
    "models/house_price_model.pkl"
)

print("\nModel Saved Successfully!")


# ==========================================
# 20. PROJECT COMPLETED
# ==========================================

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nGenerated Files:")
print("outputs/price_distribution.png")
print("outputs/correlation_heatmap.png")
print("outputs/feature_importance.png")
print("outputs/actual_vs_predicted.png")
print("models/house_price_model.pkl")