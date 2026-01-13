
# ===============================================
# Exam Score Prediction - Professional ML Pipeline
# ===============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------- 1. Data Preprocessing --------------------
def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the dataset:
    - Drops duplicates and unnecessary columns
    - Encodes categorical variables
    - Converts text features to numerical
    - Applies one-hot encoding to 'sleep_quality'
    """
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()

    # Drop unnecessary columns
    df = df.drop(['student_id', 'gender'], axis=1)

    # Encode categorical variables
    df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})

    df['study_method'] = df['study_method'].map({
        'self-study': 0, 'online videos': 1, 'coaching': 2,
        'group study': 3, 'mixed': 4
    })

    df['course'] = df['course'].map({
        'bca': 0, 'ba': 1, 'b.sc': 2, 'b.com': 3,
        'bba': 4, 'diploma': 5, 'b.tech': 6
    })

    df['exam_difficulty'] = df['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
    df['facility_rating'] = df['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})

    # One-hot encode sleep_quality
    df = pd.get_dummies(df, columns=['sleep_quality'], drop_first=False)
    sleep_cols = [col for col in df.columns if 'sleep_quality' in col]
    df[sleep_cols] = df[sleep_cols].astype(np.int16)

    return df

# -------------------- 2. Split and Scale --------------------
def split_and_scale(df: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42):
    """
    Splits dataset into train and test sets and applies StandardScaler.
    Returns: x_train, x_test, y_train, y_test, scaler
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler

# -------------------- 3. Train & Evaluate Models --------------------
def train_and_evaluate(models: dict, x_train, x_test, y_train, y_test) -> pd.DataFrame:
    """
    Trains multiple models and evaluates them on test data.
    Returns a DataFrame of R2, MAE, RMSE for comparison.
    """
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results[name] = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    return pd.DataFrame(results).T

# -------------------- 4. Plot Predictions --------------------
def plot_actual_vs_predicted(y_test, y_pred, model_name: str):
    """
    Plots actual vs predicted values for visual inspection.
    """
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Exam Score")
    plt.ylabel("Predicted Exam Score")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.show()

# -------------------- 5. Main Pipeline --------------------
def main():
    # Load and preprocess data
    df = preprocess_data("Exam_Score_Prediction.csv")

    # Split and scale
    x_train, x_test, y_train, y_test, scaler = split_and_scale(df, target_col='exam_score')

    # Define models
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    # Train & evaluate
    results_df = train_and_evaluate(models, x_train, x_test, y_train, y_test)
    print("\nModel Comparison:\n", results_df)

    # Plot best model predictions
    best_model_name = results_df['R2'].idxmax()
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(x_test)
    plot_actual_vs_predicted(y_test, y_pred_best, best_model_name)

if __name__ == "__main__":
    main()




















