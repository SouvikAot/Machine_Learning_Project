import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
df = pd.read_csv("heart.csv")

# -------------------------------------------------
# 2. Feature Engineering
# -------------------------------------------------
df.rename(columns={'Sex': 'Gender'}, inplace=True)

df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

df['ChestPainType'] = df['ChestPainType'].map({
    'ASY': 0, 'NAP': 1, 'ATA': 2, 'TA': 3
})

df['ST_Slope'] = df['ST_Slope'].map({
    'Down': 0, 'Flat': 1, 'Up': 2
})

df = pd.get_dummies(df, columns=['RestingECG'], dtype=np.int16)

# -------------------------------------------------
# 3. Train-Test Split
# -------------------------------------------------
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# 4. Pipelines + Hyperparameter Grids
# -------------------------------------------------

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear"]
        }
    },

    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },

    "SVM": {
        "model": SVC(),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"]
        }
    },

    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {}   # No hyperparameters to tune
    }
}

# -------------------------------------------------
# 5. GridSearchCV with Cross-Validation
# -------------------------------------------------
best_models = {}

print("Hyperparameter Tuning Results:\n")

for name, config in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", config["model"])
    ])

    grid = GridSearchCV(
        pipeline,
        config["params"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_

    print(f"{name}")
    print(f"Best CV Accuracy : {grid.best_score_:.4f}")
    print(f"Best Parameters  : {grid.best_params_}\n")

# -------------------------------------------------
# 6. Final Evaluation on Test Data
# -------------------------------------------------
print("Test Set Performance:\n")

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# -------------------------------------------------
# 7. Detailed Evaluation (Best Model Example)
# -------------------------------------------------
best_model = best_models["Logistic Regression"]

y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
