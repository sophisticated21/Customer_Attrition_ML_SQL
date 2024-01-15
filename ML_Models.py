import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from Data_Preprocessing import preprocess_data
from Model_Config import feature_columns, target_column, lr_params, dt_params, rf_params
import pickle

csv_path = "Data/BankChurners.csv"
data = pd.read_csv(csv_path)
ccdata_processed = preprocess_data(data)

X = ccdata_processed[feature_columns]
y = ccdata_processed[target_column[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('train_test_data.pkl', 'wb') as file:
    pickle.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), file)

print("Training and test data saved as 'train_test_data.pkl'")

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    # Calculating various metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Compiling metrics into a dictionary
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc
    }

    return metrics


def train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test):
    start_time = time.time()
    print(f"Training {model.__class__.__name__}...")

    grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    metrics = evaluate_model(best_model, X_test, y_test)

    print(f"Training completed for {model.__class__.__name__} in {time.time() - start_time:.2f} seconds.")
    return best_model, best_params, metrics


# Logistic Regression
lr_model, lr_best_params, lr_metrics = train_and_evaluate_model(
    LogisticRegression(max_iter=10000),
    lr_params,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test
)

print("Logistic Regression Metrics:", lr_metrics)


# Decision Tree
dt_model, dt_best_params, dt_metrics = train_and_evaluate_model(
    DecisionTreeClassifier(),
    dt_params,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test
)
print("Decision Tree Metrics:", dt_metrics)

# Random Forest
rf_model, rf_best_params, rf_metrics = train_and_evaluate_model(
    RandomForestClassifier(),
    rf_params,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test
)
print("Random Forest Metrics:", rf_metrics)

with open('optimal_rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("Model saved as 'optimal_rf_model.pkl'")