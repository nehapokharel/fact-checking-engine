
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from data_processing import fact_to_vector


def train_and_evaluate_model(X_train, y_train):
    """Trains a model and evaluates it using the ROC AUC metric."""
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Handle imbalance in the data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Split the training data into train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Grid search for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train_split, y_train_split)

    print("Best Parameters: ", grid.best_params_)
    print("Best AUC Score: ", grid.best_score_)

    # Evaluate on the validation set
    y_val_pred = grid.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_val_pred)
    print("ROC AUC Score on Validation Set:", roc_auc)

    return grid.best_estimator_, scaler

def check_fact_veracity(subject, predicate, object, graph, embeddings_model, scaler, model):
    """Checks the veracity of a fact using the trained model."""
    vector = fact_to_vector(subject, object, embeddings_model)
    if vector is not None:
        vector = scaler.transform([vector])  # Scale the vector
        veracity = model.predict_proba(vector)[0][1]
        return veracity
    else:
        return None

