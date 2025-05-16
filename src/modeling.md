
```python
#Multiple Linear Regression (MLR)
#1. Using both Subjective Experience (SE) and Gameplay Time as input variables
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

# Define features and target
X = df[["autonomy_freedom", "autonomy_interesting", "autonomy_options",
        "competence_matched", "competence_capable", "competence_competent",
        "related_important", "related_fulfilling", "related_not_close",
        "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
        "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape",
        "Hours"]]
y = df['happiness_value']

# Initialize model
model = LinearRegression()

# Define scoring metrics
scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
}

# Perform 5-fold cross-validation
cv_results = {metric: cross_val_score(model, X, y, cv=5, scoring=score) for metric, score in scoring.items()}


#2. Using only Gameplay Time as input
X = df[["Hours"]]
y = df['happiness_value']

model = LinearRegression()

scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
}

cv_results = {metric: cross_val_score(model, X, y, cv=5, scoring=score) for metric, score in scoring.items()}

#3. Using only Gameplay Time as input
X = df[["autonomy_freedom", "autonomy_interesting", "autonomy_options",
        "competence_matched", "competence_capable", "competence_competent",
        "related_important", "related_fulfilling", "related_not_close",
        "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
        "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape"]]
y = df['happiness_value']

model = LinearRegression()

scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)))
}

cv_results = {metric: cross_val_score(model, X, y, cv=5, scoring=score) for metric, score in scoring.items()}


#XGBoost Regression with Hyperparameter Tuning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold

X = df[["autonomy_freedom", "autonomy_interesting", "autonomy_options",
        "competence_matched", "competence_capable", "competence_competent",
        "related_important", "related_fulfilling", "related_not_close",
        "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
        "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape"]]
y = df['happiness_value']

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

scoring = {
    "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
    "r2": make_scorer(r2_score)
}

search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring=scoring,
    refit="r2",
    n_jobs=-1
)

search.fit(X, y)

# Extract best parameters and scores
best_params = search.best_params_
best_r2 = search.best_score_
best_mse = -search.cv_results_["mean_test_neg_mse"][search.best_index_]


#Random Forest Regression with Grid Search
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

X = df[["autonomy_freedom", "autonomy_interesting", "autonomy_options",
        "competence_matched", "competence_capable", "competence_competent",
        "related_important", "related_fulfilling", "related_not_close",
        "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
        "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape"]]
y = df['happiness_value']

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X, y)

# Final evaluation with best parameters
final_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
cv_results = cross_validate(final_model, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error'], n_jobs=-1)

mean_r2 = np.mean(cv_results['test_r2'])
mean_mse = -np.mean(cv_results['test_neg_mean_squared_error'])

