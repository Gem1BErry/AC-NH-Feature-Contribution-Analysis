```python
# --- Merge gender categories 3 and 4 into one ---
def merge_gender(x):
    return 3 if x in [3, 4] else x  # 3 = Non-binary/Prefer not to say

df['sex_merged'] = df['sex'].apply(merge_gender)

# --- Age grouping ---
df['age_group_stratify'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 80], 
                                  labels=['0-20', '21-30', '31-40', '41-50', '51+'])

# --- Create stratification column combining merged gender and age group ---
df['stratify_col'] = df['sex_merged'].astype(str) + '_' + df['age_group_stratify'].astype(str)

# --- Define features and target ---
X_features = [
    'age', 'sex_merged',
    'autonomy_freedom', 'autonomy_interesting', 'autonomy_options',
    'competence_matched', 'competence_capable', 'competence_competent',
    'related_important', 'related_fulfilling', 'related_not_close',
    'enjoyment_fun', 'enjoyment_attention', 'enjoymen_boring', 'enjoyment_enjoyed',
    'extrinsic_avoid', 'extrinsic_forget', 'extrinsic_compelled', 'extrinsic_escape',
    'Hours'
]
X = df[X_features]
y = df['happiness_value']

# --- Split data: 80% training with stratification ---
X_train, _, y_train, _ = train_test_split(
    X, y, train_size=0.80, random_state=42, stratify=df['stratify_col']
)

print(f"Training set size: {len(X_train)}")
print("-" * 30)

# --- 1. Random Forest hyperparameter tuning ---
print("="*50)
print("Random Forest Hyperparameter Tuning")

param_grid_rf = {
    'n_estimators': [280, 320],
    'max_depth': [8, 12],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
grid_rf.fit(X_train, y_train)

print("\nRandom Forest Results")
print("="*40)
print("Best params:", grid_rf.best_params_)
print(f"Best CV MSE: {-grid_rf.best_score_:.4f}")
print("="*40 + "\n")

# --- 2. XGBoost hyperparameter tuning ---
print("="*50)
print("XGBoost Hyperparameter Tuning")

param_grid_xgb = {
    'n_estimators': [100, 150],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'min_child_weight': [1, 3]
}

xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
grid_xgb.fit(X_train, y_train)

print("\nXGBoost Results")
print("="*40)
print("Best params:", grid_xgb.best_params_)
print(f"Best CV MSE: {-grid_xgb.best_score_:.4f}")
print("="*40)

```
