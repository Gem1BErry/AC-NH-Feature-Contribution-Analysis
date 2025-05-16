```python
import pandas as pd
import xgboost as xgb
import shap

# Define input features and target
X = df[["autonomy_freedom", "autonomy_interesting", "autonomy_options",
        "competence_matched", "competence_capable", "competence_competent",
        "related_important", "related_fulfilling", "related_not_close",
        "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
        "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape"]]
y = df['happiness_value']

# Best hyperparameters from previous tuning
best_params = {
    'colsample_bytree': 0.6,
    'learning_rate': 0.01,
    'max_depth': 7,
    'n_estimators': 300,
    'subsample': 0.6
}

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", **best_params)
model.fit(X, y)

# Compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Generate SHAP value matrix (samples × features)
shap_matrix = pd.DataFrame(shap_values.values, columns=X.columns)

# Compute mean absolute SHAP values for each feature
shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean SHAP Value": abs(shap_values.values).mean(axis=0)
}).sort_values(by="Mean SHAP Value", ascending=False)

# Define experience feature groups
dimensions = {
    "Autonomy": ["autonomy_freedom", "autonomy_interesting", "autonomy_options"],
    "Competence": ["competence_matched", "competence_capable", "competence_competent"],
    "Relatedness": ["related_important", "related_fulfilling", "related_not_close"],
    "Enjoyment": ["enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed"],
    "Extrinsic Motivation": ["extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape"]
}

# Aggregate mean SHAP values by dimension
dimension_contributions = {}
for dimension, features in dimensions.items():
    mask = shap_df['Feature'].isin(features)
    dimension_contributions[dimension] = shap_df.loc[mask, 'Mean SHAP Value'].sum()

# Convert to DataFrame and calculate percentage
df_contrib = pd.DataFrame.from_dict(dimension_contributions, orient='index', columns=['Contribution'])
df_contrib = df_contrib.sort_values('Contribution', ascending=False)
df_contrib['Percentage'] = (df_contrib['Contribution'] / df_contrib['Contribution'].sum() * 100).round(1)

