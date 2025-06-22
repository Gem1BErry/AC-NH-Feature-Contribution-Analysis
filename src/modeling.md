```python
# Recode sex variable: 1 for Male, 2 for Female, 3 for Others
def recode_sex(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    else:
        return 3

df['sex'] = df['sex'].apply(recode_sex)

# Create age groups for stratification
bins = [0, 20, 30, 40, 50, 80]
labels = ['0-20', '21-30', '31-40', '41-50', '51+']
df['age_group_stratify'] = pd.cut(df['age'], bins=bins, labels=labels)

# Combine sex and age group into one stratification column
df['stratify_col'] = df['sex'].astype(str) + '_' + df['age_group_stratify'].astype(str)

# Define features and target variable for modeling
X_features = [
    'age', 'sex', 'autonomy_freedom', 'autonomy_interesting', 'autonomy_options',
    'competence_matched', 'competence_capable', 'competence_competent',
    'related_important', 'related_fulfilling', 'related_not_close',
    'enjoyment_fun', 'enjoyment_attention', 'enjoymen_boring', 'enjoyment_enjoyed',
    'extrinsic_avoid', 'extrinsic_forget', 'extrinsic_compelled', 'extrinsic_escape',
    'Hours'
]
X = df[X_features]
y = df['happiness_value']

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df['stratify_col']
)

# Add constant term for OLS regression
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit multiple linear regression model with robust standard errors (HC3)
mlr_model = sm.OLS(y_train, X_train_sm).fit(cov_type='HC3')
y_pred_mlr = mlr_model.predict(X_test_sm)

# Calculate MSE and R² for MLR model
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)

# Fit Random Forest regressor
rf_model = RandomForestRegressor(
    n_estimators=250, max_depth=10, min_samples_split=5, min_samples_leaf=5, random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Calculate MSE and R² for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Fit XGBoost regressor
xgb_model = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
    min_child_weight=1, objective='reg:squarederror', random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Calculate MSE and R² for XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Compile results into a dataframe
results = pd.DataFrame({
    'Model': ['MLR', 'Random Forest', 'XGBoost'],
    'Test MSE': [mse_mlr, mse_rf, mse_xgb],
    'Test R²': [r2_mlr, r2_rf, r2_xgb]
})

# Plot missing data heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Define continuous variables for visualization
cont_vars = [
    "autonomy_freedom", "autonomy_interesting", "autonomy_options",
    "competence_matched", "competence_capable", "competence_competent",
    "related_important", "related_fulfilling", "related_not_close",
    "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
    "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape",
    "Hours", 'happiness_value'
]

# Plot distribution histograms for continuous variables with KDE
for var in cont_vars:
    plt.figure(figsize=(8,4))
    sns.histplot(df[var].dropna(), kde=True)
    plt.title(f"Distribution of {var}")
    plt.show()

# Define categorical variables for visualization
cat_vars = ['gender', 'age_group_stratify']

# Plot countplots for categorical variables
for var in cat_vars:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=var)
    plt.title(f"Distribution of {var}")
    plt.show()

# Plot correlation heatmap of continuous variables
plt.figure(figsize=(10,8))
corr = df[cont_vars].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix of Continuous Variables")
plt.show()

# Plot boxplots for Hours and happiness_value variables
for var in ['Hours', 'happiness_value']:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[var])
    plt.title(f"Boxplot of {var}")
    plt.show()
```
