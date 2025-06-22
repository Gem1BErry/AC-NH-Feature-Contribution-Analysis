```python
# Add a constant (intercept) to the predictors, which is required for OLS
X_const = sm.add_constant(X)

# Fit a Standard OLS Model
# This 'vanilla' OLS model is used specifically to generate the residuals for diagnostics.
# We do NOT use robust standard errors here, because the goal is to *find* the problems.
model_for_diagnostics = sm.OLS(y, X_const).fit()

# Get the residuals and fitted (predicted) values from the model
residuals = model_for_diagnostics.resid
fitted_values = model_for_diagnostics.fittedvalues

print("OLS model fitted successfully for diagnostic purposes.")
print("-" * 50)

# Generate Diagnostic Plots

# Create a figure to hold both plots for a clean presentation
plt.figure(figsize=(8, 12))

# Plot 1: Q-Q Plot for Normality of Residuals
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
plt.title('Q-Q Plot of Residuals', fontsize=14)
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)

# Plot 2: Residuals vs. Fitted Values for Homoscedasticity
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Fitted Values', fontsize=14)
plt.xlabel('Fitted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)

# Adjust layout to prevent titles from overlapping and display the plots
plt.tight_layout()
plt.show()

# Calculate and Display VIF for Multicollinearity

print("\n" + "="*50)
print("Calculating Variance Inflation Factor (VIF)...")
print("="*50)

# Create a new DataFrame to store the VIF results
vif_data = pd.DataFrame()
vif_data["feature"] = X_const.columns

# Calculate VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(len(X_const.columns))]

# The VIF for the constant is usually very high and not meaningful,
# so we can display it but focus on other features.
print(vif_data)
print("-" * 50)
```
