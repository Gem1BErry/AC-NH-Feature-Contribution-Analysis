```python

# Create SHAP explainer and calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# --- New: Calculate and display numerical global feature importance ---

# Calculate mean absolute SHAP value for each feature
mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
feature_names = X_test.columns

# Create a DataFrame to hold features and their importance
global_feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Absolute_SHAP_Value': mean_abs_shap_values
})

# Sort by importance descending
global_feature_importance_df = global_feature_importance_df.sort_values(
    by='Mean_Absolute_SHAP_Value', ascending=False
).reset_index(drop=True)

# Print the table (core output for approach A)
print("\n--- Global Feature Importance (Ranked) ---")
print(global_feature_importance_df.to_string())  # to_string() ensures full printout

# Global feature importance bar plot (without value labels)
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Mean_Absolute_SHAP_Value',
    y='Feature',
    data=global_feature_importance_df,
    orient='h',
    palette='viridis'  # nice color palette
)
plt.title('Global Feature Importance (Mean Absolute SHAP Value)', fontsize=16)
plt.xlabel('Average Impact on Model Output Magnitude', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
# plt.savefig('figure_global_feature_importance_bar.png', dpi=300)
plt.show()

# Add value labels on bars
for i, v in enumerate(global_feature_importance_df['Mean_Absolute_SHAP_Value']):
    ax.text(v + 0.001, i, f'{v:.3f}', color='black', va='center')  # offset for label

plt.xlim(0, global_feature_importance_df['Mean_Absolute_SHAP_Value'].max() * 1.1)  # adjust x-axis for labels
plt.title('Global Feature Importance with Value Labels', fontsize=16)
plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
# plt.savefig('figure_global_feature_importance_bar_with_labels.png', dpi=300)
plt.show()

# SHAP summary beeswarm plot (unchanged)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot', fontsize=16)
plt.tight_layout()
# plt.savefig('figure_shap_summary_beeswarm.png', dpi=300)
plt.show()

# Define feature to dimension mapping
dimension_mapping = {
    'age': 'Demographics', 'sex': 'Demographics',
    'autonomy_freedom': 'Autonomy', 'autonomy_interesting': 'Autonomy', 'autonomy_options': 'Autonomy',
    'competence_matched': 'Competence', 'competence_capable': 'Competence', 'competence_competent': 'Competence',
    'related_important': 'Relatedness', 'related_fulfilling': 'Relatedness', 'related_not_close': 'Relatedness',
    'enjoyment_fun': 'Enjoyment', 'enjoyment_attention': 'Enjoyment', 'enjoymen_boring': 'Enjoyment', 'enjoyment_enjoyed': 'Enjoyment',
    'extrinsic_avoid': 'Extrinsic Motivation', 'extrinsic_forget': 'Extrinsic Motivation', 'extrinsic_compelled': 'Extrinsic Motivation', 'extrinsic_escape': 'Extrinsic Motivation',
    'Hours': 'Playtime'
}

# 2. Calculate mean absolute SHAP again (unchanged)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_names = X_test.columns
shap_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
})
shap_df['dimension'] = shap_df['feature'].map(dimension_mapping)

# 4. Aggregate importance by dimension
dimensional_importance = shap_df.groupby('dimension')['mean_abs_shap'].sum().sort_values(ascending=False)

# Print the dimensional importance table
print("\n--- Dimensional Importance (Ranked) ---")
print(dimensional_importance)

# 5. Visualize dimension importance with value labels
plt.figure(figsize=(12, 7))
ax_dim = sns.barplot(x=dimensional_importance.values, y=dimensional_importance.index, palette='viridis', orient='h')

# Add value labels
for i, v in enumerate(dimensional_importance.values):
    ax_dim.text(v + 0.01, i, f'{v:.3f}', color='black', va='center', fontweight='medium')

plt.xlim(0, dimensional_importance.max() * 1.15)  # adjust x-axis
plt.title('Aggregated Importance by Dimension with Value Labels', fontsize=16)
plt.xlabel('Total Mean Absolute SHAP Value (Contribution to Prediction)', fontsize=12)
plt.ylabel('Dimension', fontsize=12)
plt.tight_layout()
# plt.savefig('figure_dimensional_importance_with_labels.png', dpi=300)
plt.show()
```
