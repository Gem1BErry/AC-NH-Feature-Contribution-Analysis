```python
# Convert SHAP values to a DataFrame with test set index and feature columns
shap_df = pd.DataFrame(shap_values, index=X_test.index, columns=X_test.columns)

# Prepare analysis DataFrame including features and group info for easier filtering
analysis_df = X_test.copy()
analysis_df['sex_code'] = df.loc[X_test.index, 'sex']  # Encoded gender
analysis_df['sex_label'] = analysis_df['sex_code'].map({1: 'Male', 2: 'Female', 3: 'Non-binary/Other'})
analysis_df['age_group_stratify'] = df.loc[X_test.index, 'age_group_stratify']

# Calculate mean absolute SHAP values by gender group
sex_importance_dfs = {}
available_sex_labels = analysis_df['sex_label'].dropna().unique()

for sex_label in available_sex_labels:
    indices = analysis_df[analysis_df['sex_label'] == sex_label].index
    if len(indices) == 0:
        continue
    group_shap = shap_df.loc[indices]
    mean_abs_shap = group_shap.abs().mean(axis=0)
    sex_importance_dfs[sex_label] = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values('Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

# Combine gender group results into one DataFrame for comparison
all_sex_importance = []
for sex_label, df_importance in sex_importance_dfs.items():
    temp = df_importance.copy()
    temp['Group'] = sex_label
    all_sex_importance.append(temp)

if all_sex_importance:
    all_sex_df = pd.concat(all_sex_importance)
    wide_sex_table = all_sex_df.pivot_table(index='Feature', columns='Group', values='Mean_Absolute_SHAP')
    overall_sex_importance = wide_sex_table.mean(axis=1)
    wide_sex_table = wide_sex_table.loc[overall_sex_importance.sort_values(ascending=False).index]
    print(wide_sex_table.to_string())
else:
    print("No gender groups with sufficient data to analyze.")

# Calculate mean absolute SHAP values by age group
age_labels = sorted(analysis_df['age_group_stratify'].dropna().unique().astype(str))
age_importance_dfs = {}

for label in age_labels:
    indices = analysis_df[analysis_df['age_group_stratify'] == label].index
    if len(indices) == 0:
        continue
    group_shap = shap_df.loc[indices]
    mean_abs_shap = group_shap.abs().mean(axis=0)
    age_importance_dfs[label] = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values('Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

# Combine age group results into one DataFrame for comparison
all_age_importance = []
for label, df_importance in age_importance_dfs.items():
    temp = df_importance.copy()
    temp['Group'] = label
    all_age_importance.append(temp)

if all_age_importance:
    all_age_df = pd.concat(all_age_importance)
    wide_age_table = all_age_df.pivot_table(index='Feature', columns='Group', values='Mean_Absolute_SHAP')
    overall_age_importance = wide_age_table.mean(axis=1)
    wide_age_table = wide_age_table.loc[overall_age_importance.sort_values(ascending=False).index]
    print(wide_age_table.to_string())
else:
    print("No age groups with sufficient data to analyze.")

# Plot heatmap of gender group feature importance
plt.figure(figsize=(8, 12))
sns.heatmap(
    wide_sex_table,
    annot=True,
    fmt=".3f",
    linewidths=0.5,
    cmap='viridis',
    cbar_kws={'label': 'Mean Absolute SHAP Value'}
)
plt.title('Feature Importance Heatmap Across Gender Groups', fontsize=16, pad=20)
plt.xlabel('Gender Group', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(fontsize=9)
plt.savefig('gender_importance_heatmap.svg', format='svg', bbox_inches='tight')
plt.savefig('gender_importance_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot heatmap of age group feature importance
plt.figure(figsize=(10, 12))
sns.heatmap(
    wide_age_table,
    annot=True,
    fmt=".3f",
    linewidths=0.5,
    cmap='magma',
    cbar_kws={'label': 'Mean Absolute SHAP Value'}
)
plt.title('Feature Importance Heatmap Across Age Groups', fontsize=16, pad=20)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(fontsize=9)
plt.savefig('age_importance_heatmap.svg', format='svg', bbox_inches='tight')
plt.show()
```
