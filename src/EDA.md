# Data Preprocessing and Visualization
```python
gender_map = {
    1: 'Male',
    2: 'Female',
    3: 'Non-binary/Prefer not to say',
    4: 'Non-binary/Prefer not to say'
}
df['gender'] = df['sex'].map(gender_map)

bins = [0, 20, 30, 40, 50, 150]
labels = ['0-20', '21-30', '31-40', '41-50', '51+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

cont_vars = [
    "autonomy_freedom", "autonomy_interesting", "autonomy_options",
    "competence_matched", "competence_capable", "competence_competent",
    "related_important", "related_fulfilling", "related_not_close",
    "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
    "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape",
    "Hours", 'happiness_value'
]

for var in cont_vars:
    plt.figure(figsize=(8,4))
    sns.histplot(df[var].dropna(), kde=True)
    plt.title(f"Distribution of {var}")
    plt.show()

cat_vars = ['gender', 'age_group']

for var in cat_vars:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=var)
    plt.title(f"Distribution of {var}")
    plt.show()

plt.figure(figsize=(10,8))
corr = df[cont_vars].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix of Continuous Variables")
plt.show()

for var in ['Hours', 'happiness_value']:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[var])
    plt.title(f"Boxplot of {var}")
    plt.show()
