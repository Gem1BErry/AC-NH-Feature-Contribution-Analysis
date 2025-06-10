```python
from statsmodels.regression.mixed_linear_model import MixedLM

# Define predictors and outcome variable
X_columns = [
    "autonomy_freedom", "autonomy_interesting", "autonomy_options",
    "competence_matched", "competence_capable", "competence_competent",
    "related_important", "related_fulfilling", "related_not_close",
    "enjoyment_fun", "enjoyment_attention", "enjoymen_boring", "enjoyment_enjoyed",
    "extrinsic_avoid", "extrinsic_forget", "extrinsic_compelled", "extrinsic_escape"
]
y_column = 'happiness_value'

# Map gender codes to labels
gender_labels = {
    1: 'Male',
    2: 'Female',
    3: 'Non-binary/Prefer not to say',
    4: 'Non-binary/Prefer not to say'
}
df['GenderGroup'] = df['sex'].map(gender_labels)

# Create interaction formula for gender
X_formula = " + ".join(X_columns)
interaction_terms = " + ".join([f'{col}:GenderGroup' for col in X_columns])
formula_with_interaction = f"{y_column} ~ {X_formula} + GenderGroup + {interaction_terms}"

# Fit the hierarchical linear model (HLM) with gender interaction
model_with_interaction = MixedLM.from_formula(formula_with_interaction, data=df, groups=df["GenderGroup"])
result_with_interaction = model_with_interaction.fit()

# Extract interaction terms with p-values less than 0.05
coefficients = result_with_interaction.params
p_values = result_with_interaction.pvalues
interaction_variables = [var for var in coefficients.index if 'GenderGroup' in var]
significant_interactions = [var for var in interaction_variables if p_values[var] < 0.05]

# Manually construct full slopes per gender group from main effect + interaction term
base_vars = ['competence_capable', 'competence_competent', 'related_not_close', 'extrinsic_escape']

gender_map = {
    'Female': '',
    'Male': ':GenderGroup[T.Male]',
    'Non-binary': ':GenderGroup[T.Non-binary/Prefer not to say]'
}

plot_data = []

for var in base_vars:
    for gender, suffix in gender_map.items():
        var_name = var if suffix == '' else f'{var}{suffix}'
        row = df[df['Variable'] == var_name]
        if not row.empty:
            coef = row['Coefficient'].values[0]
            plot_data.append({'Variable': var, 'Gender': gender, 'Coefficient': coef})

plot_df = pd.DataFrame(plot_data)

# Create age group categories
bins = [0, 20, 30, 40, 50, 80]
labels = ['0-20', '21-30', '31-40', '41-50', '51+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

# Create interaction formula for age group
interaction_terms = " + ".join([f'{col}:age_group' for col in X_columns])
formula_with_interaction = f"{y_column} ~ {X_formula} + age_group + {interaction_terms}"

# Fit the HLM with age group interaction
model_with_interaction = MixedLM.from_formula(formula_with_interaction, data=df, groups=df["age_group"])
result_with_interaction = model_with_interaction.fit()

# Extract significant interaction terms for age group
coefficients = result_with_interaction.params
p_values = result_with_interaction.pvalues
interaction_variables = [var for var in coefficients.index if 'age_group' in var]
significant_interactions = [var for var in interaction_variables if p_values[var] < 0.05]

# List of variables with significant moderation by age group
base_vars = [
    'autonomy_interesting', 'autonomy_options', 'competence_matched',
    'related_fulfilling', 'related_not_close', 'enjoyment_attention', 'enjoymen_boring'
]

age_group_map = {
    '0-20': '',
    '21-30': ':age_group[T.21-30]',
    '31-40': ':age_group[T.31-40]',
    '41-50': ':age_group[T.41-50]',
    '51+': ':age_group[T.51+]'
}

plot_data = []

for var in base_vars:
    for age_label, suffix in age_group_map.items():
        var_name = var if suffix == '' else f'{var}{suffix}'
        row = df[df['Variable'] == var_name]
        if not row.empty:
            coef = row['Coefficient'].values[0]
            plot_data.append({'Variable': var, 'AgeGroup': age_label, 'Coefficient': coef})

plot_df = pd.DataFrame(plot_data)

