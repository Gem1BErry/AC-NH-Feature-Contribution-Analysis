# 假设 X_test 和 y_test 也已准备好
# 准备 X_test_grouped，添加分组信息
X_test_grouped = X_test.copy()
X_test_grouped['age_group'] = df.loc[X_test.index, 'age_group_stratify']

# 为了MLR模型，准备一个带有常数项的测试集
X_test_sm = sm.add_constant(X_test)

# --- 2. 定义模型字典，方便循环 ---
# 将模型和它们的名称存储在一个字典中
models = {
    "MLR": mlr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}
gender_groups = {1: 'Male', 2: 'Female', 3: 'Other'}
age_groups = sorted(X_test_grouped['age_group'].unique().tolist())

# --- 4. 循环遍历模型和子组，计算性能 (修正版) ---
performance_results = []

# 外层循环：遍历每一个模型
for model_name, model in models.items():
    print(f"--- Evaluating Model: {model_name} ---")

    # 内层循环：合并性别和年龄的评估逻辑
    all_groups = [(('Gender', sex_label), (X_test_grouped['sex'] == sex_code)) for sex_code, sex_label in gender_groups.items()] + \
                 [(('Age', age_label), (X_test_grouped['age_group'] == age_label)) for age_label in age_groups]

    for (group_type, group_name), subgroup_filter in all_groups:
        
        X_test_sub = X_test_grouped[subgroup_filter]
        y_test_sub = y_test[subgroup_filter]
        
        if len(X_test_sub) > 0:
            # --- 核心修正部分 ---
            if model_name == "MLR":
                # 1. 先只选取模型需要的特征列
                X_test_sub_features = X_test_sub[X_train.columns] # 使用 X_train 的列来确保一致性
                # 2. 然后再为这个子集添加常数项
                X_test_sub_final = sm.add_constant(X_test_sub_features, has_constant='add')
                # 3. 预测
                y_pred_sub = model.predict(X_test_sub_final)
            else:
                # scikit-learn模型只需要特征列
                X_test_sub_features = X_test_sub[X_features]
                y_pred_sub = model.predict(X_test_sub_features)
            
            mse = mean_squared_error(y_test_sub, y_pred_sub)
            r2 = r2_score(y_test_sub, y_pred_sub)
            
            performance_results.append({
                'Model': model_name,
                'Group Type': group_type,
                'Group Name': group_name,
                'N_Samples': len(X_test_sub),
                'MSE': mse,
                'R2': r2
            })

# (后面的 pivot_table 代码保持不变)
# ...
performance_df = pd.DataFrame(performance_results)
print("\n--- Performance Summary Table ---")
performance_pivot_r2 = performance_df.pivot_table(
    index=['Group Type', 'Group Name', 'N_Samples'],
    columns='Model',
    values='R2'
)
print(performance_pivot_r2.to_string(float_format="%.4f"))


# --- 1. 数据准备 (您的原始代码) ---
df['age_group_stratify'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 80],
                                  labels=['0-20', '21-30', '31-40', '41-50', '51+'])
df['stratify_col'] = df['sex'].astype(str) + '_' + df['age_group_stratify'].astype(str)

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

# --- 2. 数据划分 (您的原始代码) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df['stratify_col']
)

# ==============================================================================
#                      新增的步骤：创建 train_df
# ==============================================================================
# 将训练集的X和y合并成一个DataFrame，以供formula API使用
train_df = X_train.join(y_train)


# --- 3. 交互项回归分析 (我之前提供的代码) ---
# 现在 train_df 存在了，下面的代码可以正常运行

# 3.1 正确创建字符串格式的性别变量
def map_sex_to_str(sex_code):
    if sex_code == 1: return 'Male'
    elif sex_code == 2: return 'Female'
    elif sex_code >= 3: return 'Other'
    return None

train_df['sex_str'] = train_df['sex'].apply(map_sex_to_str)

# 3.2 检验所有特征与性别的交互作用 (完整模型)
print("\n--- Running Full Interaction Model for All Features and Gender ---")

# 准备特征列表
all_features_no_demo = [f for f in X_features if f not in ['age', 'sex']]
main_terms = ' + '.join(all_features_no_demo)

# 构建交互项部分，明确指定 'Female' 为基准组
interaction_terms = ' + '.join([f + ":C(sex_str, Treatment(reference='Female'))" for f in all_features_no_demo])

# 构建完整的公式
# 注意：在公式中，因变量名必须和DataFrame中的列名完全一致，即 'happiness_value'
formula_full = f"happiness_value ~ {main_terms} + C(sex_str, Treatment(reference='Female')) + {interaction_terms}"

# 拟合完整交互模型
full_interaction_model = smf.ols(formula_full, data=train_df).fit(cov_type='HC3')

print("\n--- Full Interaction Model for All Features with 'Female' as Reference ---")
print(full_interaction_model.summary())

results_df = pd.DataFrame({
    'coef': full_interaction_model.params,
    'std_err': full_interaction_model.bse,
    't_value': full_interaction_model.tvalues,
    'p_value': full_interaction_model.pvalues,
    'conf_lower': full_interaction_model.conf_int()[0],
    'conf_upper': full_interaction_model.conf_int()[1]
})

# 保存到CSV文件
results_df.to_csv('/content/drive/My Drive/Thesis/second/gender_interaction.csv')

print("Full interaction model summary saved to 'full_interaction_model_summary.csv'.")

train_df['age_group'] = df.loc[train_df.index, 'age_group_stratify']

# 检查一下 age_group 是否成功添加
print("--- age_group added to train_df. Value counts: ---")
print(train_df['age_group'].value_counts())


# --- 2. 使用交互项回归，并明确指定基准组 ---
# 选择一个基准组。通常选择样本量最大或最有代表性的组。
# 从上面的 value_counts() 结果看，'21-30' 组通常是最大的，我们选它。
# C(age_group, Treatment(reference='21-30')) 
# 这会把 '21-30' 岁年龄组设为对照组/基准组。

# 准备特征列表 (不包含人口统计学变量)
all_features_no_demo = [f for f in X_features if f not in ['age', 'sex']]
main_terms = ' + '.join(all_features_no_demo)

# 构建与年龄组的交互项
# 注意：列名中可能包含'-'，在formula中最好用 Q() 包裹起来，或者替换掉
# 我们先创建一个“干净”的列名以避免潜在问题
train_df['age_group_clean'] = train_df['age_group'].str.replace('-', '_').astype(str)
# 现在使用 age_group_clean
base_age_group = '21_30' # 新的基准组名称

interaction_terms_age = ' + '.join([f + f":C(age_group_clean, Treatment(reference='{base_age_group}'))" for f in all_features_no_demo])

# 构建完整的公式
formula_full_age = f"happiness_value ~ {main_terms} + C(age_group_clean, Treatment(reference='{base_age_group}')) + {interaction_terms_age}"

# --- 3. 拟合模型并打印结果 ---
print("\n--- Running Full Interaction Model for All Features and Age Group ---")
full_interaction_model_age = smf.ols(formula_full_age, data=train_df).fit(cov_type='HC3')

print(f"\n--- Full Interaction Model for All Features with '{base_age_group}' as Reference ---")
print(full_interaction_model_age.summary())

import pandas as pd

# 构造包含系数、标准误、t值、p值和置信区间的DataFrame
results_age_df = pd.DataFrame({
    'coef': full_interaction_model_age.params,
    'std_err': full_interaction_model_age.bse,
    't_value': full_interaction_model_age.tvalues,
    'p_value': full_interaction_model_age.pvalues,
    'conf_lower': full_interaction_model_age.conf_int()[0],
    'conf_upper': full_interaction_model_age.conf_int()[1]
})

# 导出到CSV文件
results_age_df.to_csv('/content/drive/My Drive/Thesis/second/age_interaction.csv')

print("Full interaction model (age) summary saved to 'full_interaction_model_age_summary.csv'.")

