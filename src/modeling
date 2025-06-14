import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import statsmodels.api as sm

# --- 性别合并处理 ---
def recode_sex(x):
    if x == 1:
        return 1  # Male
    elif x == 2:
        return 2  # Female
    else:
        return 3  # Non-binary / Prefer not to say 合并3和4

df['sex'] = df['sex'].apply(recode_sex)

# --- 创建分层抽样列，包含年龄组和合并后的性别 ---
df['age_group_stratify'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 80],
                                  labels=['0-20', '21-30', '31-40', '41-50', '51+'])
df['stratify_col'] = df['sex'].astype(str) + '_' + df['age_group_stratify'].astype(str)

# --- 定义特征和目标变量 ---
X_features = [
    'age', 'sex',
    'autonomy_freedom', 'autonomy_interesting', 'autonomy_options',
    'competence_matched', 'competence_capable', 'competence_competent',
    'related_important', 'related_fulfilling', 'related_not_close',
    'enjoyment_fun', 'enjoyment_attention', 'enjoymen_boring', 'enjoyment_enjoyed',
    'extrinsic_avoid', 'extrinsic_forget', 'extrinsic_compelled', 'extrinsic_escape',
    'Hours'
]
X = df[X_features]
y = df['happiness_value']

# --- 数据划分：训练集80%，测试集20% ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df['stratify_col']
)

print(f"Train size: {len(X_train)}; Test size: {len(X_test)}")

# --- 1. MLR模型训练及稳健标准误估计 ---
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

mlr_model = sm.OLS(y_train, X_train_sm).fit(cov_type='HC3')

y_pred_mlr = mlr_model.predict(X_test_sm)

mse_mlr = mean_squared_error(y_test, y_pred_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)

print(f"MLR Test MSE: {mse_mlr:.4f}")
print(f"MLR Test R²: {r2_mlr:.4f}")

# --- 2. 随机森林训练及预测 ---
rf_model = RandomForestRegressor(
    n_estimators=250,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=5,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Test MSE: {mse_rf:.4f}")
print(f"Random Forest Test R²: {r2_rf:.4f}")

# --- 3. XGBoost训练及预测 ---
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=1,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Test MSE: {mse_xgb:.4f}")
print(f"XGBoost Test R²: {r2_xgb:.4f}")

# --- 汇总结果 ---
results = pd.DataFrame({
    'Model': ['MLR', 'Random Forest', 'XGBoost'],
    'Test MSE': [mse_mlr, mse_rf, mse_xgb],
    'Test R²': [r2_mlr, r2_rf, r2_xgb]
})
print("\nFinal Test Results:")
print(results)
