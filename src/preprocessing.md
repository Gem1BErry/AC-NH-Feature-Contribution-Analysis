# Load gameplay duration data, rename columns, convert seconds to hours, and aggregate total hours per player rounded to 2 decimals
gt = pd.read_csv('telem_data.txt', sep='\t')[['hashed_id', 'duration']]
gt = gt.rename(columns={'hashed_id': 'code', 'duration': 'Hours'})
gt['Hours'] = gt['Hours'] / 3600
gt = gt.groupby('code', as_index=False)['Hours'].sum().round(2)
# Calculate mean and std deviation of hours, detect outliers beyond ±6 std dev and replace them with NaN
mean_hours = gt['Hours'].mean()
std_hours = gt['Hours'].std()
lower_bound = mean_hours - 6 * std_hours
upper_bound = mean_hours + 6 * std_hours
gt['Hours'] = np.where((gt['Hours'] < lower_bound) | (gt['Hours'] > upper_bound), np.nan, gt['Hours'])
# Rescale hours to a range between 1 and 7 using Min-Max normalization
gt['Hours'] = 1 + 6 * (gt['Hours'] - gt['Hours'].min()) / (gt['Hours'].max() - gt['Hours'].min())
# Load main survey data and merge with gameplay hours by player code
df = pd.read_csv('RawData.csv')
df = pd.merge(df, gt, on='code', how='left')
# Drop irrelevant or redundant columns by their index positions
cols_to_drop = [0, 2, 5] + list(range(6, 20)) + list(range(37, 42))
df.drop(df.columns[cols_to_drop], axis=1, inplace=True)
# Filter out rows with too many missing values or missing key demographic info (sex or age)
missing_values_count = df.isna().sum(axis=1)
df_cleaned = df[~((missing_values_count > len(df.columns) / 2) | df[['sex', 'age']].isna().any(axis=1))].copy()
# Extract numeric columns for imputation
numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
numeric_df = df_cleaned[numeric_cols].copy()
# Impute missing values using Iterative Imputer with Decision Tree Regressor
imputer = IterativeImputer(estimator=DecisionTreeRegressor(max_depth=5, random_state=61), max_iter=30, random_state=61)
imputed_data = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
# Calculate happiness score as difference between means of positive and negative SPANE items
pos_emo = ['spane_acnh_positive', 'spane_acnh_good', 'spane_acnh_pleasant', 'spane_acnh_happy', 'spane_acnh_joyful', 'spane_acnh_contented']
neg_emo = ['spane_acnh_negative', 'spane_acnh_bad', 'spane_acnh_unpleasant', 'spane_acnh_sad', 'spane_acnh_afraid', 'spane_acnh_angry']
imputed_data['happiness_value'] = imputed_data[pos_emo].mean(axis=1) - imputed_data[neg_emo].mean(axis=1)
# Round most float columns to integers except the last two columns, which are rounded to 2 decimals
for col in imputed_data.columns[:-2]:
    if pd.api.types.is_float_dtype(imputed_data[col]):
        imputed_data[col] = np.round(imputed_data[col]).astype(int)
imputed_data.iloc[:, -2:] = np.round(imputed_data.iloc[:, -2:], 2)
# Generate random 4-character player IDs consisting of letters and digits, set random seed for reproducibility
def random_id(n=4):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=n))
np.random.seed(61)
player_ids = [random_id(4) for _ in range(imputed_data.shape[0])]
imputed_data.insert(0, 'player_id', player_ids)
