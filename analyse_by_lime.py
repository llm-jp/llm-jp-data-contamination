import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

# 加载并合并数据
dataset_idxs = [0, 1, 2]
truncated_results_list = []
relative_results_list = []
untruncated_results_list = []

for idx in dataset_idxs:
    df_truncated = pd.read_csv(f"auc_results_absolute_truncated_all_length_all_model_size_{idx}.csv")
    df_truncated['seed'] = idx  # 添加种子列
    df_truncated["truncation"] = "Absolute Semantic Incomplete"
    truncated_results_list.append(df_truncated)

    df_relative = pd.read_csv(f"auc_results_relative_truncated_all_length_all_model_size_{idx}.csv")
    df_relative['seed'] = idx
    df_relative["truncation"] = "Relative Semantic Complete"
    relative_results_list.append(df_relative)

    df_untruncated = pd.read_csv(f"auc_results_absolute_untruncated_all_length_all_model_size_{idx}.csv")
    df_untruncated['seed'] = idx
    df_untruncated["truncation"] = "Absolute Semantic Complete"
    untruncated_results_list.append(df_untruncated)

# 合并数据框
truncated_results = pd.concat(truncated_results_list, ignore_index=True)
relative_results = pd.concat(relative_results_list, ignore_index=True)
untruncated_results = pd.concat(untruncated_results_list, ignore_index=True)
concated_results = pd.concat([truncated_results, relative_results, untruncated_results], ignore_index=True)

data = concated_results.copy()
selected_dataset = 'Wikipedia (en)'
selected_feature = 'Loss'

# 过滤出特定的数据集和特征
filtered_data = data[(data['dataset'] == selected_dataset) & (data['feature'] == selected_feature) & (data['truncation'] == "Absolute Semantic Complete")]

# One-Hot编码
one_hot_encoder = OneHotEncoder(sparse=False)
X_encoded = one_hot_encoder.fit_transform(filtered_data[['length', 'model_size', 'truncation']])
X_columns = one_hot_encoder.get_feature_names_out(['length', 'model_size', 'truncation'])
X = pd.DataFrame(X_encoded, columns=X_columns)

# 添加常数项
X = sm.add_constant(X)
y = filtered_data['auc']

# 拟合线性回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 获取特征重要性
importances = model.feature_importances_
feature_names = X_columns

# 输出特征重要性
print("Feature Importances:")
sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importances:
    print(f"{feature}: {importance}")

# 可视化特征重要性
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
sns.barplot(x=[imp for _, imp in sorted_importances], y=[feat for feat, _ in sorted_importances])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
