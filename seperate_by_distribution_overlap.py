import pickle
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from scipy.stats import ks_2samp


["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc","pubmed_central", "wikipedia_(en)", "full_pile"]
dataset_name = "github"
model_size = "12b"

loss_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_zlib_dict.pkl", "rb"))

member_data = np.array(loss_dict[dataset_name]["member"])
nonmember_data = np.array(loss_dict[dataset_name]["nonmember"])


# 估计概率密度函数
member_kde = gaussian_kde(member_data)
nonmember_kde = gaussian_kde(nonmember_data)

# 合并数据，以便统一处理
all_data = np.concatenate([member_data, nonmember_data])

# 计算所有数据在member和nonmember分布下的概率
member_probs = member_kde(all_data)
nonmember_probs = nonmember_kde(all_data)

# 判定那些容易误分类的数据点
error_indices = np.where(member_probs < nonmember_probs)[0]

# 获取需要保留的数据点
member_indices = np.where(member_probs >= nonmember_probs)[0]
nonmember_indices = np.where(nonmember_probs >= member_probs)[0]

filtered_member_data = all_data[np.intersect1d(member_indices, np.where(all_data <= np.max(member_data)))]
filtered_nonmember_data = all_data[np.intersect1d(nonmember_indices, np.where(all_data > np.min(nonmember_data)))]

print(f"原始member数据大小: {len(member_data)}, 筛选后member数据大小: {len(filtered_member_data)}")
print(f"原始nonmember数据大小: {len(nonmember_data)}, 筛选后nonmember数据大小: {len(filtered_nonmember_data)}")
print(f"原始member数据大小: {len(member_data)}, 筛选后member数据大小: {len(filtered_member_data)}")
print(f"原始nonmember数据大小: {len(nonmember_data)}, 筛选后nonmember数据大小: {len(filtered_nonmember_data)}")
plt.hist(filtered_member_data, bins=50, alpha=0.5, label="member")
# gaussian_kde(filtered_member_data).set_bandwidth(bw_method=0.25)
#
plt.hist(filtered_nonmember_data, bins=50, alpha=0.5, label="nonmember")
# gaussian_kde(filtered_nonmember_data).set_bandwidth(bw_method=0.25)

#sns.kdeplot(filtered_member_data, bw_method=0.25, label="member")
#sns.kdeplot(filtered_nonmember_data, bw_method=0.25, label="nonmember")
plt.legend()
plt.show()
# # 计算均值和标准差
# member_mean, member_std = np.mean(member_data), np.std(member_data)
