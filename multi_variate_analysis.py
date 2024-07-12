import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model_size", type=str, default="160m")
parser.add_argument("--dataset_name", type=str, default="Pile-CC", choices=["ArXiv", "DM Mathematics",
                 "FreeLaw", "Github",  "HackerNews", "NIH ExPorter",
                "Pile-CC", "PubMed Abstracts", "PubMed Central", "StackExchange",
                "USPTO Backgrounds", "Wikipedia (en)", "WikiMIA", "all"])
parser.add_argument("--cuda", type=int, default=0, help="cuda device")
parser.add_argument("--skip_calculation", type=str, default="True")
parser.add_argument("--reference_model", type=str, default="True")
parser.add_argument("--samples", type=int, default=5000)
args = parser.parse_args()

dataset_name = args.dataset_name
loss_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_loss_dict.pkl", "rb"))
prob_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_prob_dict.pkl", "rb"))
ppl_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_ppl_dict.pkl", "rb"))
mink_plus_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_mink_plus_dict.pkl", "rb"))
zlib_dict = pickle.load(open(f"feature_result/{dataset_name}_{args.model_size}_zlib_dict.pkl", "rb"))
scaler = StandardScaler()
loss_train_std = scaler.fit_transform(np.array(loss_dict[dataset_name]["train"]).reshape(-1, 1))
loss_test_std = scaler.fit_transform(np.array(loss_dict[dataset_name]["test"]).reshape(-1, 1))
prob_train_std = scaler.fit_transform(np.array(prob_dict[dataset_name]["train"]).reshape(-1, 1))
prob_test_std = scaler.fit_transform(np.array(prob_dict[dataset_name]["test"]).reshape(-1, 1))
ppl_train_std = scaler.fit_transform(np.array(ppl_dict[dataset_name]["train"]).reshape(-1, 1))
ppl_test_std = scaler.fit_transform(np.array(ppl_dict[dataset_name]["test"]).reshape(-1, 1))
mink_plus_train_std = scaler.fit_transform(np.array(mink_plus_dict[dataset_name]["train"]).reshape(-1, 1))
mink_plus_test_std = scaler.fit_transform(np.array(mink_plus_dict[dataset_name]["test"]).reshape(-1, 1))
zlib_train_std = scaler.fit_transform(np.array(zlib_dict[dataset_name]["train"]).reshape(-1, 1))
zlib_test_std = scaler.fit_transform(np.array(zlib_dict[dataset_name]["test"]).reshape(-1, 1))

combined_data_train_vertical = np.hstack((loss_train_std, zlib_train_std, ppl_train_std)).T
combined_data_test_vertical = np.hstack((loss_test_std, zlib_test_std, ppl_test_std)).T

added_data_train = loss_train_std + zlib_train_std + ppl_train_std
added_data_test = loss_test_std + zlib_test_std + ppl_test_std
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制训练数据的归一化直方图
ax.hist(added_data_train, bins=1000, color='blue', alpha=0.6, label='Train Data', density=True)

# 绘制测试数据的归一化直方图
ax.hist(added_data_test, bins=1000, color='red', alpha=0.6, label='Test Data', density=True)

# 设置标题和标签
ax.set_title('Normalized Distribution of Added Data', fontsize=15)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)

# 添加图例
ax.legend()

# 保存图像
plt.savefig("normalized_added_data_distribution.png")

# 展示图像
plt.show()
#
# # 创建图形
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制训练数据的3D散点图
# ax.scatter(combined_data_train.T[:,0], combined_data_train.T[:,0], combined_data_train.T[:,0],
#            c='blue', alpha=0.3, label='Train Data', marker='o')
#
# # 绘制测试数据的3D散点图
# ax.scatter(combined_data_test.T[:,0], combined_data_test.T[:,0], combined_data_test.T[:,0],
#            c='red', alpha=0.3, label='Test Data', marker='^')
#
# # 设置标题和标签
# ax.set_title('3D Distribution of Train and Test Data')
# ax.set_xlabel('Loss')
# ax.set_ylabel('Zlib')
# ax.set_zlabel('Ppl')
#
# # 添加图例
# ax.legend()
#
# # 保存图像
# plt.savefig("3d_distribution_large.png")
#
# # 展示图像
# plt.show()