import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

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
loss_train_std = scaler.fit(np.array(loss_dict[dataset_name]["train"]).reshape(-1, 1))
loss_test_std = scaler.transform(np.array(loss_dict[dataset_name]["test"]).reshape(-1, 1))
prob_train_std = scaler.fit(np.array(prob_dict[dataset_name]["train"]).reshape(-1, 1))
prob_test_std = scaler.transform(np.array(prob_dict[dataset_name]["test"]).reshape(-1, 1))
ppl_train_std = scaler.fit(np.array(ppl_dict[dataset_name]["train"]).reshape(-1, 1))
ppl_test_std = scaler.transform(np.array(ppl_dict[dataset_name]["test"]).reshape(-1, 1))
mink_plus_train_std = scaler.fit(np.array(mink_plus_dict[dataset_name]["train"]).reshape(-1, 1))
mink_plus_test_std = scaler.transform(np.array(mink_plus_dict[dataset_name]["test"]).reshape(-1, 1))
zlib_train_std = scaler.fit(np.array(zlib_dict[dataset_name]["train"]).reshape(-1, 1))
zlib_test_std = scaler.transform(np.array(zlib_dict[dataset_name]["test"]).reshape(-1, 1))

combined_data_train = np.vstack([loss_train_std, zlib_train_std, ppl_train_std]).T
combined_data_test = np.vstack([loss_test_std, zlib_test_std, ppl_test_std]).T

# 创建一个新的图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 估计训练数据的3D核密度
kde_train = gaussian_kde(combined_data_train.T)

# 生成一个坐标栅格
x = np.linspace(combined_data_train[:, 0].min() - 1,
                combined_data_train[:, 0].max() + 1, 50)
y = np.linspace(combined_data_train[:, 1].min() - 1,
                combined_data_train[:, 1].max() + 1, 50)
z = np.linspace(combined_data_train[:, 2].min() - 1,
                combined_data_train[:, 2].max() + 1, 50)
X, Y, Z = np.meshgrid(x, y, z)
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
density_train = np.reshape(kde_train(positions), X.shape)

# 绘制训练数据的3D密度图
ax.plot_surface(X[:, :, 0], Y[:, :, 0], density_train[:, :, 25], cmap='viridis', alpha=0.6)

# 估计测试数据的3D核密度
kde_test = gaussian_kde(combined_data_test.T)
density_test = np.reshape(kde_test(positions), X.shape)

# 绘制测试数据的3D密度图
ax.plot_wireframe(X[:, :, 0], Y[:, :, 0], density_test[:, :, 25], color='red', alpha=0.6)

# 设置标题和标签
ax.set_title('3D Density Estimation of Train and Test Data')
ax.set_xlabel('Loss')
ax.set_ylabel('Zlib')
ax.set_zlabel('Density')

# 添加图例
train_proxy = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.5)
test_proxy = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5)
ax.legend([train_proxy, test_proxy], ["Train Data", "Test Data"], loc='upper right')

# 保存图像
plt.savefig("3d_density_distribution.png")

# 展示图像
plt.show()