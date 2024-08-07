import pickle
from sklearn.decomposition import PCA
#from utils import *
from sklearn.metrics import roc_auc_score, silhouette_score, f1_score, davies_bouldin_score
from matplotlib import pyplot as plt
import numpy as np
dataset_names = ["arxiv", "dm_mathematics", "github", "hackernews", "pile_cc",
                      "pubmed_central", "wikipedia_(en)", "full_pile"]
model_size = "12b"
for dataset_name in dataset_names:
    loss_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_loss_dict.pkl", "rb"))
    prob_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_prob_dict.pkl", "rb"))
    ppl_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_ppl_dict.pkl", "rb"))
    mink_plus_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_mink_plus_dict.pkl", "rb"))
    zlib_dict = pickle.load(open(f"feature_result_online/{dataset_name}_{model_size}_zlib_dict.pkl", "rb"))
    aggregated_train = []
    aggregated_test = []
    aggregated_val = []
    for idx, dict in enumerate([loss_dict, prob_dict, ppl_dict, mink_plus_dict, zlib_dict]):
        for set_name in ["member", "nonmember"]:
            data = np.array(dict[dataset_name][set_name])
            data[np.isnan(data)] = data[~np.isnan(data)].mean()
            mean1, std1 = np.mean(data), np.std(data)
            normalized_value = (dict[dataset_name][set_name] - mean1) / std1
            normalized_value[np.isnan(normalized_value)] =  normalized_value[~np.isnan(normalized_value)].mean()
            if set_name == "member":
                aggregated_train.append(normalized_value.tolist())
            elif set_name == "nonmember":
                aggregated_test.append(normalized_value.tolist())

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 去除离群值
    train_no_outliers = aggregated_train
    test_no_outliers = aggregated_test
    val_no_outliers = aggregated_val

    member_embed_array = np.array(aggregated_train).T
    non_member_embed_array = np.array(aggregated_test).T
    # Concatenate for PCA
    all_embed_array = np.vstack([member_embed_array, non_member_embed_array])
    labels = np.array([1] * len(member_embed_array) + [0] * len(non_member_embed_array))
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embed_array)
    # Separate the results
    pca_member_embed = pca_result[labels == 1]
    pca_non_member_embed = pca_result[labels == 0]
    # Plotting
    plt.figure(figsize=(14, 8))
    plt.scatter(pca_member_embed[:, 0], pca_member_embed[:, 1], c='blue', label='Member Text', alpha=0.5)
    plt.scatter(pca_non_member_embed[:, 0], pca_non_member_embed[:, 1], c='red', label='Non-Member Text',
                alpha=0.5)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Member and Non-Member Embeddings')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'PCA_output_feature_{dataset_name}_{model_size}.png')
    plt.show()
    labels = np.array([1] * len(member_embed_array) + [0] * len(non_member_embed_array))
    X = np.vstack((member_embed_array, non_member_embed_array))
    db_index = davies_bouldin_score(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    print(dataset_name)
    print(db_index)
    print(silhouette_avg)

    # print(f"DB Index : ", db_index)
    # f.write(f"{dataset_name} DB Index : {db_index}\n")
    # print(f"Silhouette Score: ", silhouette_avg)
    # f.write(f"{dataset_name} Silhouette Score : {silhouette_avg}\n")






# 第一个子图：绘制KDE
# sns.kdeplot(train_no_outliers, ax=axs[0], label="train", alpha=0.5, bw_adjust=0.1, shade=True)
# sns.kdeplot(test_no_outliers, ax=axs[0], label="test", alpha=0.5, bw_adjust=0.1, shade=True)
# sns.kdeplot(val_no_outliers, ax=axs[0], label="val", alpha=0.5, bw_adjust=0.1, shade=True)
# axs[0].set_title(f'{dataset_name} KDE PDF at {model_size} model')
# axs[0].set_xlabel("Normalized Value")
# axs[0].set_ylabel('KDE PDF Value')
# axs[0].legend()
#
# # 第二个子图：绘制直方图
# bins = np.linspace(-2.5, 2.5, 300)  # 可以根据数据范围调整
# sns.histplot(train_no_outliers, ax=axs[1], label="train", bins=bins, alpha=0.5, stat="density", kde=False)
# sns.histplot(test_no_outliers, ax=axs[1], label="test", bins=bins, alpha=0.5, stat="density", kde=False)
# sns.histplot(val_no_outliers, ax=axs[1], label="val", bins=bins, alpha=0.5, stat="density", kde=False)
# axs[1].set_title(f'{dataset_name} Histogram PDF at {model_size} model')
# axs[1].set_xlabel("Normalized Value")
# axs[1].set_ylabel('Histogram PDF Value')
# axs[1].legend()
#
# # 调整布局并显示图形
# plt.tight_layout()
# plt.show()


