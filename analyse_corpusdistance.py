import pandas as pd
def modify_length_option(value):
    value_str = str(value)
    if '_' in value_str:
        # 按下划线分割
        prefix = int(value_str.split('_')[0])
        # 如果是数字，并且为5，改为0
        if int(prefix) == 5:
            return 0
        else:
            return int(prefix)
    else:
        return int(value_str)
corpus_similarity = pd.read_excel("0930_corpus_distance.xlsx", header=1, index_col=0)

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


corpus_similarity['length_option'] = corpus_similarity['length_option'].apply(modify_length_option)

corpus_similarity['group'] = corpus_similarity['class'].str.extract(r'(mia_dataset_[^_]+)')

# 首先按 'corpus' 和 'group' 分组，然后对其他列进行平均计算
grouped_means = corpus_similarity.groupby(['corpus', 'group', "length_option", "n_sample", 'ref_model']).mean().reset_index()

# 仅包括计算上面的聚合列
#grouped_means = grouped_means.drop(columns=['length_option', 'n_sample', 'ref_model'])  # 删除非数值列

print("Grouped means by corpus:")
print(grouped_means)
corpus_similarity_relative = grouped_means[grouped_means['group'] == 'mia_dataset_relative']
corpus_similarity_truncated = grouped_means[grouped_means['group'] == 'mia_dataset_truncated']
corpus_similarity_untruncated = grouped_means[grouped_means['group'] == 'mia_dataset_untruncated']

merged_df = pd.merge(grouped_means, concated_results, left_on='corpus', right_on='dataset')

# 列出所有需要分析的指标
correlation_results = {}
indicators = ['Classifier', 'PR', 'IRPR', 'DC', 'MAUVE', 'FID', 'Chi-squared', 'Zipf',
't-test', 'Medoid']
#'Classifier', 'PR', 'IRPR', 'DC', 'MAUVE', 'FID', 'Chi-squared', 'Zipf',
#'t-test', 'Medoid'
#analyse_dimension = ['feature', 'model_size', 'length']
analyse_dimension = ['model_size']
for (model_size), group_data in merged_df.groupby(analyse_dimension):
    # 初始化存储单元
    if (model_size) not in correlation_results:
        correlation_results[(model_size)] = {}

    # 计算每个指标的相关系数
    for indicator in indicators:
        # 确保组数据足够进行相关性分析（例如至少有2个数据点）
        if len(group_data) > 1:
            correlation = group_data[[indicator, 'auc']].corr().iloc[0, 1]
        else:
            correlation = None  # 数据点不足，无法计算相关系数

        correlation_results[(model_size)][indicator] = correlation

# 输出所有结果
for key, correlations in correlation_results.items():
    model_size = key
    print(f"\nModel Size: {model_size}")
    for indicator, correlation in correlations.items():
        print(f"  Correlation between {indicator} and AUC: {correlation}")