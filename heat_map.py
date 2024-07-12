import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 你的新数据
new_data = np.array([[0,        0.053, 0.032],
 [0.053, 0,        0.053       ],
 [0.033, 0.053,       0,       ]])

ax = sns.heatmap(new_data, cmap='hot', annot=True)

# 设置x轴和y轴的标签
labels = ['Train', 'Valid', 'Test']
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# 添加标题
plt.title('PPL')

plt.show()