import deep_net_mc as tn
import numpy as np
import csv
import shap
epoch = 5
net, X, Y, X_eval, Y_eval = tn.train_speck_distinguisher(40, num_rounds=epoch, diff=(0x40, 0), group_size=2, depth=1)

shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
# 创建 SHAP DeepExplainer 实例，使用部分训练数据作为背景数据
explainer = shap.KernelExplainer(net.predict, shap.sample(X, 100))  # 这里的 X_train[:100] 是背景数据集

# 计算 SHAP 值，解释前 10 个测试样本
shap_values = explainer.shap_values(X_eval[:10])

# 可视化 SHAP 值
shap.initjs()  # 初始化 JavaScript 可视化
shap.summary_plot(shap_values, X_eval[:10])  # 显示 summary_plot

import pandas as pd
shap_values_array_0 = shap_values[0]  # 二分类，选择一个类的 SHAP 值
# shap_values_array_1 = shap_values[1]
# 将 SHAP 值与对应的特征合并
num_samples, num_features = shap_values_array_0.shape
shap_0_df = pd.DataFrame(shap_values_array_0, columns=[f"Feature_{i}" for i in range(num_features)])
# shap_1_df = pd.DataFrame(shap_values_array_1, columns=[f"Feature_{i}" for i in range(num_features)])
original_df = pd.DataFrame(X_eval[:10], columns=[f"Feature_{i}" for i in range(num_features)])

# 添加样本 ID 列
shap_0_df["Sample ID"] = range(1, num_samples + 1)
# shap_1_df["Sample ID"] = range(1, num_samples + 1)
original_df["Sample ID"] = range(1, num_samples + 1)

# 转换为长格式，将每个特征展开
shap_0_long = pd.melt(shap_0_df, id_vars=["Sample ID"], var_name="Feature", value_name="SHAP Value")
# shap_1_long = pd.melt(shap_1_df, id_vars=["Sample ID"], var_name="Feature", value_name="SHAP Value 1")
original_long = pd.melt(original_df, id_vars=["Sample ID"], var_name="Feature", value_name="Original Value")

# 合并 SHAP 值和原始值
combined_long = pd.merge(shap_0_long, original_long, on=["Sample ID", "Feature"])
# combined_long = pd.merge(original_long, combined_long, on=["Sample ID", "Feature"])
combined_long["Feature Number"] = combined_long["Feature"].str.extract(r'(\d+)', expand=False).astype(int)
combined_long = combined_long.sort_values(by=["Sample ID", "Feature Number"]).reset_index(drop=True)

# 删除辅助列 "Feature Number"（如果不需要）
combined_long = combined_long.drop(columns=["Feature Number"])

# 保存到 CSV
combined_long.to_csv("Chen_shap_values_with_10sample_6_round.csv", index=False)
print("done")