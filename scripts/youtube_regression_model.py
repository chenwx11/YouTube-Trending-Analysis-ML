import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib
import os

#1.环境与保存路径设置
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = 'YouTube_ML_Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2.加载数据
file_path = 'YouTube_Report_Assets/youtube_processed_data.csv'
df = pd.read_csv(file_path)

# 3.特征工程
# 选择用来预测的指标
# 点赞数、评论数、标题长度、上榜速度、所属分类 是否能决定播放量
feature_cols = ['likes', 'comment_count', 'category_name', 'days_to_trend']

X = df[feature_cols].copy()
y = df['views']

# 将文字分类转为数字
le = LabelEncoder()
X['category_name'] = le.fit_transform(X['category_name'].astype(str))

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4.训练回归模型
print("模型炼制中... 正在预测...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5.模型评估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- 模型评估报告 ---")
print(f"R2 Score (解释力得分): {r2:.4f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")

# 6.结果可视化

# 【图 1】特征重要性
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values(ascending=True).plot(kind='barh', color='gold')
plt.title('哪些因素最能决定视频播放量？', fontsize=14)
plt.savefig(f'{output_dir}/01_feature_importance.png', dpi=300)
plt.show()

# 【图 2】预测值 真实值 (看模型准不准)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实播放量')
plt.ylabel('预测播放量')
plt.title('回归模型：预测值与真实值的偏差分布', fontsize=14)
plt.savefig(f'{output_dir}/02_prediction_scatter.png', dpi=300)
plt.show()