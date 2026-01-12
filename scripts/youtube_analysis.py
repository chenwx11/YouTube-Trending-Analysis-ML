import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib

# 1.环境配置与路径设置
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置Pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 创建结果保存目录
output_dir = 'YouTube_Report_Assets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义数据输入路径
csv_path = 'E:/archive/USvideos.csv'
json_path = 'E:/archive/US_category_id.json'

# 2.原始数据加载与结构展示
df = pd.read_csv(csv_path)

print("\n" + "="*20 + " [阶段 A] 原始数据集字段清单 " + "="*20)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# 3.数据清洗：处理缺失值
# description字段存在空值，填充为预设字符串以保持数据完整性
df['description'] = df['description'].fillna('No description')

# 4.分类映射与时间维度计算

# 解析JSON维表，构建分类映射字典
with open(json_path, 'r') as f:
    categories = json.load(f)
cat_dict = {int(item['id']): item['snippet']['title'] for item in categories['items']}

# 将category_id映射为具体的业务分类名称
df['category_name'] = df['category_id'].map(cat_dict).fillna('Other')

# 时间字段标准化
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time']).dt.tz_localize(None)

# 计算视频发布至登上热门的天数差 (days_to_trend)
df['days_to_trend'] = (df['trending_date'] - df['publish_time']).dt.days

# 5.处理后结构展示与对比
print("\n" + "="*20 + " [阶段 B] 转换后新增业务字段 " + "="*20)
print(f">> 新增字段: category_name (分类语义化)")
print(f">> 新增字段: days_to_trend (上榜时效性)")

print("\n" + "="*20 + " [完整结构] 处理后全量字段清单 " + "="*20)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# 6.数据可视化分析
print("\n正在生成可视化图表...")

# [图 1] 热门视频各分类上榜数量分布
plt.figure(figsize=(10, 6))
df['category_name'].value_counts().plot(kind='bar', color='steelblue')
plt.title('YouTube 各类别热门视频上榜数量')
plt.xlabel('视频类别')
plt.ylabel('视频数量')
plt.savefig(f'{output_dir}/01_category_counts.png', dpi=300, bbox_inches='tight')
plt.close()

# [图 2] 各分类平均上榜速度对比
plt.figure(figsize=(10, 6))
df.groupby('category_name')['days_to_trend'].mean().sort_values().plot(kind='barh', color='seagreen')
plt.title('各类别视频平均上榜所需天数')
plt.xlabel('平均天数')
plt.ylabel('视频类别')
plt.savefig(f'{output_dir}/02_viral_speed.png', dpi=300, bbox_inches='tight')
plt.close()

# [图 3] 观看量与点赞数的相关性探索
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(1000), x='views', y='likes', hue='category_name', alpha=0.6)
plt.title('视频观看量与点赞数的关系')
plt.xlabel('观看量')
plt.ylabel('点赞数')
plt.savefig(f'{output_dir}/03_views_vs_likes.png', dpi=300, bbox_inches='tight')
plt.close()

# [图 4] 热门视频标题爆词分析 (词云图)
text = " ".join(str(title) for title in df.title)
wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
plt.figure(figsize=(15, 8))
plt.imshow(wc.to_image(), interpolation='bilinear')
plt.axis("off")
plt.savefig(f'{output_dir}/04_wordcloud.png', dpi=300)
plt.close()

# 7.数据导出
# 保存清洗并重构后的数据集
save_path = os.path.join(output_dir, 'youtube_processed_data.csv')
df.to_csv(save_path, index=False, encoding='utf-8-sig')
