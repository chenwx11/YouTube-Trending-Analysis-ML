import pandas as pd
from sqlalchemy import create_engine

# 设置路径（先导美国站的数据：USvideos.csv）
FILE_PATH = 'E:/archive/USvideos.csv'
MY_PASSWORD = 'cwx.021228'
engine = create_engine(f'mysql+pymysql://root:{MY_PASSWORD}@127.0.0.1:3308/youtube_project')

# 读取并导入
df = pd.read_csv(FILE_PATH)
# 转换时间格式，方便 SQL 处理
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')

df.to_sql('raw_youtube_us', con=engine, if_exists='replace', index=False, chunksize=5000)
print("美国站视频数据导入成功！")