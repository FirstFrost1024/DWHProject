import pandas as pd

# 1. 读取 Excel 文件
df = pd.read_excel("weibo.xlsx")

# 2. 删除完全相同的行
df = df.drop_duplicates(keep="first")

# 3. 仅删除 微博正文 为空 或 只有空格 的行
df = df[df["微博正文"].notna()]                # 去掉 NaN
df = df[df["微博正文"].str.strip() != ""]      # 去掉空字符串

# 4. 保存结果
df.to_excel("weibo_cleaned.xlsx", index=False)

print("清洗完成，结果保存到 weibo_cleaned.xlsx")
