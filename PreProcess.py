import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("Instagram—评论表.xlsx")

# 设置显示所有列
pd.set_option('display.max_columns', None)

# 设置显示所有行（如果你只看前几行，这个可不改）
# pd.set_option('display.max_rows', None)

# 打印前 5 行，完整表头会显示
print(df.head())
